# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restoring a snapshot must not run attacker-chosen callables.

A stored snapshot can come from an untrusted place (a session store shipped in
a cloned repository). Its envelopes name classes, and its ``type_allowlist``
names which classes may be built, so both are attacker data. ``deserialize()``
must build only classes of the envelope's kind from code the process already
loaded, whatever the stored allowlist says.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import sys
import textwrap
from pathlib import Path

import pytest
from pydantic import BaseModel

from nooa import Agent
from nooa.errors.storage import DeserializationError
from nooa.storage.markers import snapshotable
from nooa.storage.serialization import deserialize, serialize
from nooa.storage.snapshot import AgentSnapshot, StaticContextBlock
from nooa.storage.snapshot_vars import SnapshotVars
from nooa.unifiedllm import FakeLLMClient

KINDS = ("dataclass", "pydantic", "dict_class")
SIDE_EFFECTS: list[str] = []


class SideEffectClass:
    """A plain class whose ``__init__`` has a side effect (no snapshot kind)."""

    def __init__(self, **kwargs: object) -> None:
        SIDE_EFFECTS.append("SideEffectClass")


def side_effect_function(**kwargs: object) -> None:
    SIDE_EFFECTS.append("side_effect_function")


@dataclasses.dataclass
class Point:
    x: int
    y: int = 0


class Model(BaseModel):
    name: str
    point: Point | None = None


@snapshotable
class Config:
    def __init__(self, host: str, port: int = 80) -> None:
        self.host = host
        self.port = port


class Color(enum.Enum):
    RED = "red"


def _fqn(obj: object) -> str:
    return f"{obj.__module__}.{obj.__qualname__}"


def _envelope(kind: str, fqn: str, data: dict) -> dict:
    return {"__type__": kind, "__class__": fqn, "data": data}


def _restore(blob: dict) -> object:
    # The stored allowlist is attacker data: it names the gadget class.
    return deserialize(blob, {blob["__class__"]})


@pytest.fixture(autouse=True)
def _clear_side_effects():
    SIDE_EFFECTS.clear()
    yield
    SIDE_EFFECTS.clear()


class TestEnvelopeKindIsEnforced:
    @pytest.mark.parametrize("kind", KINDS)
    def test_popen_is_refused(self, kind: str, tmp_path: Path) -> None:
        marker = tmp_path / "popen-ran"
        blob = _envelope(kind, "subprocess.Popen", {"args": ["touch", str(marker)]})
        with pytest.raises(DeserializationError, match=r"'subprocess.Popen' is not a"):
            _restore(blob)
        assert not marker.exists()

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize(
        ("fqn", "data"),
        [
            ("os.system", {"command": "true"}),
            ("builtins.eval", {"source": "1"}),
            ("builtins.exec", {"source": "pass"}),
            ("os.remove", {"path": "/nonexistent"}),
        ],
    )
    def test_functions_are_refused(self, kind: str, fqn: str, data: dict) -> None:
        with pytest.raises(DeserializationError, match="This blob may have been tampered with"):
            _restore(_envelope(kind, fqn, data))

    @pytest.mark.parametrize("kind", KINDS)
    @pytest.mark.parametrize("target", [SideEffectClass, side_effect_function])
    def test_plain_callables_with_side_effects_are_refused(self, kind: str, target) -> None:
        with pytest.raises(DeserializationError, match="may not build it"):
            _restore(_envelope(kind, _fqn(target), {}))
        assert SIDE_EFFECTS == []

    @pytest.mark.parametrize(
        ("kind", "target"),
        [
            ("dataclass", Model),
            ("dataclass", Config),
            ("pydantic", Point),
            ("pydantic", Config),
            ("dict_class", Point),
            ("dict_class", Model),
            ("dataclass", Point.__init__),
        ],
    )
    def test_wrong_kind_is_refused(self, kind: str, target) -> None:
        with pytest.raises(DeserializationError, match="may not build it"):
            _restore(_envelope(kind, _fqn(target), {"x": 1, "name": "n", "host": "h"}))

    def test_nested_gadget_is_refused(self, tmp_path: Path) -> None:
        marker = tmp_path / "nested-ran"
        gadget = _envelope("dataclass", "subprocess.Popen", {"args": ["touch", str(marker)]})
        blob = {"outer": [{"inner": gadget}]}
        with pytest.raises(DeserializationError, match="is not a dataclass"):
            deserialize(blob, {"subprocess.Popen"})
        assert not marker.exists()

    def test_snapshotable_dunder_attribute_is_refused(self) -> None:
        blob = _envelope("dict_class", _fqn(Config), {"host": "h", "__class__": "x"})
        with pytest.raises(DeserializationError, match="special attribute"):
            _restore(blob)


class TestNoNewModuleImports:
    def test_unloaded_package_is_not_imported(self, tmp_path: Path, monkeypatch) -> None:
        marker = tmp_path / "imported"
        pkg = tmp_path / "evil_snapshot_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text(
            textwrap.dedent(f"""
                from pathlib import Path
                Path({str(marker)!r}).write_text("ran")
                import dataclasses

                @dataclasses.dataclass
                class Gadget:
                    x: int = 0
            """)
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        blob = _envelope("dataclass", "evil_snapshot_pkg.Gadget", {"x": 1})
        with pytest.raises(DeserializationError, match="never imports"):
            _restore(blob)
        assert not marker.exists()
        assert "evil_snapshot_pkg" not in sys.modules

    def test_namespace_package_submodule_is_not_imported(self, tmp_path: Path, monkeypatch) -> None:
        # A namespace package's ``__path__`` follows ``sys.path``, so once it is
        # loaded, a directory of the same name anywhere on ``sys.path`` (e.g. the
        # workspace) adds submodules to it. Restore must not import one.
        marker = tmp_path / "imported"
        installed = tmp_path / "site"
        (installed / "nooa_test_ns_pkg").mkdir(parents=True)
        (installed / "nooa_test_ns_pkg" / "legit.py").write_text("X = 1\n")
        attacker = tmp_path / "workspace" / "nooa_test_ns_pkg"
        attacker.mkdir(parents=True)
        (attacker / "pwn.py").write_text(
            textwrap.dedent(f"""
                from pathlib import Path
                Path({str(marker)!r}).write_text("ran")
                import dataclasses

                @dataclasses.dataclass
                class Gadget:
                    x: int = 0
            """)
        )
        monkeypatch.syspath_prepend(str(installed))
        import importlib

        ns = importlib.import_module("nooa_test_ns_pkg")
        try:
            assert ns.__spec__.origin is None or not getattr(ns, "__file__", None)
            monkeypatch.syspath_prepend(str(tmp_path / "workspace"))
            importlib.invalidate_caches()
            blob = _envelope("dataclass", "nooa_test_ns_pkg.pwn.Gadget", {"x": 1})
            with pytest.raises(DeserializationError, match="never imports"):
                _restore(blob)
            assert not marker.exists()
            assert "nooa_test_ns_pkg.pwn" not in sys.modules
        finally:
            for name in [m for m in sys.modules if m.split(".")[0] == "nooa_test_ns_pkg"]:
                del sys.modules[name]

    def test_extend_path_package_submodule_is_not_imported(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # A regular package whose ``__init__`` calls ``pkgutil.extend_path``
        # gets every same-named directory on ``sys.path`` (e.g. the workspace)
        # in its plain-list ``__path__``. Restore must not import from it.
        marker = tmp_path / "imported"
        installed = tmp_path / "site" / "nooa_test_ext_pkg"
        installed.mkdir(parents=True)
        (installed / "__init__.py").write_text(
            "from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n"
        )
        attacker = tmp_path / "workspace" / "nooa_test_ext_pkg"
        attacker.mkdir(parents=True)
        (attacker / "pwn.py").write_text(
            textwrap.dedent(f"""
                from pathlib import Path
                Path({str(marker)!r}).write_text("ran")
                import dataclasses

                @dataclasses.dataclass
                class Gadget:
                    x: int = 0
            """)
        )
        monkeypatch.syspath_prepend(str(tmp_path / "workspace"))
        monkeypatch.syspath_prepend(str(tmp_path / "site"))
        import importlib

        importlib.invalidate_caches()
        pkg = importlib.import_module("nooa_test_ext_pkg")
        try:
            assert str(attacker) in list(pkg.__path__)
            blob = _envelope("dataclass", "nooa_test_ext_pkg.pwn.Gadget", {"x": 1})
            with pytest.raises(DeserializationError, match="never imports"):
                _restore(blob)
            assert not marker.exists()
            assert "nooa_test_ext_pkg.pwn" not in sys.modules
        finally:
            for name in [m for m in sys.modules if m.split(".")[0] == "nooa_test_ext_pkg"]:
                del sys.modules[name]

    def test_unloaded_submodule_of_loaded_package_is_refused(self) -> None:
        # Restore never imports, not even a submodule of a loaded package.
        import json

        missing = object()
        saved_module = sys.modules.pop("json.tool", missing)
        saved_attr = json.__dict__.pop("tool", missing)
        try:
            from nooa.storage.serialization import _import_class

            with pytest.raises(DeserializationError, match="never imports"):
                _import_class("json.tool.main")
            assert "json.tool" not in sys.modules
            import json.tool

            assert _import_class("json.tool.main") is json.tool.main
        finally:
            sys.modules.pop("json.tool", None)
            json.__dict__.pop("tool", None)
            if saved_module is not missing:
                sys.modules["json.tool"] = saved_module
            if saved_attr is not missing:
                json.tool = saved_attr

    def test_module_getattr_hook_is_not_run(self, monkeypatch) -> None:
        # A lazy package's module-level ``__getattr__`` could import a
        # submodule the snapshot named; restore must not call it.
        import types

        calls: list[str] = []
        lazy = types.ModuleType("nooa_test_lazy_pkg")

        def _hook(name: str):
            calls.append(name)
            raise AttributeError(name)

        lazy.__getattr__ = _hook  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "nooa_test_lazy_pkg", lazy)
        from nooa.storage.serialization import _import_class

        with pytest.raises(DeserializationError):
            _import_class("nooa_test_lazy_pkg.Gadget")
        assert calls == []

    def test_custom_getattribute_is_not_run(self, monkeypatch) -> None:
        # A loaded object with a custom ``__getattribute__`` must not have it
        # called while a class path is resolved through it.
        import types

        calls: list[str] = []

        class _Hooked(types.ModuleType):
            def __getattribute__(self, name: str):
                calls.append(name)
                return super().__getattribute__(name)

        hooked = _Hooked("nooa_test_hooked_mod")

        @dataclasses.dataclass
        class Inner:
            x: int = 0

        types.ModuleType.__setattr__(hooked, "Inner", Inner)
        monkeypatch.setitem(sys.modules, "nooa_test_hooked_mod", hooked)
        from nooa.storage.serialization import _import_class

        calls.clear()
        assert _import_class("nooa_test_hooked_mod.Inner") is Inner
        with pytest.raises(DeserializationError):
            _import_class("nooa_test_hooked_mod.Missing")
        assert calls == []


class _Holder(Agent, llm=FakeLLMClient()):
    value: int = 0


class TestAgentSnapshotRestore:
    def test_tampered_attribute_is_skipped_and_others_restore(self, tmp_path: Path) -> None:
        marker = tmp_path / "agent-popen"
        gadget = _envelope("dataclass", "subprocess.Popen", {"args": ["touch", str(marker)]})
        snap = AgentSnapshot(
            attributes={"value": 3, "zz_evil": gadget},
            type_allowlist={"subprocess.Popen"},
        )
        agent = _Holder()
        skipped = snap.restore(agent)
        assert agent.value == 3
        assert "zz_evil" in skipped
        assert "is not a dataclass" in skipped["zz_evil"]
        assert "zz_evil" not in agent.__dict__
        assert not marker.exists()

    def test_context_block_of_unloaded_package_is_skipped(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        # A static context block whose class lives in a package that is not
        # imported yet is skipped with a warning; the rest of the restore goes on.
        marker = tmp_path / "imported"
        pkg = tmp_path / "user_ctx_pkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text(
            textwrap.dedent(f"""
                from pathlib import Path
                Path({str(marker)!r}).write_text("ran")
                import dataclasses

                @dataclasses.dataclass
                class Note:
                    x: int = 0
            """)
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        snap = AgentSnapshot(
            context=[
                StaticContextBlock(
                    key="note", value=_envelope("dataclass", "user_ctx_pkg.Note", {"x": 1})
                ),
                StaticContextBlock(key="plain", value="kept"),
            ],
            attributes={"value": 4},
            type_allowlist={"user_ctx_pkg.Note"},
        )
        agent = _Holder()
        with caplog.at_level(logging.WARNING, logger="nooa.storage.snapshot"):
            snap.restore(agent)
        assert agent.value == 4
        assert agent.context_manager["plain"] == "kept"
        assert "note" not in agent.context_manager
        assert any("context block 'note'" in r.getMessage() for r in caplog.records)
        assert not marker.exists()
        assert "user_ctx_pkg" not in sys.modules


class TestTrustedRoundTripsStillWork:
    def test_round_trip_of_every_kind(self) -> None:
        value = {
            "point": Point(1, 2),
            "model": Model(name="m", point=Point(3)),
            "config": Config("h", 8080),
            "color": Color.RED,
            "tuple": (1, Point(4)),
        }
        blob, allowlist = serialize(value)
        restored = deserialize(blob, allowlist)
        assert restored["point"] == Point(1, 2)
        assert restored["model"] == Model(name="m", point=Point(3))
        assert vars(restored["config"]) == {"host": "h", "port": 8080}
        assert restored["color"] == "red"  # enums are stored by value
        assert restored["tuple"] == (1, Point(4))

    def test_snapshot_vars_round_trip(self) -> None:
        sv = SnapshotVars()
        sv["point"] = Point(5, 6)
        sv["model"] = Model(name="v")
        sv["plain"] = {"__class__": "subprocess.Popen", "__type__": "dataclass", "data": {}}
        blob, allowlist = serialize(sv)
        restored = deserialize(blob, allowlist)
        assert restored["point"] == Point(5, 6)
        assert restored["model"] == Model(name="v")
        # A plain dict that looks like an envelope stays a plain dict.
        assert restored["plain"] == sv["plain"]

    def test_snapshot_vars_drops_only_the_gadget_entry(self, tmp_path: Path) -> None:
        marker = tmp_path / "vars-popen"
        sv = SnapshotVars()
        sv["keep"] = Point(1)
        blob, allowlist = serialize(sv)
        entries = blob["data"]["_data"]
        entries["evil"] = _envelope(
            "dataclass", "subprocess.Popen", {"args": ["touch", str(marker)]}
        )
        restored = deserialize(blob, allowlist | {"subprocess.Popen"})
        assert restored["keep"] == Point(1)
        assert "evil" not in restored
        assert not marker.exists()
