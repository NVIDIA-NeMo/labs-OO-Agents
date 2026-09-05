from __future__ import annotations

from types import SimpleNamespace

import pytest
from nooa_cybergym import run


def test_missing_required_image_fails_before_run(monkeypatch):
    class Missing(Exception):
        pass

    client = SimpleNamespace(
        images=SimpleNamespace(get=lambda image: (_ for _ in ()).throw(Missing()))
    )
    monkeypatch.setattr(run, "ImageNotFound", Missing)

    with pytest.raises(RuntimeError, match="required runner image is not local"):
        run.require_local_image(client, "runner:tag", role="runner")


def test_internal_route_probe_uses_runner_image_and_real_network():
    calls = []

    class Containers:
        def run(self, image, **kwargs):
            calls.append((image, kwargs))

    client = SimpleNamespace(containers=Containers())
    env = {"HTTP_PROXY": "http://cybergym-proxy:3128"}

    run.preflight_internal_route(
        client,
        image="runner:tag",
        network="cybergym-internal",
        env=env,
        server="http://server:8666",
    )

    image, kwargs = calls[0]
    assert image == "runner:tag"
    assert kwargs["network"] == "cybergym-internal"
    assert kwargs["environment"] == env
    assert kwargs["remove"] is True
    assert "http://server:8666/docs" in kwargs["command"][2]
