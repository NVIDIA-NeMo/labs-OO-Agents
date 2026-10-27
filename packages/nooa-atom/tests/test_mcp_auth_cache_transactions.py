# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline shared-cache transaction and stale-provider regressions."""

import asyncio
import json
import os
import stat
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from mcp.client.auth.exceptions import OAuthFlowError
from mcp.shared.auth import OAuthClientInformationFull, OAuthMetadata, OAuthToken
from nooa_atom.skills.mcp_auth import FileTokenStorage, PastedSignIn, oauth_provider


def test_unrelated_writer_cannot_restore_forgotten_entry(tmp_path, monkeypatch):
    path = tmp_path / "tokens.json"
    a = FileTokenStorage("https://a.example/mcp", path)
    b = FileTokenStorage("https://b.example/mcp", path)
    a._update(marker="a")
    read = threading.Event()
    release = threading.Event()
    original = b._write

    def delayed_write(data):
        read.set()
        assert release.wait(5)
        original(data)

    monkeypatch.setattr(b, "_write", delayed_write)
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(b._update, marker="b")
        assert read.wait(5)
        forgotten = pool.submit(a.forget_auth)
        # Before the fix forget completes while B holds a stale snapshot.
        # With a transaction lock it waits, then deletes from B's new file.
        try:
            forgotten.result(timeout=0.1)
        except TimeoutError:
            pass
        release.set()
        writer.result(timeout=5)
        forgotten.result(timeout=5)
    saved = json.loads(path.read_text())
    assert a.server_url not in saved
    assert saved[b.server_url]["marker"] == "b"


@pytest.mark.parametrize(
    "winner",
    [
        "valid",
        "expired",
        "unknown_expiry",
        "scope",
        "previous_scope",
        "issuer",
        "native_issuer",
        "client",
        "malformed",
        "unbound",
        "cimd_issuer",
    ],
)
async def test_stale_failed_refresh_preserves_and_reuses_rotated_winner(tmp_path, winner):
    url = "https://resource.example/mcp"
    issuer = "https://issuer.example/tenant"
    path = tmp_path / "tokens.json"
    storage = FileTokenStorage(url, path)
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id="synthetic-client",
            redirect_uris=["http://localhost/callback"],
            token_endpoint_auth_method="none",
        )
    )
    await storage.set_tokens(
        OAuthToken(
            access_token="synthetic-old",
            refresh_token="synthetic-old-refresh",
            token_type="Bearer",
            expires_in=3600,
            scope="read extra" if winner == "previous_scope" else "read",
        )
    )
    storage._update(issuer=issuer)
    if winner == "cimd_issuer":
        storage._update(
            client_info={
                **storage._entry()["client_info"],
                "client_id": "https://client.example/metadata",
            }
        )
    providers = [
        oauth_provider(PastedSignIn(), FileTokenStorage(url, path), scope="read") for _ in range(2)
    ]
    if winner == "cimd_issuer":
        for provider in providers:
            provider.context.client_metadata_url = "https://client.example/metadata"
    for provider in providers:
        await provider._initialize()
        provider.context.token_expiry_time = time.time() - 1
    refreshes = 0

    def respond(request):
        nonlocal refreshes
        if str(request.url) == url:
            return httpx.Response(200)
        if "oauth-protected-resource" in request.url.path:
            return httpx.Response(200, json={"resource": url, "authorization_servers": [issuer]})
        if request.method == "POST":
            refreshes += 1
            if refreshes == 1:
                return httpx.Response(
                    200,
                    json={
                        "access_token": "synthetic-new",
                        "refresh_token": "synthetic-new-refresh",
                        "token_type": "Bearer",
                        "expires_in": 900,
                        "scope": "read write",
                    },
                )
            entry = storage._entry()
            changes = {}
            if winner == "expired":
                changes["expires_at"] = 0
            elif winner == "unknown_expiry":
                changes["expires_at"] = None
            elif winner == "scope":
                changes["tokens"] = {**entry["tokens"], "scope": "write"}
            elif winner in {"issuer", "cimd_issuer"}:
                changes["issuer"] = "https://foreign.example/tenant"
            elif winner == "native_issuer":
                changes["client_info"] = {
                    **entry["client_info"],
                    "issuer": "https://foreign.example/tenant",
                }
            elif winner == "client":
                changes["client_info"] = {**entry["client_info"], "client_id": "different-client"}
            elif winner == "malformed":
                changes["tokens"] = {"token_type": "Bearer"}
            elif winner == "unbound":
                changes["issuer"] = None
            storage._update(**changes)
            return httpx.Response(400, json={"error": "invalid_grant"})
        return httpx.Response(
            200,
            json={
                "issuer": issuer,
                "authorization_endpoint": f"{issuer}/authorize",
                "token_endpoint": f"{issuer}/token",
            },
        )

    for provider in providers:
        async with httpx.AsyncClient(
            auth=provider, transport=httpx.MockTransport(respond)
        ) as client:
            assert (await client.get(url)).status_code == 200
    saved = storage._entry()
    assert "tokens" in saved
    if winner == "valid":
        assert saved["tokens"]["access_token"] == "synthetic-new"
        assert providers[1].context.current_tokens is not None
        assert providers[1].context.current_tokens.access_token == "synthetic-new"
        assert providers[1].context.token_expiry_time == saved["expires_at"]
        assert 0 < saved["expires_at"] - time.time() <= 900
    else:
        assert providers[1].context.current_tokens is None
    assert refreshes == 2
    assert all(not provider.context.lock.locked() for provider in providers)


def test_process_transaction_prevents_forget_resurrection(tmp_path):
    path = tmp_path / "tokens.json"
    a = FileTokenStorage("https://a.example/mcp", path)
    a._update(marker="a")
    # Fresh interpreters avoid fork's inherited locks/threads and prove the
    # production lock does not depend on a process-local Python mutex.
    code = """
import sys, time
from pathlib import Path
from nooa_atom.skills.mcp_auth import FileTokenStorage
path = Path(sys.argv[1])
if sys.argv[2] == 'write':
    storage = FileTokenStorage('https://b.example/mcp', path)
    original = storage._write
    def delayed(data):
        path.with_suffix('.ready').touch()
        deadline = time.monotonic() + 10
        while not path.with_suffix('.release').exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        original(data)
    storage._write = delayed
    storage._update(marker='b')
else:
    path.with_suffix('.forget_started').touch()
    FileTokenStorage('https://a.example/mcp', path).forget_auth()
    path.with_suffix('.forgotten').touch()
"""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    writer = subprocess.Popen([sys.executable, "-c", code, str(path), "write"], env=env)
    forgetter = None
    try:
        deadline = time.monotonic() + 10
        while not path.with_suffix(".ready").exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        forgetter = subprocess.Popen([sys.executable, "-c", code, str(path), "forget"], env=env)
        deadline = time.monotonic() + 10
        while not path.with_suffix(".forget_started").exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        time.sleep(0.1)
        assert not path.with_suffix(".forgotten").exists()
        path.with_suffix(".release").touch()
        assert writer.wait(timeout=10) == forgetter.wait(timeout=10) == 0
        assert path.with_suffix(".forgotten").exists()
    finally:
        path.with_suffix(".release").touch()
        for process in (writer, forgetter):
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=10)
    saved = json.loads(path.read_text())
    assert a.server_url not in saved
    assert saved["https://b.example/mcp"]["marker"] == "b"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.with_name(path.name + ".lock").stat().st_mode) == 0o600


@pytest.mark.parametrize("operation", ["set_tokens", "set_client_info", "invalidate"])
async def test_cancellation_of_lock_waiter_does_not_publish_or_leak(tmp_path, operation):
    path = tmp_path / "tokens.json"
    storage = FileTokenStorage("https://resource.example/mcp", path)
    storage._update(marker="unchanged")
    before = storage._entry()
    held, release = threading.Event(), threading.Event()

    def hold():
        with storage._transaction():
            held.set()
            assert release.wait(5)

    with ThreadPoolExecutor(max_workers=1) as pool:
        holder = pool.submit(hold)
        assert held.wait(5)
        if operation == "set_tokens":
            work = storage.set_tokens(OAuthToken(access_token="synthetic", token_type="Bearer"))
        elif operation == "set_client_info":
            work = storage.set_client_info(
                OAuthClientInformationFull(
                    client_id="synthetic", redirect_uris=["http://localhost/callback"]
                )
            )
        else:
            work = storage.invalidate(before)
        task = asyncio.create_task(work)
        try:
            await asyncio.sleep(0.03)
            assert not task.done()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
            assert storage._entry() == before
        finally:
            release.set()
            holder.result(timeout=5)
    await asyncio.wait_for(storage.update(after="released"), 1)
    assert storage._entry() == {**before, "after": "released"}
    assert not list(tmp_path.glob(".mcp_oauth.*"))


def test_failure_releases_transaction_and_preserves_cache(tmp_path, monkeypatch):
    storage = FileTokenStorage("https://resource.example/mcp", tmp_path / "tokens.json")
    storage._update(marker="unchanged")
    original = storage._write

    def fail(data):
        raise OSError("synthetic write failure")

    monkeypatch.setattr(storage, "_write", fail)
    with pytest.raises(OSError):
        storage.clear_tokens()
    monkeypatch.setattr(storage, "_write", original)
    storage._update(after="released")
    assert storage._entry() == {"marker": "unchanged", "after": "released"}


async def test_clear_and_forget_preserve_unrelated_entries_and_client_bindings(tmp_path):
    path = tmp_path / "tokens.json"
    a, b = (
        FileTokenStorage("https://a.example/mcp", path),
        FileTokenStorage("https://b.example/mcp", path),
    )
    client = {"client_id": "synthetic", "issuer": "https://issuer.example/"}
    a._update(
        client_info=client,
        issuer=client["issuer"],
        tokens={"access_token": "synthetic"},
        expires_at=1,
    )
    b._update(marker="sibling")
    a.clear_tokens()
    assert a._entry() == {"client_info": client, "issuer": client["issuer"]}
    assert b._entry() == {"marker": "sibling"}
    a.forget_auth()
    assert a._entry() == {}
    assert b._entry() == {"marker": "sibling"}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


async def test_older_sdk_registration_roundtrip_preserves_native_issuer_without_tokens(tmp_path):
    storage = FileTokenStorage("https://resource.example/mcp", tmp_path / "tokens.json")
    storage._update(
        client_info={
            "client_id": "synthetic",
            "redirect_uris": ["http://localhost/callback"],
            "issuer": "https://issuer.example/",
        }
    )
    client = await storage.get_client_info()
    assert client is not None
    await storage.set_client_info(client)
    assert storage.get_client_issuer() == "https://issuer.example/"


async def test_pending_human_signin_holds_no_cache_transaction_lock(tmp_path):
    url = "https://resource.example/mcp"
    issuer = "https://issuer.example/"
    path = tmp_path / "tokens.json"
    storage = FileTokenStorage(url, path, client_id="synthetic")
    await storage.set_tokens(
        OAuthToken(
            access_token="synthetic-old",
            refresh_token="synthetic-refresh",
            token_type="Bearer",
            expires_in=3600,
        )
    )
    storage._update(issuer=issuer, expires_at=0)
    signin = PastedSignIn()
    provider = oauth_provider(signin, storage)

    def respond(request):
        if str(request.url) == url:
            return httpx.Response(401)
        if "oauth-protected-resource" in request.url.path:
            return httpx.Response(200, json={"resource": url, "authorization_servers": [issuer]})
        if request.method == "POST":
            return httpx.Response(400, json={"error": "invalid_grant"})
        return httpx.Response(
            200,
            json={
                "issuer": issuer,
                "authorization_endpoint": f"{issuer}authorize",
                "token_endpoint": f"{issuer}token",
            },
        )

    async with httpx.AsyncClient(auth=provider, transport=httpx.MockTransport(respond)) as client:
        task = asyncio.create_task(client.get(url))
        try:
            await asyncio.wait_for(signin.url_ready.wait(), 3)
            assert not task.done()
            assert "tokens" not in storage._entry()
            sibling = FileTokenStorage("https://sibling.example/mcp", path)
            await asyncio.wait_for(sibling.update(marker="human-wait-unlocked"), 1)
            assert sibling._entry() == {"marker": "human-wait-unlocked"}
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    assert not provider.context.lock.locked()
    await asyncio.wait_for(storage.update(after="cancelled"), 1)


@pytest.mark.parametrize(
    ("registration", "actual_client", "accepted"),
    [
        ("unknown", "synthetic-client-A", False),
        ("unknown", "synthetic-client-B", False),
        ("unknown", None, False),
        ("malformed", "synthetic-client-A", False),
        ("malformed", "synthetic-client-B", False),
        ("bound", "synthetic-client-A", True),
        ("bound", "synthetic-client-B", False),
        ("bound", None, False),
    ],
)
async def test_refresh_winner_requires_valid_persisted_actual_client_binding(
    tmp_path, registration, actual_client, accepted
):
    url = "https://resource.example/mcp"
    issuer = "https://issuer.example/"
    path = tmp_path / "tokens.json"
    a = FileTokenStorage(url, path, client_id="synthetic-client-A")
    b = FileTokenStorage(url, path, client_id=actual_client)
    if registration == "bound":
        await a.set_client_info(
            OAuthClientInformationFull(
                client_id="synthetic-client-A",
                redirect_uris=["http://localhost/callback"],
                token_endpoint_auth_method="none",
            )
        )
    elif registration == "malformed":
        a._update(client_info={"client_id": "synthetic-client-A", "redirect_uris": "invalid"})
    with a.bind_token_writes(issuer):
        await a.set_tokens(
            OAuthToken(
                access_token="synthetic-old",
                refresh_token="synthetic-refresh",
                token_type="Bearer",
                expires_in=3600,
                scope="read",
            )
        )
    providers = [oauth_provider(PastedSignIn(), storage, scope="read") for storage in (a, b)]
    for provider in providers:
        await provider._initialize()
        provider.context.oauth_metadata = OAuthMetadata(
            issuer=issuer,
            authorization_endpoint=f"{issuer}authorize",
            token_endpoint=f"{issuer}token",
            response_types_supported=["code"],
        )
    if registration == "bound":
        # Model the actual initialized provider's client identity, independently
        # of the raw snapshot equality checked by winner adoption.
        providers[1].context.client_info = (
            OAuthClientInformationFull(
                client_id=actual_client,
                redirect_uris=["http://localhost/callback"],
                token_endpoint_auth_method="none",
            )
            if actual_client is not None
            else None
        )
    if actual_client is not None:
        assert providers[1].context.client_info.client_id == actual_client
    else:
        assert providers[1].context.client_info is None
    with a.bind_token_writes(issuer):
        await a.set_tokens(
            OAuthToken(
                access_token="synthetic-winner-A",
                refresh_token="synthetic-rotated-A",
                token_type="Bearer",
                expires_in=900,
                scope="read",
            )
        )
    winner = a._entry()
    result = await providers[1]._handle_refresh_response(
        httpx.Response(400, json={"error": "invalid_grant"})
    )
    assert result is accepted
    assert a._entry() == winner
    if accepted:
        assert providers[1].context.current_tokens.access_token == "synthetic-winner-A"
        assert providers[1].context.token_expiry_time == winner["expires_at"]
    else:
        assert providers[1].context.current_tokens is None


@pytest.mark.parametrize("binding", ["atom", "native"])
@pytest.mark.parametrize("cimd", [False, True])
@pytest.mark.parametrize("expired", [False, True])
async def test_refresh_checks_actual_snapshot_not_concurrent_issuer_cache(
    tmp_path, binding, cimd, expired
):
    url = "https://resource.example/mcp"
    old_issuer, new_issuer = "https://old.example/", "https://new.example/"
    path = tmp_path / "tokens.json"
    storage = FileTokenStorage(url, path)
    client_id = "https://client.example/metadata" if cimd else "synthetic-old-client"
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id=client_id,
            client_secret="synthetic-old-secret",
            redirect_uris=["http://localhost/callback"],
            token_endpoint_auth_method="client_secret_post",
        )
    )
    await storage.set_tokens(
        OAuthToken(
            access_token="synthetic-old-access",
            refresh_token="synthetic-old-refresh",
            token_type="Bearer",
            expires_in=3600,
        )
    )
    if binding == "atom":
        storage._update(issuer=old_issuer)
    else:
        storage._update(client_info={**storage._entry()["client_info"], "issuer": old_issuer})
    signin = PastedSignIn()
    provider = oauth_provider(signin, storage)
    if cimd:
        provider.context.client_metadata_url = client_id
    await provider._initialize()
    if expired:
        provider.context.token_expiry_time = time.time() - 1
    # Another provider publishes a different issuer after this provider loaded
    # its old credentials. Validation must never bless those old credentials
    # by reading the new on-disk binding.
    if binding == "atom":
        storage._update(issuer=new_issuer)
    else:
        storage._update(client_info={**storage._entry()["client_info"], "issuer": new_issuer})
    winner = path.read_bytes()
    foreign_token_posts = 0

    def respond(request):
        nonlocal foreign_token_posts
        if str(request.url) == url:
            return httpx.Response(401)
        if "oauth-protected-resource" in request.url.path:
            return httpx.Response(
                200, json={"resource": url, "authorization_servers": [new_issuer]}
            )
        if request.method == "POST":
            foreign_token_posts += 1
            return httpx.Response(400, json={"error": "invalid_grant"})
        return httpx.Response(
            200,
            json={
                "issuer": new_issuer,
                "authorization_endpoint": f"{new_issuer}authorize",
                "token_endpoint": f"{new_issuer}token",
            },
        )

    async with httpx.AsyncClient(auth=provider, transport=httpx.MockTransport(respond)) as client:
        # Bound the vulnerable implementation's human wait while reporting
        # only counts/types, never a request containing synthetic credentials.
        with pytest.raises((OAuthFlowError, TimeoutError)) as error:
            await asyncio.wait_for(client.get(url), 0.2)
    assert foreign_token_posts == 0
    assert isinstance(error.value, OAuthFlowError)
    assert path.read_bytes() == winner
    assert signin.url is None
    assert not provider.context.lock.locked()


@pytest.mark.parametrize("binding", ["atom", "native"])
@pytest.mark.parametrize("cimd", [False, True])
async def test_manual_issuer_validation_uses_loaded_snapshot(tmp_path, binding, cimd):
    url = "https://resource.example/mcp"
    old_issuer, new_issuer = "https://old.example/", "https://new.example/"
    storage = FileTokenStorage(url, tmp_path / "tokens.json")
    client_id = "https://client.example/metadata" if cimd else "synthetic-client"
    await storage.set_client_info(
        OAuthClientInformationFull(
            client_id=client_id,
            redirect_uris=["http://localhost/callback"],
            token_endpoint_auth_method="none",
        )
    )
    if binding == "atom":
        storage._update(issuer=old_issuer)
    else:
        storage._update(client_info={**storage._entry()["client_info"], "issuer": old_issuer})
    provider = oauth_provider(PastedSignIn(), storage)
    if cimd:
        provider.context.client_metadata_url = client_id
    await provider._initialize()
    if binding == "atom":
        storage._update(issuer=new_issuer)
    else:
        storage._update(client_info={**storage._entry()["client_info"], "issuer": new_issuer})
    if cimd:
        # Portable identity permits a *new* authorization, not refresh reuse.
        provider._validate_cached_issuer(new_issuer, refresh=False)
    else:
        with pytest.raises(OAuthFlowError):
            provider._validate_cached_issuer(new_issuer, refresh=False)
