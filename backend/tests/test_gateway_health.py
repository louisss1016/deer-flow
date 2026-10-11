"""Unit tests for the gateway readiness probe (app.gateway.health)."""

import asyncio
import pathlib
import sqlite3
import sys
import threading
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.gateway.health as health_module
from app.gateway.health import (
    DATABASE_NOT_CONFIGURED,
    DATABASE_OK,
    DATABASE_UNREACHABLE,
    HEALTH_NOT_CONFIGURED,
    HEALTH_OK,
    HEALTH_UNREACHABLE,
    _probe_checkpointer_backend,
    check_database_health,
    check_provisioner_health,
    check_stream_bridge_health,
    readiness_payload,
    register_readiness_route,
    resolve_checkpointer_config,
)
from deerflow.config.checkpointer_config import CheckpointerConfig


class _FakeConnection:
    async def execute(self, *args, **kwargs):
        return None


class _FakeEngine:
    def __init__(self, *, unreachable: bool = False):
        self._unreachable = unreachable

    def connect(self):
        @asynccontextmanager
        async def _connect():
            if self._unreachable:
                raise RuntimeError("database is down")
            yield _FakeConnection()

        return _connect()


def _create_sqlite_file(path: pathlib.Path) -> None:
    """Create a valid (empty) SQLite database file at *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sqlite3.connect(str(path)).close()


@pytest.mark.anyio
async def test_check_database_health_without_engine(monkeypatch):
    """backend=memory (no engine) must report not_configured, never unreachable."""
    monkeypatch.setattr("app.gateway.health.get_engine", lambda: None)

    assert await check_database_health() == DATABASE_NOT_CONFIGURED


@pytest.mark.anyio
async def test_check_database_health_reachable(monkeypatch):
    monkeypatch.setattr("app.gateway.health.get_engine", lambda: _FakeEngine())

    assert await check_database_health() == DATABASE_OK


@pytest.mark.anyio
async def test_check_database_health_unreachable(monkeypatch):
    monkeypatch.setattr("app.gateway.health.get_engine", lambda: _FakeEngine(unreachable=True))

    assert await check_database_health() == DATABASE_UNREACHABLE


@pytest.mark.anyio
async def test_readiness_payload_ready_when_database_ok_and_memory_backend(monkeypatch):
    monkeypatch.setattr("app.gateway.health.get_engine", lambda: _FakeEngine())

    status_code, payload = await readiness_payload(CheckpointerConfig(type="memory"))

    assert status_code == 200
    assert payload["status"] == "ready"
    assert payload["database"] == DATABASE_OK
    assert payload["checkpointer"] == DATABASE_NOT_CONFIGURED


@pytest.mark.anyio
async def test_readiness_payload_degraded_when_database_unreachable(monkeypatch):
    monkeypatch.setattr("app.gateway.health.get_engine", lambda: _FakeEngine(unreachable=True))

    status_code, payload = await readiness_payload(CheckpointerConfig(type="memory"))

    assert status_code == 503
    assert payload["status"] == "degraded"
    assert payload["database"] == DATABASE_UNREACHABLE
    assert payload["checkpointer"] == DATABASE_NOT_CONFIGURED


@pytest.mark.anyio
async def test_readiness_payload_ready_when_nothing_configured(monkeypatch):
    """backend=memory end to end must stay ready with not_configured results."""
    monkeypatch.setattr("app.gateway.health.get_engine", lambda: None)

    status_code, payload = await readiness_payload(CheckpointerConfig(type="memory"))

    assert status_code == 200
    assert payload["status"] == "ready"
    assert payload["database"] == DATABASE_NOT_CONFIGURED
    assert payload["checkpointer"] == DATABASE_NOT_CONFIGURED


@pytest.mark.anyio
async def test_readiness_payload_degraded_when_checkpointer_unreachable_but_database_ok(tmp_path, monkeypatch):
    """A healthy ORM engine must not mask an unreachable legacy checkpointer backend."""
    monkeypatch.setattr("app.gateway.health.get_engine", lambda: _FakeEngine())
    config = CheckpointerConfig(type="sqlite", connection_string=str(tmp_path / "missing" / "checkpoints.db"))

    status_code, payload = await readiness_payload(config)

    assert status_code == 503
    assert payload["status"] == "degraded"
    assert payload["database"] == DATABASE_OK
    assert payload["checkpointer"] == DATABASE_UNREACHABLE


@pytest.mark.anyio
async def test_readiness_payload_fails_closed_without_startup_snapshot(monkeypatch):
    """No startup config snapshot must degrade readiness, never report ready."""
    monkeypatch.setattr("app.gateway.health.get_engine", lambda: _FakeEngine())

    status_code, payload = await readiness_payload(None)

    assert status_code == 503
    assert payload["status"] == "degraded"
    assert payload["database"] == DATABASE_OK
    assert payload["checkpointer"] == DATABASE_UNREACHABLE


@pytest.mark.anyio
async def test_readiness_probes_run_concurrently(monkeypatch):
    """Slow-but-healthy probes must not add their budgets together."""

    async def _slow_ok(*args) -> str:
        await asyncio.sleep(0.35)
        return DATABASE_OK

    monkeypatch.setattr(health_module, "check_database_health", _slow_ok)
    monkeypatch.setattr(health_module, "_probe_checkpointer_backend", _slow_ok)

    started = time.perf_counter()
    status_code, payload = await readiness_payload(CheckpointerConfig(type="memory"))
    elapsed = time.perf_counter() - started

    assert status_code == 200
    assert payload["database"] == DATABASE_OK
    assert payload["checkpointer"] == DATABASE_OK
    # Two sequential 0.35s probes would take ~0.7s; concurrent ones finish
    # within a single probe window.
    assert elapsed < 0.6


@pytest.mark.anyio
async def test_readiness_payload_enforces_endpoint_deadline(monkeypatch):
    """A probe ignoring its own budget must trip the endpoint-wide deadline."""

    async def _hanging(*args) -> str:
        await asyncio.sleep(30)
        return DATABASE_OK

    monkeypatch.setattr(health_module, "check_database_health", _hanging)
    monkeypatch.setattr(health_module, "_probe_checkpointer_backend", _hanging)
    monkeypatch.setattr(health_module, "_READINESS_DEADLINE_SECONDS", 0.05)

    status_code, payload = await readiness_payload(CheckpointerConfig(type="memory"))

    assert status_code == 503
    assert payload["status"] == "degraded"
    assert payload["database"] == DATABASE_UNREACHABLE
    assert payload["checkpointer"] == DATABASE_UNREACHABLE


@pytest.mark.anyio
async def test_concurrent_readiness_requests_do_not_open_concurrent_probe_connections(monkeypatch):
    """Public /health/ready must serialize connection-opening probes.

    An unauthenticated thundering herd must never translate into an unbounded
    number of new database connections (e.g. past PostgreSQL
    max_connections): at most one probe connection may be in flight at a time
    per process.
    """
    active = 0
    max_active = 0

    async def _tracked_probe(conn_string: str | None) -> str:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            await asyncio.sleep(0.05)
            return DATABASE_OK
        finally:
            active -= 1

    monkeypatch.setattr(health_module, "_probe_sqlite_backend", _tracked_probe)
    monkeypatch.setattr("app.gateway.health.get_engine", lambda: None)
    config = CheckpointerConfig(type="sqlite", connection_string=":memory:")

    results = await asyncio.gather(*(readiness_payload(config) for _ in range(8)))

    assert [status_code for status_code, _ in results] == [200] * 8
    assert max_active == 1


@pytest.mark.anyio
async def test_probe_checkpointer_memory_reports_not_configured():
    assert await _probe_checkpointer_backend(CheckpointerConfig(type="memory")) == DATABASE_NOT_CONFIGURED


@pytest.mark.anyio
async def test_probe_checkpointer_sqlite_reachable(tmp_path):
    db_path = tmp_path / "checkpoints.db"
    _create_sqlite_file(db_path)

    result = await _probe_checkpointer_backend(CheckpointerConfig(type="sqlite", connection_string=str(db_path)))

    assert result == DATABASE_OK


@pytest.mark.anyio
@pytest.mark.parametrize(
    "conn_string",
    ["file:checkpoints.db", "file::memory:?cache=shared", "file:memdb1?mode=memory&cache=shared"],
)
async def test_probe_checkpointer_sqlite_uri_is_unreachable(conn_string, tmp_path, monkeypatch):
    """The runtime refuses SQLite URIs, so the probe must not report them healthy."""
    monkeypatch.chdir(tmp_path)
    _create_sqlite_file(tmp_path / "checkpoints.db")

    result = await _probe_checkpointer_backend(CheckpointerConfig(type="sqlite", connection_string=conn_string))

    assert result == DATABASE_UNREACHABLE
    assert [path.name for path in tmp_path.iterdir()] == ["checkpoints.db"]


@pytest.mark.anyio
async def test_probe_checkpointer_sqlite_missing_file_stays_missing_and_unreachable(tmp_path):
    """The probe must never create a missing SQLite file (regression)."""
    missing = tmp_path / "checkpoints.db"

    result = await _probe_checkpointer_backend(CheckpointerConfig(type="sqlite", connection_string=str(missing)))

    assert result == DATABASE_UNREACHABLE
    assert not missing.exists()


@pytest.mark.anyio
async def test_probe_checkpointer_sqlite_unreachable_when_parent_missing(tmp_path):
    missing_parent = tmp_path / "does-not-exist" / "checkpoints.db"

    result = await _probe_checkpointer_backend(CheckpointerConfig(type="sqlite", connection_string=str(missing_parent)))

    assert result == DATABASE_UNREACHABLE


@pytest.mark.anyio
async def test_probe_checkpointer_sqlite_in_memory_is_not_configured():
    """In-memory SQLite has no external state, mirroring the memory backend."""
    result = await _probe_checkpointer_backend(CheckpointerConfig(type="sqlite", connection_string=":memory:"))

    assert result == DATABASE_NOT_CONFIGURED


@pytest.mark.anyio
async def test_probe_checkpointer_postgres_without_psycopg_is_unreachable(monkeypatch):
    monkeypatch.setitem(sys.modules, "psycopg", None)

    result = await _probe_checkpointer_backend(
        CheckpointerConfig(
            type="postgres",
            connection_string="postgresql://user:pass@localhost:5432/deerflow",
        )
    )

    assert result == DATABASE_UNREACHABLE


def test_resolve_checkpointer_config_passes_through_resolution(monkeypatch):
    resolved = CheckpointerConfig(type="memory")
    monkeypatch.setattr(
        "deerflow.runtime.checkpointer.provider._resolve_checkpointer_config",
        lambda app_config: resolved,
    )

    assert resolve_checkpointer_config(object()) is resolved


def test_resolve_checkpointer_config_failure_fails_closed(monkeypatch):
    """A resolution failure must surface as None, never as a memory default."""

    def _raise(app_config):
        raise RuntimeError("broken checkpointer config")

    monkeypatch.setattr(
        "deerflow.runtime.checkpointer.provider._resolve_checkpointer_config",
        _raise,
    )

    assert resolve_checkpointer_config(object()) is None


class _BlockingProbeConnection:
    def __init__(self) -> None:
        self.close_started = asyncio.Event()
        self.allow_close = asyncio.Event()
        self.close_finished = asyncio.Event()

    async def execute(self, *_args, **_kwargs):
        return None

    async def close(self) -> None:
        self.close_started.set()
        await self.allow_close.wait()
        self.close_finished.set()


async def _assert_probe_close_is_drained(probe_coro, connection: _BlockingProbeConnection, label: str) -> None:
    probe = asyncio.create_task(probe_coro)
    try:
        await asyncio.wait_for(connection.close_started.wait(), 1)

        probe.cancel("first cancellation")
        await asyncio.sleep(0)
        probe.cancel("second cancellation")
        for _ in range(5):
            await asyncio.sleep(0)

        assert not probe.done(), f"readiness probe returned before its {label} connection closed"

        connection.allow_close.set()
        with pytest.raises(asyncio.CancelledError):
            await probe
        assert connection.close_finished.is_set()
    finally:
        connection.allow_close.set()
        await asyncio.gather(probe, return_exceptions=True)


@pytest.mark.anyio
async def test_sqlite_probe_drains_connection_close_across_repeated_cancellation(monkeypatch):
    import aiosqlite

    connection = _BlockingProbeConnection()

    async def fake_connect(*_args, **_kwargs):
        return connection

    monkeypatch.setattr(aiosqlite, "connect", fake_connect)

    await _assert_probe_close_is_drained(
        health_module._probe_sqlite_backend("/tmp/deerflow-health-probe.db"),
        connection,
        "SQLite",
    )


class _FakeProbeCursor:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, *_args, **_kwargs):
        return None


class _BlockingPostgresProbeConnection(_BlockingProbeConnection):
    def cursor(self):
        return _FakeProbeCursor()


@pytest.mark.anyio
async def test_postgres_probe_drains_connection_close_across_repeated_cancellation(monkeypatch):
    import types

    connection = _BlockingPostgresProbeConnection()

    class FakeAsyncConnection:
        @classmethod
        async def connect(cls, *_args, **_kwargs):
            return connection

    fake_psycopg = types.ModuleType("psycopg")
    fake_psycopg.AsyncConnection = FakeAsyncConnection
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)

    await _assert_probe_close_is_drained(
        health_module._probe_postgres_backend(
            "postgresql://user:pass@localhost:5432/deerflow",
            "",
        ),
        connection,
        "PostgreSQL",
    )


class _FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


class _FakeBridge:
    """Minimal StreamBridge stand-in for readiness-probe tests."""

    def __init__(self, result: str | None = None, *, raises: bool = False):
        self._result = result
        self._raises = raises
        self.calls = 0

    async def health_check(self) -> str:
        self.calls += 1
        if self._raises:
            raise RuntimeError("bridge health check exploded")
        return self._result


@pytest.mark.anyio
async def test_check_stream_bridge_health_none_is_not_configured():
    assert await check_stream_bridge_health(None) == HEALTH_NOT_CONFIGURED


@pytest.mark.anyio
async def test_check_stream_bridge_health_ok():
    assert await check_stream_bridge_health(_FakeBridge(HEALTH_OK)) == HEALTH_OK


@pytest.mark.anyio
async def test_check_stream_bridge_health_unreachable():
    assert await check_stream_bridge_health(_FakeBridge(HEALTH_UNREACHABLE)) == HEALTH_UNREACHABLE


@pytest.mark.anyio
async def test_check_stream_bridge_health_raising_bridge_fails_closed():
    """A custom bridge raising must degrade readiness, never 500 the route."""
    assert await check_stream_bridge_health(_FakeBridge(raises=True)) == HEALTH_UNREACHABLE


@pytest.mark.anyio
async def test_check_provisioner_health_empty_url_is_not_configured():
    assert await check_provisioner_health("") == HEALTH_NOT_CONFIGURED


@pytest.mark.anyio
async def test_check_provisioner_health_ok(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, timeout=None: _FakeResponse(200))

    assert await check_provisioner_health("http://provisioner:8002/") == HEALTH_OK


@pytest.mark.anyio
async def test_check_provisioner_health_non_200_is_unreachable(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, timeout=None: _FakeResponse(500))

    assert await check_provisioner_health("http://provisioner:8002") == HEALTH_UNREACHABLE


@pytest.mark.anyio
async def test_check_provisioner_health_transport_error_is_unreachable(monkeypatch):
    def _boom(url, timeout=None):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "get", _boom)

    assert await check_provisioner_health("http://provisioner:8002") == HEALTH_UNREACHABLE


@pytest.mark.anyio
async def test_check_provisioner_health_runs_off_event_loop_thread(monkeypatch):
    """The synchronous provisioner GET must never block the event loop."""
    loop_thread = threading.get_ident()

    def fake_get(url, timeout=None):
        assert threading.get_ident() != loop_thread
        return _FakeResponse(200)

    monkeypatch.setattr(requests, "get", fake_get)

    assert await check_provisioner_health("http://provisioner:8002") == HEALTH_OK


@pytest.mark.anyio
async def test_readiness_payload_degraded_when_stream_bridge_unreachable(monkeypatch):
    monkeypatch.setattr("app.gateway.health.get_engine", lambda: _FakeEngine())

    status_code, payload = await readiness_payload(CheckpointerConfig(type="memory"), _FakeBridge(HEALTH_UNREACHABLE))

    assert status_code == 503
    assert payload["status"] == "degraded"
    assert payload["database"] == DATABASE_OK
    assert payload["checkpointer"] == DATABASE_NOT_CONFIGURED
    assert payload["stream_bridge"] == HEALTH_UNREACHABLE


@pytest.mark.anyio
async def test_readiness_payload_ready_with_memory_bridge(monkeypatch):
    """An in-process bridge has no external dependency to probe."""
    monkeypatch.setattr("app.gateway.health.get_engine", lambda: _FakeEngine())

    status_code, payload = await readiness_payload(CheckpointerConfig(type="memory"), _FakeBridge(HEALTH_NOT_CONFIGURED))

    assert status_code == 200
    assert payload["status"] == "ready"
    assert payload["stream_bridge"] == HEALTH_NOT_CONFIGURED


@pytest.mark.anyio
async def test_readiness_payload_provisioner_reported_without_degrading(monkeypatch):
    """A shared provisioner blip must be reported, never drain the service."""
    monkeypatch.setattr("app.gateway.health.get_engine", lambda: _FakeEngine())
    monkeypatch.setattr(requests, "get", lambda url, timeout=None: _FakeResponse(503))

    status_code, payload = await readiness_payload(
        CheckpointerConfig(type="memory"),
        _FakeBridge(HEALTH_OK),
        "http://provisioner:8002",
    )

    assert status_code == 200
    assert payload["status"] == "ready"
    assert payload["stream_bridge"] == HEALTH_OK
    assert payload["provisioner"] == HEALTH_UNREACHABLE


@pytest.mark.anyio
async def test_readiness_payload_provisioner_ok_reported(monkeypatch):
    monkeypatch.setattr("app.gateway.health.get_engine", lambda: _FakeEngine())
    monkeypatch.setattr(requests, "get", lambda url, timeout=None: _FakeResponse(200))

    status_code, payload = await readiness_payload(CheckpointerConfig(type="memory"), None, "http://provisioner:8002")

    assert status_code == 200
    assert payload["provisioner"] == HEALTH_OK


def _make_readiness_client(monkeypatch, *, provisioner_url: str = "", checkpointer_config=None) -> TestClient:
    monkeypatch.setattr("app.gateway.health.get_engine", lambda: None)
    monkeypatch.setattr(
        "app.gateway.health.get_app_config",
        lambda: SimpleNamespace(sandbox=SimpleNamespace(provisioner_url=provisioner_url)),
    )
    app = FastAPI()
    register_readiness_route(app)
    # ``langgraph_runtime`` always records the startup snapshot before serving;
    # a missing snapshot is a separate fail-closed contract covered above.
    app.state.checkpointer_config = checkpointer_config or CheckpointerConfig(type="memory")
    return TestClient(app)


def test_readiness_route_degrades_when_stream_bridge_unreachable(monkeypatch):
    """Route level: app.state wiring must reach the payload (issue #6446)."""
    client = _make_readiness_client(monkeypatch)
    client.app.state.checkpointer_config = CheckpointerConfig(type="memory")
    client.app.state.stream_bridge = _FakeBridge(HEALTH_UNREACHABLE)

    response = client.get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["database"] == DATABASE_NOT_CONFIGURED
    assert body["checkpointer"] == DATABASE_NOT_CONFIGURED
    assert body["stream_bridge"] == HEALTH_UNREACHABLE
    assert body["provisioner"] == HEALTH_NOT_CONFIGURED


def test_readiness_route_ready_when_nothing_configured(monkeypatch):
    """A bare app with no state attrs and memory backends must stay ready."""
    client = _make_readiness_client(monkeypatch)

    response = client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["stream_bridge"] == HEALTH_NOT_CONFIGURED
    assert body["provisioner"] == HEALTH_NOT_CONFIGURED


def test_readiness_route_reports_provisioner_health_from_app_config(monkeypatch):
    monkeypatch.setattr(requests, "get", lambda url, timeout=None: _FakeResponse(200))
    client = _make_readiness_client(monkeypatch, provisioner_url="http://provisioner:8002/")

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["provisioner"] == HEALTH_OK


def test_readiness_route_provisioner_unreachable_does_not_degrade(monkeypatch):
    def _boom(url, timeout=None):
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "get", _boom)
    client = _make_readiness_client(monkeypatch, provisioner_url="http://provisioner:8002")

    response = client.get("/health/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["provisioner"] == HEALTH_UNREACHABLE
