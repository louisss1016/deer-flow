"""Readiness probe helpers for the gateway health endpoints.

``GET /health`` stays a pure liveness signal: 200 whenever the process is up.
``GET /health/ready`` additionally probes the persistence the gateway actually
depends on, so orchestrators (Docker healthchecks, Kubernetes probes) treat the
gateway as ready only when the databases behind agent runs are reachable. Two
backends can be configured independently:

* the ORM engine behind ``database:`` (application repositories), and
* the effective LangGraph checkpointer/Store backend - the legacy
  ``checkpointer:`` section when present, otherwise derived from ``database:``
  (memory/sqlite/postgres).

Both probes run concurrently beneath a single endpoint-wide deadline
(:data:`_READINESS_DEADLINE_SECONDS`), so normal probe work completes within
one probe window rather than the sum of both budgets. Connection teardown is
ownership-critical and is drained to completion after cancellation, so a
stalled close may outlive the probe/deadline budget. The checkpointer config
is resolved once at startup from the same snapshot ``langgraph_runtime`` builds
its resources from and is stored on ``app.state``; probing a hot-reloaded
config instead could check a backend the running process is not using. A
``backend=memory`` deployment has nothing to probe and is always considered
ready; a startup config that cannot be resolved fails closed as unreachable.
Connection-opening probes are serialized behind a strict per-process gate: the
route is public through the ``/health`` auth prefix, so unlimited concurrent
requests must never translate into unlimited new PostgreSQL connections.

Two further runtime dependencies are probed when configured:

* the StreamBridge behind agent runs (``app.state.stream_bridge``): a Redis
  bridge answers a bounded PING, an in-process memory bridge has no external
  dependency and reports ``not_configured``; and
* the sandbox provisioner (``sandbox.provisioner_url``), probed with a bounded
  HTTP GET of the provisioner's own ``/health`` route.

The stream bridge degrades the endpoint like the database probes do: since
#6328 the Redis bridge is mandatory for multi-instance deployments, so an
unreachable bridge must fail readiness closed. The provisioner is reported in
the payload but never degrades the endpoint - every replica shares one
provisioner, so a blip there must not drain the whole service.
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
import weakref
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, Response
from sqlalchemy import text

from deerflow.config.app_config import get_app_config
from deerflow.persistence.engine import get_engine
from deerflow.runtime.stream_bridge.base import HEALTH_NOT_CONFIGURED, HEALTH_OK, HEALTH_UNREACHABLE
from deerflow.utils.file_io import await_drained

if TYPE_CHECKING:
    from deerflow.config.app_config import AppConfig
    from deerflow.config.checkpointer_config import CheckpointerConfig
    from deerflow.runtime.stream_bridge.base import StreamBridge

logger = logging.getLogger(__name__)

# Upper bound for connection/open/query work in a single probe attempt. Cleanup
# is ownership-critical: once a probe owns a connection, close is drained even
# if this timeout expires, so a stalled teardown may outlive this bound.
_PROBE_TIMEOUT_SECONDS = 2.0

# Whole-endpoint deadline for ordinary probe work. They run concurrently, so a
# healthy response normally completes within a single probe window; the extra
# margin absorbs scheduling/cancellation overhead. Owned connection teardown is
# intentionally exempt: ``await_drained`` defers cancellation until close
# finishes, so a wedged close can outlive this deadline while preserving
# resource ownership. Waiting requests are still shed by their own deadline.
# Orchestrator timeouts should therefore exceed this bound and may still fire
# first if teardown itself stalls.
_READINESS_DEADLINE_SECONDS = 3.0

# ``app.state`` attribute under which :func:`app.gateway.deps.langgraph_runtime`
# records the startup-bound checkpointer/Store config the probe targets.
READINESS_CHECKPOINTER_CONFIG_ATTR = "checkpointer_config"

# ``app.state`` attribute under which ``langgraph_runtime`` records the
# StreamBridge agent runs publish through (set in ``deps.langgraph_runtime``).
READINESS_STREAM_BRIDGE_ATTR = "stream_bridge"

# One gate per running event loop (one per worker process in production; one
# per test loop in the suite). ``/health/ready`` is public and unauthenticated,
# so a thundering herd of probes - or an attacker - must never be able to open
# an unbounded number of new connections: every connection-opening probe below
# is serialized through this gate, bounding in-flight probe connections to one
# per process. Waiting requests are still shed by the endpoint-wide deadline.
_PROBE_GATES: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = weakref.WeakKeyDictionary()


def _probe_gate() -> asyncio.Lock:
    """Return the serialization gate bound to the currently running loop."""
    loop = asyncio.get_running_loop()
    gate = _PROBE_GATES.get(loop)
    if gate is None:
        gate = asyncio.Lock()
        _PROBE_GATES[loop] = gate
    return gate


# Result vocabulary for the database probe.
DATABASE_OK = "ok"
DATABASE_NOT_CONFIGURED = "not_configured"
DATABASE_UNREACHABLE = "unreachable"


async def check_database_health() -> str:
    """Probe the persistence engine; return one of the DATABASE_* values."""
    engine = get_engine()
    if engine is None:
        # backend=memory, or the engine has not been initialized yet: there is
        # no database to probe.
        return DATABASE_NOT_CONFIGURED
    try:
        async with asyncio.timeout(_PROBE_TIMEOUT_SECONDS):
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
    except Exception:
        logger.warning("Readiness database probe failed", exc_info=True)
        return DATABASE_UNREACHABLE
    return DATABASE_OK


async def check_stream_bridge_health(stream_bridge: StreamBridge | None) -> str:
    """Probe the agent-run StreamBridge; return one of the HEALTH_* values.

    ``None`` (no bridge recorded on ``app.state``) means there is nothing to
    probe, mirroring ``backend=memory`` for the database probes. A bridge whose
    own ``health_check`` raises is reported unreachable rather than propagating:
    readiness must fail closed, never 500, and a custom bridge must not be able
    to take the endpoint down with an unexpected exception type.
    """
    if stream_bridge is None:
        return HEALTH_NOT_CONFIGURED
    try:
        return await stream_bridge.health_check()
    except Exception:
        logger.warning("Readiness probe: stream bridge health check raised", exc_info=True)
        return HEALTH_UNREACHABLE


async def check_provisioner_health(provisioner_url: str) -> str:
    """Probe the sandbox provisioner's own ``/health`` route.

    An empty URL means no provisioner is configured. The Gateway already talks
    to the provisioner with synchronous ``requests`` (see
    ``deerflow.community.aio_sandbox.remote_backend``), so the probe stays on
    the same client and is pushed onto a worker thread to keep the event loop
    free. Any transport error or non-200 response is ``unreachable``.
    """
    if not provisioner_url:
        return HEALTH_NOT_CONFIGURED
    try:
        import requests
    except ImportError:
        logger.error("Readiness probe: requests is not installed for the provisioner probe")
        return HEALTH_UNREACHABLE
    try:
        response = await asyncio.to_thread(
            requests.get,
            f"{provisioner_url.rstrip('/')}/health",
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.warning("Readiness provisioner probe failed", exc_info=True)
        return HEALTH_UNREACHABLE
    if response.status_code != 200:
        logger.warning("Readiness provisioner probe returned HTTP %s", response.status_code)
        return HEALTH_UNREACHABLE
    return HEALTH_OK


def resolve_checkpointer_config(startup_config: AppConfig) -> CheckpointerConfig | None:
    """Resolve the checkpointer/Store backend bound to a startup config snapshot.

    Mirrors the runtime's own selection (the legacy ``checkpointer`` section
    first, otherwise derived from the unified ``database`` section), so the
    probe targets the exact backend ``langgraph_runtime`` built at startup -
    which can differ from the ORM ``database:`` backend and from a later,
    hot-reloaded config. Returns None when the config cannot be resolved;
    callers must treat that as a failure (unreachable), never as
    ``not_configured``.
    """
    from deerflow.runtime.checkpointer.provider import _resolve_checkpointer_config

    try:
        return _resolve_checkpointer_config(startup_config)
    except Exception:
        logger.warning(
            "Readiness probe: unable to resolve the startup checkpointer config; failing closed",
            exc_info=True,
        )
        return None


def _sqlite_disk_uri(conn_str: str) -> str:
    """Return a non-creating (``mode=rw``) SQLite URI for a disk-backed database.

    Opening with ``mode=rw`` refuses to create a missing database file, so a
    readiness probe can never resurrect a checkpointer/Store file that was
    deleted or lost after startup - absence must surface as unreachable. The
    path is already absolute after
    ``deerflow.runtime.store._sqlite_utils.resolve_sqlite_conn_str`` and is
    converted with ``Path.as_uri`` for correct percent-encoding.
    """
    return f"{pathlib.Path(conn_str).as_uri()}?mode=rw"


async def _probe_sqlite_backend(conn_string: str | None) -> str:
    """Probe a SQLite checkpointer/Store database with bounded open/query work.

    Disk-backed databases are opened non-creating (``mode=rw``): a missing
    file stays missing and fails the probe instead of being recreated empty.
    ``:memory:`` only exists inside the running process, so there is nothing
    external to probe and it reports ``not_configured`` like the memory
    backend. A connection string the runtime refuses (a SQLite ``file:`` URI)
    is unreachable.
    """
    try:
        import aiosqlite
    except ImportError:
        logger.error("Readiness probe: aiosqlite is not installed for the sqlite checkpointer backend")
        return DATABASE_UNREACHABLE
    from deerflow.runtime.store._sqlite_utils import resolve_sqlite_conn_str

    try:
        conn_str = resolve_sqlite_conn_str(conn_string or "store.db")
    except ValueError as exc:
        logger.error("Readiness probe: %s", exc)
        return DATABASE_UNREACHABLE
    if conn_str == ":memory:":
        return DATABASE_NOT_CONFIGURED
    try:
        async with asyncio.timeout(_PROBE_TIMEOUT_SECONDS):
            connection = await aiosqlite.connect(_sqlite_disk_uri(conn_str), uri=True)
            try:
                await connection.execute("SELECT 1")
            finally:
                await await_drained(connection.close())
    except Exception:
        logger.warning("Readiness sqlite checkpointer probe failed", exc_info=True)
        return DATABASE_UNREACHABLE
    return DATABASE_OK


async def _probe_postgres_backend(conn_string: str, schema: str) -> str:
    """Probe PostgreSQL with bounded open/query work and drained connection teardown."""
    try:
        from psycopg import AsyncConnection
    except ImportError:
        logger.error("Readiness probe: psycopg is not installed for the postgres checkpointer backend")
        return DATABASE_UNREACHABLE
    try:
        from deerflow.persistence.postgres_schema import dsn_with_search_path, normalize_libpq_dsn

        dsn = dsn_with_search_path(normalize_libpq_dsn(conn_string), schema)
        async with asyncio.timeout(_PROBE_TIMEOUT_SECONDS):
            connection = await AsyncConnection.connect(dsn, connect_timeout=int(_PROBE_TIMEOUT_SECONDS))
            try:
                async with connection.cursor() as cursor:
                    await cursor.execute("SELECT 1")
            finally:
                await await_drained(connection.close())
    except Exception:
        logger.warning("Readiness postgres checkpointer probe failed", exc_info=True)
        return DATABASE_UNREACHABLE
    return DATABASE_OK


async def _probe_checkpointer_backend(config: CheckpointerConfig) -> str:
    """Probe the LangGraph checkpointer/Store backend described by *config*.

    *config* is the startup-bound snapshot (see :func:`resolve_checkpointer_config`);
    an in-process memory backend has nothing external to probe. Probes that
    open a connection (sqlite file, postgres) are serialized so concurrent
    unauthenticated requests cannot exhaust the database's connections.
    """
    if config.type == "memory":
        # In-process backend: there is nothing external to probe.
        return DATABASE_NOT_CONFIGURED
    if config.type not in ("sqlite", "postgres"):
        logger.warning("Readiness probe: unknown checkpointer backend %r", config.type)
        return DATABASE_UNREACHABLE
    async with _probe_gate():
        if config.type == "sqlite":
            return await _probe_sqlite_backend(config.connection_string)
        if not config.connection_string:
            return DATABASE_UNREACHABLE
        return await _probe_postgres_backend(config.connection_string, config.postgres_schema)


async def readiness_payload(
    checkpointer_config: CheckpointerConfig | None = None,
    stream_bridge: StreamBridge | None = None,
    provisioner_url: str | None = None,
) -> tuple[int, dict[str, str]]:
    """Return the (status_code, body) pair served by ``GET /health/ready``.

    Probes the runtime dependencies the gateway actually uses: the ORM engine
    behind ``database:`` (repositories), the effective LangGraph
    checkpointer/Store backend (the legacy ``checkpointer:`` section, otherwise
    derived from ``database:``), the agent-run StreamBridge, and - when
    ``sandbox.provisioner_url`` is configured - the sandbox provisioner's own
    ``/health`` route. The probes run concurrently beneath one endpoint-wide
    deadline, so normal probe work is bounded by the slowest single probe
    rather than their sum. Connection teardown is drained after cancellation to
    preserve ownership and can therefore extend the in-flight request beyond
    that deadline if close itself stalls.

    ``checkpointer_config`` is the startup snapshot recorded by
    ``langgraph_runtime``; None means no snapshot could be resolved, which
    fails closed as an unreachable backend rather than reporting ready.
    ``stream_bridge`` is the startup-recorded bridge; None means nothing to
    probe (``not_configured``), an unreachable Redis bridge degrades the
    endpoint because multi-instance deployments cannot serve runs without it.
    ``provisioner_url`` empty/None means no provisioner is configured; a
    configured but unhealthy provisioner is reported in the payload without
    degrading the endpoint - every replica shares one provisioner, so a blip
    there must not drain the whole service.
    """

    async def _probe_engine() -> str:
        return await check_database_health()

    async def _probe_checkpointer() -> str:
        if checkpointer_config is None:
            # Fail closed: without the startup-bound config we cannot know what
            # backend agent runs use, so readiness must not be claimed.
            logger.error("Readiness probe: no startup checkpointer config snapshot recorded; failing closed")
            return DATABASE_UNREACHABLE
        return await _probe_checkpointer_backend(checkpointer_config)

    async def _probe_stream_bridge() -> str:
        return await check_stream_bridge_health(stream_bridge)

    async def _probe_provisioner() -> str:
        return await check_provisioner_health(provisioner_url or "")

    try:
        async with asyncio.timeout(_READINESS_DEADLINE_SECONDS):
            database, checkpointer, stream_bridge_health, provisioner = await asyncio.gather(
                _probe_engine(),
                _probe_checkpointer(),
                _probe_stream_bridge(),
                _probe_provisioner(),
            )
    except TimeoutError:
        logger.error(
            "Readiness probes exceeded the %.1fs endpoint deadline",
            _READINESS_DEADLINE_SECONDS,
        )
        database = checkpointer = stream_bridge_health = provisioner = DATABASE_UNREACHABLE
    # The provisioner is deliberately absent: a shared dependency must not be
    # able to drain every replica from the Service on its own.
    degraded = DATABASE_UNREACHABLE in (database, checkpointer, stream_bridge_health)
    payload = {
        "status": "degraded" if degraded else "ready",
        "service": "deer-flow-gateway",
        "database": database,
        "checkpointer": checkpointer,
        "stream_bridge": stream_bridge_health,
        "provisioner": provisioner,
    }
    return (503 if degraded else 200, payload)


def register_readiness_route(app: FastAPI) -> None:
    """Register ``GET /health/ready`` on *app*.

    Extracted from the app factory so route-level tests can exercise the real
    handler - including its ``app.state`` wiring - without building the whole
    gateway. Every dependency is resolved per request, never captured at
    registration time: the checkpointer config and StreamBridge come from
    ``app.state`` (recorded by ``langgraph_runtime`` at startup), the
    provisioner URL from the app config, so a hot-reloaded
    ``sandbox.provisioner_url`` is picked up by the next probe.
    """

    @app.get("/health/ready", tags=["health"])
    async def readiness_check(request: Request, response: Response) -> dict[str, str]:
        """Readiness endpoint: 200 when the runtime dependencies are reachable.

        Probes the ORM engine behind ``database:``, the effective LangGraph
        checkpointer/Store backend (legacy ``checkpointer:`` section, otherwise
        derived from ``database:``), and the agent-run StreamBridge concurrently
        beneath one bounded deadline; a configured provisioner is reported
        without affecting the status code. The checkpointer config and bridge
        come from the startup snapshots recorded by ``langgraph_runtime`` (never
        hot-reloaded config), so orchestrators can gate on the gateway actually
        being ready rather than merely alive. Returns 503 with
        ``status: degraded`` when a probed backend is unreachable or the startup
        backend cannot be resolved.
        """
        checkpointer_config = getattr(request.app.state, READINESS_CHECKPOINTER_CONFIG_ATTR, None)
        stream_bridge = getattr(request.app.state, READINESS_STREAM_BRIDGE_ATTR, None)
        provisioner_url = getattr(get_app_config().sandbox, "provisioner_url", None) or ""
        status_code, payload = await readiness_payload(checkpointer_config, stream_bridge, provisioner_url)
        response.status_code = status_code
        return payload
