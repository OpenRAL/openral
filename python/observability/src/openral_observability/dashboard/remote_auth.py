"""Who is asking: the principal behind a dashboard request (ADR-0129 §6, Interface B).

Two listeners serve one app:

* **local** (default ``127.0.0.1:4318``, no auth): a loopback client is ``mode=local``,
  role ``operator``; any other client (the dashboard bound to a LAN address) is
  ``mode=lan``, role ``viewer``, and may only press E-STOP.
* **remote** (``127.0.0.1:4319``, the per-robot Cloudflare Tunnel's target): every
  request carries ``Authorization: Bearer <JWT>`` minted by openral.com for this robot.
  ``verify`` checks it in Interface B's order; the JWT's ``role`` is the role.

A request reaches the app through the remote listener only via ``remote_listener``,
which tags the ASGI scope; nothing a client sends can set that tag. Authorization
headers and viz tickets are never logged.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import time
from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal  # Any: ASGI scope values and JWT claims are untyped

import httpx
import jwt
import structlog
from openral_core import RobotEnrolment, enrolment_path
from pydantic import ValidationError

__all__ = [
    "ROLE_RANK",
    "JwksCache",
    "Principal",
    "TokenVerifier",
    "VizTickets",
    "allowed_origins",
    "bearer",
    "is_remote",
    "jwks_cache_file",
    "load_enrolment",
    "remote_listener",
]

_logger = structlog.get_logger(__name__)

Role = Literal["viewer", "operator", "admin"]
ROLE_RANK: dict[str, int] = {"viewer": 0, "operator": 1, "admin": 2}
"""Role order; admin acts as operator on the robot (ADR-0129 §6)."""

_REMOTE_SCOPE_KEY = "openral.remote"
_LEEWAY_S = 60
_MAX_LIFETIME_S = 300
_JWKS_COOLDOWN_S = 60.0
_JWKS_MAX_AGE_S = 3600.0
_REQUIRED_CLAIMS = ("sub", "jti", "org", "role", "robot")

_ASGIApp = Callable[
    [
        MutableMapping[str, Any],
        Callable[[], Awaitable[MutableMapping[str, Any]]],
        Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ],
    Awaitable[None],
]


@dataclass(frozen=True)
class Principal:
    """The caller of one request.

    Attributes:
        mode: ``local`` (loopback), ``lan`` (another host, local listener) or ``remote``.
        role: What it may do (``ROLE_RANK``).
        sub: The openral.com user id (remote only).
        jti: The token id, recorded with every action (remote only).
        exp: Unix time the token expires; streams close then (remote only).
    """

    mode: Literal["local", "lan", "remote"]
    role: Role
    sub: str | None = None
    jti: str | None = None
    exp: float | None = None


def remote_listener(app: _ASGIApp) -> _ASGIApp:
    """Wrap ``app`` for the remote (tunnel) listener: its requests need a token."""

    async def tagged(
        scope: MutableMapping[str, Any],
        receive: Callable[[], Awaitable[MutableMapping[str, Any]]],
        send: Callable[[MutableMapping[str, Any]], Awaitable[None]],
    ) -> None:
        await app({**scope, _REMOTE_SCOPE_KEY: True}, receive, send)

    return tagged


def is_remote(scope: MutableMapping[str, Any]) -> bool:
    """Whether a request came in through ``remote_listener``."""
    return scope.get(_REMOTE_SCOPE_KEY) is True


def load_enrolment(path: Path | None = None) -> RobotEnrolment | None:
    """The enrolment ``openral robot link`` wrote, or ``None`` when the robot is not linked.

    Raises:
        ValueError: The file exists but is not a valid enrolment.
    """
    path = path if path is not None else enrolment_path()
    if not path.is_file():
        return None
    try:
        return RobotEnrolment.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ValueError(f"{path}: not a robot enrolment: {exc}") from exc


class JwksCache:
    """The issuer's signing keys, cached on disk; refetched for an unknown ``kid``, and hourly.

    At most one refetch per ``cooldown_s`` (Interface B: once a minute), so a stream of
    tokens with made-up ``kid`` values cannot turn the robot into a request amplifier.
    The hourly refetch (the first token after a start included) drops a key the issuer
    retired; while the issuer is unreachable the keys held keep verifying.
    """

    def __init__(self, url: str, cache_file: Path, *, cooldown_s: float = _JWKS_COOLDOWN_S) -> None:
        """Load whatever is cached at ``cache_file``; fetch nothing yet."""
        self._url = url
        self._cache_file = cache_file
        self._cooldown_s = cooldown_s
        self._last_fetch = -float("inf")
        self._keys: dict[str, jwt.PyJWK] = {}
        self._lock = asyncio.Lock()
        if cache_file.is_file():
            try:
                self._keys = self._parse(cache_file.read_text(encoding="utf-8"))
                # A restart keeps the cache's age: under an hour old, no fetch.
                age = max(0.0, time.time() - cache_file.stat().st_mtime)
                self._last_fetch = time.monotonic() - age
            except (ValueError, jwt.PyJWTError) as exc:
                _logger.warning(
                    "dashboard.jwks_cache_unreadable", path=str(cache_file), error=str(exc)
                )

    @staticmethod
    def _parse(text: str) -> dict[str, jwt.PyJWK]:
        keys = jwt.PyJWKSet.from_json(text).keys
        return {k.key_id: k for k in keys if k.key_id is not None and k.algorithm_name == "ES256"}

    def _stale(self) -> bool:
        return time.monotonic() - self._last_fetch >= _JWKS_MAX_AGE_S

    async def key(self, kid: str) -> jwt.PyJWK | None:
        """The ES256 key ``kid``, refetching the JWKS if it is unknown or an hour old."""
        if kid in self._keys and not self._stale():
            return self._keys[kid]
        async with self._lock:
            now = time.monotonic()
            if not self._stale() and (
                kid in self._keys or now - self._last_fetch < self._cooldown_s
            ):
                return self._keys.get(kid)
            self._last_fetch = now
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(self._url)
                resp.raise_for_status()
                self._keys = self._parse(resp.text)
            except (httpx.HTTPError, ValueError, jwt.PyJWTError) as exc:
                _logger.warning("dashboard.jwks_fetch_failed", url=self._url, error=str(exc))
                return self._keys.get(kid)
            try:
                await asyncio.to_thread(self._write_cache, resp.text)
            except OSError as exc:
                _logger.warning("dashboard.jwks_cache_write_failed", error=str(exc))
            return self._keys.get(kid)

    def _write_cache(self, text: str) -> None:
        self._cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache_file.write_text(text, encoding="utf-8")


class TokenVerifier:
    """Checks an openral.com robot token in Interface B's order."""

    def __init__(self, enrolment: RobotEnrolment, jwks: JwksCache) -> None:
        """Bind to this robot's enrolment and its issuer's keys."""
        self.enrolment = enrolment
        self._jwks = jwks

    async def verify(self, token: str) -> Principal | str:  # noqa: PLR0911  # reason: one return per Interface B check, in its order
        """The principal ``token`` authenticates, or why it does not (never the token)."""
        try:
            header = jwt.get_unverified_header(token)
        except jwt.DecodeError:
            return "malformed token"
        # 1. Header: ES256 only (never "none" or HMAC), typ absent or at+jwt, a known kid.
        if header.get("alg") != "ES256":
            return "alg must be ES256"
        if header.get("typ") not in (None, "at+jwt"):
            return "typ must be absent or at+jwt"
        kid = header.get("kid")
        key = await self._jwks.key(kid) if isinstance(kid, str) else None
        if key is None:
            return "unknown signing key"
        # 2.-5. Signature, iss, a single aud equal to our origin, exp/iat with leeway.
        try:
            claims: dict[str, Any] = jwt.decode(
                token,
                key=key,
                algorithms=["ES256"],
                issuer=self.enrolment.iss,
                audience=self.enrolment.aud,
                leeway=_LEEWAY_S,
                options={"require": ["exp", "iat", "iss", "aud"], "strict_aud": True},
            )
        except jwt.InvalidTokenError as exc:
            return f"invalid token: {type(exc).__name__}"
        if claims["exp"] - claims["iat"] > _MAX_LIFETIME_S:
            return f"token lifetime over {_MAX_LIFETIME_S} s"
        # 6. Required claims, all strings (org may be "" for a free robot).
        if any(not isinstance(claims.get(c), str) for c in _REQUIRED_CLAIMS):
            return "missing claim"
        if not all(claims[c] for c in ("sub", "jti", "robot")):
            return "empty claim"
        # 7. Enrolment.
        if claims["robot"] != self.enrolment.robot or claims["org"] != self.enrolment.org:
            return "token is for another robot"
        # 8. Role.
        role = claims["role"]
        if role not in ROLE_RANK:
            return "unknown role"
        return Principal(
            mode="remote", role=role, sub=claims["sub"], jti=claims["jti"], exp=float(claims["exp"])
        )


@dataclass(frozen=True)
class _Ticket:
    sub: str | None
    role: Role
    expires_at: float


class VizTickets:
    """Random 256-bit tickets for the 3D view (``/ws``).

    A WebSocket and an iframe URL cannot carry a bearer token, and a JWT never goes in
    a URL (RFC 9700 §4.3.2), so the robot issues its own: bound to the caller's
    ``sub``/role, reusable for Lichtblick's reconnects until it expires.
    """

    TTL_S = 900
    MAX = 256
    """Outstanding tickets kept; past it the oldest go (each viewer needs one at a time)."""

    def __init__(self) -> None:
        """No tickets yet."""
        self._tickets: dict[str, _Ticket] = {}

    def issue(self, principal: Principal) -> str:
        """A new ticket for ``principal``, valid ``TTL_S`` seconds."""
        now = time.time()
        if len(self._tickets) >= self.MAX:
            self._tickets = {t: v for t, v in self._tickets.items() if v.expires_at > now}
            while len(self._tickets) >= self.MAX:  # dicts keep issue order: oldest first
                del self._tickets[next(iter(self._tickets))]
        ticket = secrets.token_urlsafe(32)
        self._tickets[ticket] = _Ticket(principal.sub, principal.role, now + self.TTL_S)
        return ticket

    def expires_at(self, ticket: str | None) -> float | None:
        """When ``ticket`` expires, or ``None`` if it is unknown or expired."""
        found = self._tickets.get(ticket or "")
        return found.expires_at if found is not None and found.expires_at > time.time() else None


def jwks_cache_file() -> Path:
    """Where the issuer's JWKS is cached (under ``$OPENRAL_CACHE_DIR``)."""
    base = Path(os.environ.get("OPENRAL_CACHE_DIR") or Path.home() / ".cache" / "openral")
    return base / "dashboard" / "jwks.json"


def bearer(authorization: str | None) -> str | None:
    """The token of an ``Authorization: Bearer <token>`` header."""
    scheme, _, token = (authorization or "").partition(" ")
    return (token.strip() or None) if scheme.lower() == "bearer" else None


def allowed_origins() -> frozenset[str]:
    """``$OPENRAL_DASHBOARD_ALLOWED_ORIGINS`` (comma-separated), else openral.com's origins."""
    raw = os.environ.get(
        "OPENRAL_DASHBOARD_ALLOWED_ORIGINS",
        "https://www.openral.com,https://dev.openral.com,https://viz.openral.com",
    )
    return frozenset(o.strip().rstrip("/") for o in raw.split(",") if o.strip())
