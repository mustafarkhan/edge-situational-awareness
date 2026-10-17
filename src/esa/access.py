"""Access control for the ground station: who may connect, who may decide, and a log that shows tampering.

The ground station sees every casualty position and its operators' decisions change what rescuers see: a rejection
deletes the TAK marker and suppresses new alerts at that spot. So every WebSocket client must prove it holds an
operator token, browsers may only connect from allowed origins, review messages are validated, and each decision is
attributed to the operator the server authenticated (never a name the client typed) in a hash-chained audit log.

Token file format (one operator per line, ``#`` comments allowed)::

    alice  pV3pj0x8...   # name, then token
    bob    Zr7qkE2c...
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass, field
from pathlib import Path

REVIEW_DECISIONS = ("confirm", "reject")
DEFAULT_ORIGINS = ("http://localhost:5173", "http://127.0.0.1:5173")
MIN_TOKEN_LEN = 16


def new_token() -> str:
    """A random operator token (32 URL-safe characters, 192 bits)."""
    return secrets.token_urlsafe(24)


@dataclass
class OperatorDirectory:
    """Operator tokens -> operator names. Lookups take constant time per stored token."""

    tokens: dict[str, str] = field(default_factory=dict)  # token -> name

    @classmethod
    def single(cls, token: str, name: str = "operator") -> OperatorDirectory:
        return cls({_checked(token): name})

    @classmethod
    def from_file(cls, path: str | Path) -> OperatorDirectory:
        tokens: dict[str, str] = {}
        for n, raw in enumerate(Path(path).read_text().splitlines(), 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"{path}:{n}: expected '<name> <token>'")
            name, token = parts
            if _checked(token) in tokens:
                raise ValueError(f"{path}:{n}: token used twice")
            tokens[token] = name
        if not tokens:
            raise ValueError(f"{path}: no operators")
        return cls(tokens)

    def authenticate(self, token: object) -> str | None:
        """The operator's name if `token` is valid, else None."""
        if not isinstance(token, str) or not token:
            return None
        found = None
        for known, name in self.tokens.items():  # compare against every token: timing does not reveal which matched
            if hmac.compare_digest(known.encode(), token.encode()):
                found = name
        return found


def _checked(token: str) -> str:
    if len(token) < MIN_TOKEN_LEN:
        raise ValueError(f"operator tokens must be at least {MIN_TOKEN_LEN} characters")
    return token


def origin_allowed(origin: str | None, allowed: tuple[str, ...] | list[str]) -> bool:
    """Browsers always send Origin, so a page on another site cannot use an operator's open map session.

    Clients that are not browsers (scripts, tests) send none; they still need a token.
    """
    if origin is None:
        return True
    return origin.rstrip("/") in {o.rstrip("/") for o in allowed}


def is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "::1", "localhost")


def parse_review(msg: object, known_alert_uids) -> tuple[str, str] | None:
    """(alert uid, decision) for a well-formed review of a known alert, else None.

    Only these two fields are read. Anything else the client sends (an operator name, coordinates, a label) is
    ignored: those come from the server's own records.
    """
    if not isinstance(msg, dict) or msg.get("type") != "review":
        return None
    uid, decision = msg.get("uid"), msg.get("decision")
    if not isinstance(uid, str) or len(uid) > 64 or decision not in REVIEW_DECISIONS:
        return None
    if uid not in known_alert_uids:
        return None
    return uid, decision


class AuditLog:
    """Append-only JSON lines; each record carries the SHA-256 of the previous line, so an edited, removed or
    reordered record breaks the chain (`verify`). It shows tampering; it does not prevent it."""

    GENESIS = "0" * 64

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._prev = self.GENESIS
        if self.path.exists():
            lines = self.path.read_text().splitlines()
            if lines:
                self._prev = hashlib.sha256(lines[-1].encode()).hexdigest()

    def append(self, record: dict) -> dict:
        rec = {**record, "prev": self._prev}
        line = json.dumps(rec, sort_keys=True)
        with self.path.open("a") as fh:
            fh.write(line + "\n")
        self._prev = hashlib.sha256(line.encode()).hexdigest()
        return rec

    @classmethod
    def verify(cls, path: str | Path) -> tuple[bool, int]:
        """(intact, number of records checked before the first break)."""
        prev, lines = cls.GENESIS, Path(path).read_text().splitlines()
        for n, line in enumerate(lines):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                return False, n
            if rec.get("prev") != prev:
                return False, n
            prev = hashlib.sha256(line.encode()).hexdigest()
        return True, len(lines)


class RateLimiter:
    """Token bucket per connection: `rate` messages per second, bursts up to `burst`."""

    def __init__(self, rate: float = 5.0, burst: int = 10):
        self.rate, self.burst = rate, burst
        self.tokens, self.t = float(burst), None

    def allow(self, now: float) -> bool:
        if self.t is not None:
            self.tokens = min(self.burst, self.tokens + (now - self.t) * self.rate)
        self.t = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
