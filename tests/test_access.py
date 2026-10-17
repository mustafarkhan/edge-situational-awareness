"""Access control for the ground station (esa.access)."""

import json

import pytest

from esa.access import (
    AuditLog,
    OperatorDirectory,
    RateLimiter,
    is_loopback,
    new_token,
    origin_allowed,
    parse_review,
)


def test_tokens_authenticate_to_their_operator_only():
    a, b = new_token(), new_token()
    d = OperatorDirectory({a: "alice", b: "bob"})
    assert d.authenticate(a) == "alice" and d.authenticate(b) == "bob"
    for bad in (None, "", 123, a[:-1], a + "x", b"bytes"):
        assert d.authenticate(bad) is None


def test_short_tokens_are_refused():
    with pytest.raises(ValueError):
        OperatorDirectory.single("short")


def test_token_file(tmp_path):
    a, b = new_token(), new_token()
    f = tmp_path / "ops.txt"
    f.write_text(f"# rescue team\nalice {a}\n\nbob {b}  # night shift\n")
    d = OperatorDirectory.from_file(f)
    assert d.authenticate(a) == "alice" and d.authenticate(b) == "bob"
    f.write_text(f"alice {a}\nbob {a}\n")
    with pytest.raises(ValueError, match="twice"):
        OperatorDirectory.from_file(f)
    f.write_text("alice\n")
    with pytest.raises(ValueError):
        OperatorDirectory.from_file(f)


def test_origins():
    allowed = ["http://localhost:5173", "https://gcs.local"]
    assert origin_allowed("http://localhost:5173", allowed)
    assert origin_allowed("https://gcs.local/", allowed)
    assert origin_allowed(None, allowed)  # not a browser: still needs a token
    for bad in ("http://evil.example", "http://localhost:5174", "null", "https://localhost:5173"):
        assert not origin_allowed(bad, allowed)


def test_loopback():
    assert all(is_loopback(h) for h in ("127.0.0.1", "::1", "localhost"))
    assert not any(is_loopback(h) for h in ("0.0.0.0", "192.168.1.10", "::"))


def test_review_messages_are_validated():
    known = {"rgb-3", "thermal-7"}
    assert parse_review({"type": "review", "uid": "rgb-3", "decision": "reject"}, known) == ("rgb-3", "reject")
    assert parse_review({"type": "review", "uid": "thermal-7", "decision": "confirm", "operator": "x"}, known) == (
        "thermal-7",
        "confirm",
    )
    for bad in (
        {"type": "review", "uid": "rgb-99", "decision": "reject"},  # unknown alert
        {"type": "review", "uid": 3, "decision": "reject"},  # wrong type
        {"type": "review", "track_id": 3, "decision": "reject"},  # the old form
        {"type": "review", "uid": "rgb-3" + "x" * 80, "decision": "reject"},  # too long
        {"type": "review", "uid": "rgb-3", "decision": "delete"},
        {"type": "auth", "uid": "rgb-3", "decision": "reject"},
        ["review", "rgb-3", "reject"],
        None,
    ):
        assert parse_review(bad, known) is None


def test_audit_log_detects_edits_and_removals(tmp_path):
    p = tmp_path / "reviews.jsonl"
    log = AuditLog(p)
    for i in range(4):
        log.append({"track_id": i, "decision": "reject", "operator": "alice"})
    assert AuditLog.verify(p) == (True, 4)

    AuditLog(p).append({"track_id": 9, "decision": "confirm", "operator": "bob"})  # reopened: chain continues
    assert AuditLog.verify(p) == (True, 5)

    lines = p.read_text().splitlines()
    edited = json.loads(lines[1])
    edited["decision"] = "confirm"
    p.write_text("\n".join([lines[0], json.dumps(edited, sort_keys=True), *lines[2:]]) + "\n")
    assert AuditLog.verify(p) == (False, 2)  # the record after the edit no longer chains

    p.write_text("\n".join([lines[0], *lines[2:]]) + "\n")  # a record removed
    assert AuditLog.verify(p)[0] is False


def test_rate_limiter():
    r = RateLimiter(rate=2.0, burst=3)
    assert [r.allow(0.0) for _ in range(4)] == [True, True, True, False]
    assert r.allow(0.5) and not r.allow(0.5)  # 0.5 s at 2/s refills one
