"""Ground-station server: authentication, origins, review handling (web/backend/server.py)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from esa.access import AuditLog, OperatorDirectory  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TOKEN_A, TOKEN_B = "a" * 32, "b" * 32
ALERT = {"track_id": 5, "label": "swimmer", "lat": 61.49, "lon": 23.76, "message": "swimmer detected (track 5)"}


@pytest.fixture()
def server(tmp_path):
    spec = importlib.util.spec_from_file_location("gcs_server", REPO / "web/backend/server.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gcs_server"] = mod
    spec.loader.exec_module(mod)
    mod.ARGS = mod.parse_args(["--replay", "unused.jsonl", "--reviews", str(tmp_path / "reviews.jsonl")])
    mod.ARGS.replay = None  # no replay task: frames are injected by the test
    mod.OPERATORS = OperatorDirectory({TOKEN_A: "alice", TOKEN_B: "bob"})
    yield mod
    sys.modules.pop("gcs_server", None)


def connect(client, token=TOKEN_A, origin="http://localhost:5173"):
    ws = client.websocket_connect("/ws", headers={"origin": origin} if origin else {})
    s = ws.__enter__()
    s.send_text(json.dumps({"type": "auth", "token": token}))
    return ws, s


def test_valid_token_gets_hello_with_server_side_name(server):
    with TestClient(server.app) as c:
        ws, s = connect(c)
        assert s.receive_json() == {"type": "hello", "operator": "alice"}
        ws.__exit__(None, None, None)


@pytest.mark.parametrize("token", ["wrong-token-wrong-token", "", None])
def test_bad_or_missing_token_is_closed_without_data(server, token):
    with TestClient(server.app) as c:
        with c.websocket_connect("/ws", headers={"origin": "http://localhost:5173"}) as s:
            s.send_text(json.dumps({"type": "auth", "token": token}))
            with pytest.raises(WebSocketDisconnect) as e:
                s.receive_text()
            assert e.value.code == 1008


def test_first_message_must_be_auth(server):
    with TestClient(server.app) as c:
        with c.websocket_connect("/ws") as s:
            s.send_text(json.dumps({"type": "review", "track_id": 5, "decision": "reject"}))
            with pytest.raises(WebSocketDisconnect):
                s.receive_text()


def test_foreign_origin_is_refused_at_handshake(server):
    with TestClient(server.app) as c:
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect("/ws", headers={"origin": "http://evil.example"}) as s:
                s.receive_text()


def test_review_is_attributed_to_the_token_owner_and_chained(server, tmp_path):
    server.on_frame(json.dumps({"frame": 1, "alerts": [ALERT]}))
    with TestClient(server.app) as c:
        ws, s = connect(c, TOKEN_B)
        s.receive_json()  # hello
        s.send_text(json.dumps({"type": "review", "track_id": 5, "decision": "reject", "operator": "alice"}))
        r = s.receive_json()
        ws.__exit__(None, None, None)
    assert r["operator"] == "bob" and r["decision"] == "reject" and r["label"] == "swimmer"
    log = tmp_path / "reviews.jsonl"
    assert AuditLog.verify(log) == (True, 1)
    assert json.loads(log.read_text())["operator"] == "bob"


def test_review_of_unknown_alert_is_ignored(server, tmp_path):
    with TestClient(server.app) as c:
        ws, s = connect(c)
        s.receive_json()
        s.send_text(json.dumps({"type": "review", "track_id": 404, "decision": "reject"}))
        ws.__exit__(None, None, None)
    assert server.STATE["reviews"] == {}
    assert not (tmp_path / "reviews.jsonl").exists() or (tmp_path / "reviews.jsonl").read_text() == ""


def test_alert_memory_is_bounded(server):
    server.MAX_ALERTS = 10
    for i in range(25):
        server.on_frame(json.dumps({"frame": i + 1, "alerts": [{**ALERT, "track_id": i}]}))
    assert list(server.STATE["alerts"]) == list(range(15, 25))


def test_health_does_not_reveal_the_tak_endpoint(server):
    with TestClient(server.app) as c:
        server.ARGS.tak = "tcp://10.0.0.5:8087"  # after startup, so no connection is attempted
        h = c.get("/health").json()
        server.ARGS.tak = None
    assert h["tak"] is True and "10.0.0.5" not in json.dumps(h)


def test_network_listen_needs_explicit_tokens(server):
    with pytest.raises(SystemExit):
        server.main(["--replay", "x.jsonl", "--host", "0.0.0.0"])
