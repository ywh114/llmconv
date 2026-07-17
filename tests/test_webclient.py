"""Tests for the web VN gateway."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from ara.webclient.server import create_app, _normalise_event


@pytest.fixture
def client():
    """TestClient with a mocked AgentProxy."""
    app = create_app(socket_path="sockets/test_agent.sock")
    with TestClient(app) as tc:
        yield tc


class TestNormaliseEvent:
    def test_scene_loaded(self) -> None:
        result = {"event": "scene_loaded", "scene": {"id": "x", "starting_location": "room", "characters": [{"name": "A"}]}}
        out = _normalise_event(result)
        assert out["type"] == "scene_loaded"
        assert out["scene_id"] == "x"

    def test_turn(self) -> None:
        result = {"event": "turn", "output": "hello"}
        out = _normalise_event(result)
        assert out["type"] == "turn"
        assert out["output"] == "hello"

    def test_story_complete(self) -> None:
        result = {"event": "story_complete"}
        out = _normalise_event(result)
        assert out["type"] == "story_complete"


class TestWebGateway:
    def test_static_index(self, client: TestClient) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        assert "<!DOCTYPE html>" in resp.text

    def test_start_endpoint(self, client: TestClient) -> None:
        with patch.object(
            client.app.state.proxy, "start", return_value={"finished": False}
        ) as mock_start:
            resp = client.post("/start", json={})
            assert resp.status_code == 200
            assert resp.json()["finished"] is False
            mock_start.assert_called_once_with(scene_id=None)

    def test_start_with_scene_id(self, client: TestClient) -> None:
        with patch.object(
            client.app.state.proxy, "start", return_value={"finished": False}
        ) as mock_start:
            resp = client.post("/start", json={"scene_id": "tea_scene"})
            assert resp.status_code == 200
            mock_start.assert_called_once_with(scene_id="tea_scene")

    def test_step_endpoint(self, client: TestClient) -> None:
        with patch.object(
            client.app.state.proxy, "step", return_value={"event": "turn", "output": "hi"}
        ):
            resp = client.post("/step", json={})
            assert resp.status_code == 200
            assert resp.json()["type"] == "turn"

    def test_input_endpoint(self, client: TestClient) -> None:
        with patch.object(
            client.app.state.proxy, "input", return_value={"submitted": "hello"}
        ) as mock_input:
            resp = client.post("/input", json={"text": "hello"})
            assert resp.status_code == 200
            assert resp.json()["submitted"] == "hello"
            mock_input.assert_called_once_with(text="hello", attempt=None)

    def test_debug_endpoint(self, client: TestClient) -> None:
        with patch.object(
            client.app.state.proxy, "debug", return_value={"here": ["A"]}
        ) as mock_debug:
            resp = client.post("/debug", json={"command": "here", "args": []})
            assert resp.status_code == 200
            mock_debug.assert_called_once_with(command="here", args=[])

    def test_state_endpoint(self, client: TestClient) -> None:
        with patch.object(
            client.app.state.proxy, "state", return_value={"story": {}, "engine": {}}
        ):
            resp = client.get("/state")
            assert resp.status_code == 200

    def test_session_endpoint_inactive(self, client: TestClient) -> None:
        with patch.object(
            client.app.state.proxy, "continue_session", return_value={"active": False}
        ):
            resp = client.get("/session")
            assert resp.status_code == 200
            assert resp.json()["active"] is False

    def test_session_endpoint_active(self, client: TestClient) -> None:
        with patch.object(
            client.app.state.proxy,
            "continue_session",
            return_value={"active": True, "scene": {"asset_story_name": "demo"}},
        ):
            resp = client.get("/session")
            assert resp.status_code == 200
            assert resp.json() == {"active": True, "story_id": "demo"}

    def test_continue_endpoint_no_session(self, client: TestClient) -> None:
        with patch.object(
            client.app.state.proxy, "continue_session", return_value={"active": False}
        ):
            resp = client.post("/continue")
            assert resp.status_code == 409
            assert resp.json()["active"] is False

    def test_continue_endpoint_establishes_session(self, client: TestClient) -> None:
        visual = {"active": True, "scene": {"asset_story_name": "demo"}, "history": []}
        with patch.object(
            client.app.state.proxy, "continue_session", return_value=visual
        ):
            resp = client.post("/continue")
            assert resp.status_code == 200
            token = resp.json().get("session_token")
            assert token
        # The fresh token unlocks mutating endpoints.
        with patch.object(
            client.app.state.proxy, "input", return_value={"ok": True}
        ):
            resp = client.post(
                "/input", json={"text": "hi"}, headers={"x-session-token": token}
            )
            assert resp.status_code == 200
