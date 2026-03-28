"""Tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["env"] == "village-welfare-allocator"


def test_list_tasks_returns_three():
    response = client.get("/tasks")
    assert response.status_code == 200
    data = response.json()
    assert "tasks" in data
    assert len(data["tasks"]) == 3
    task_ids = {t["task_id"] for t in data["tasks"]}
    assert task_ids == {"easy", "medium", "hard"}


def test_list_tasks_have_action_schema():
    response = client.get("/tasks")
    data = response.json()
    for task in data["tasks"]:
        assert "action_schema" in task
        schema = task["action_schema"]
        assert "mgnrega_allocation" in schema
        assert "pm_awas_allocation" in schema
        assert "ration_upgrade_allocation" in schema


def test_reset_easy_returns_observation():
    response = client.post("/reset", json={"task_id": "easy"})
    assert response.status_code == 200
    data = response.json()
    assert data["task_id"] == "easy"
    assert len(data["families"]) == 20
    assert data["step_number"] == 0
    assert data["episode_done"] is False
    assert data["available_resources"]["mgnrega_days"] == 200


def test_reset_medium_returns_observation():
    response = client.post("/reset", json={"task_id": "medium"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["families"]) == 40


def test_reset_hard_returns_observation():
    response = client.post("/reset", json={"task_id": "hard"})
    assert response.status_code == 200
    data = response.json()
    assert len(data["families"]) == 80


def test_reset_invalid_task_returns_400():
    response = client.post("/reset", json={"task_id": "impossible"})
    assert response.status_code == 400


def test_step_with_valid_action_returns_reward():
    client.post("/reset", json={"task_id": "easy"})
    action = {
        "mgnrega_allocation": {"F001": 10, "F002": 15},
        "pm_awas_allocation": [],
        "ration_upgrade_allocation": [],
    }
    response = client.post("/step", json=action)
    assert response.status_code == 200
    data = response.json()
    assert "observation" in data
    assert "reward" in data
    assert "done" in data
    assert "info" in data
    reward = data["reward"]
    assert 0.0 <= reward["total_reward"] <= 1.0


def test_step_without_reset_returns_400():
    import api.app as api_module
    original = api_module._env
    api_module._env = None  # force uninitialised state
    try:
        fresh_client = TestClient(api_module.app)
        action = {"mgnrega_allocation": {}, "pm_awas_allocation": [], "ration_upgrade_allocation": []}
        response = fresh_client.post("/step", json=action)
        assert response.status_code == 400
    finally:
        api_module._env = original


def test_state_returns_dict_after_reset():
    client.post("/reset", json={"task_id": "easy"})
    response = client.get("/state")
    assert response.status_code == 200
    data = response.json()
    assert "village" in data
    assert "task_id" in data
    assert data["task_id"] == "easy"


def test_grader_endpoint():
    action = {
        "mgnrega_allocation": {"F001": 20, "F002": 15},
        "pm_awas_allocation": [],
        "ration_upgrade_allocation": [],
    }
    response = client.post("/grader", json={"action": action, "task_id": "easy"})
    assert response.status_code == 200
    data = response.json()
    assert "total_reward" in data
    assert "grader_score" in data
    assert 0.0 <= data["grader_score"] <= 1.0


def test_episode_completes_after_three_steps():
    client.post("/reset", json={"task_id": "easy"})
    action = {"mgnrega_allocation": {"F001": 10}, "pm_awas_allocation": [], "ration_upgrade_allocation": []}
    client.post("/step", json=action)
    client.post("/step", json=action)
    response = client.post("/step", json=action)
    assert response.status_code == 200
    data = response.json()
    assert data["done"] is True
