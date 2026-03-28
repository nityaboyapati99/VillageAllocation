"""Tests for VillageWelfareEnv."""
import pytest

from environment.env import VillageWelfareEnv
from environment.models import Action, Observation, Reward


@pytest.fixture
def easy_env():
    env = VillageWelfareEnv("easy")
    env.reset()
    return env


def test_reset_returns_observation():
    env = VillageWelfareEnv("easy")
    obs = env.reset()
    assert isinstance(obs, Observation)
    assert obs.task_id == "easy"
    assert len(obs.families) == 20
    assert obs.step_number == 0
    assert obs.episode_done is False
    assert obs.village_name
    assert obs.available_resources["mgnrega_days"] == 200


def test_reset_returns_correct_families_for_each_task():
    for task_id, expected in [("easy", 20), ("medium", 40), ("hard", 80)]:
        env = VillageWelfareEnv(task_id)
        obs = env.reset()
        assert len(obs.families) == expected, f"{task_id} should have {expected} families"


def test_step_with_valid_action_returns_correct_types(easy_env):
    action = Action(
        mgnrega_allocation={"F001": 10, "F002": 15},
        pm_awas_allocation=[],
        ration_upgrade_allocation=[],
    )
    obs, reward, done, info = easy_env.step(action)
    assert isinstance(obs, Observation)
    assert isinstance(reward, Reward)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_step_increments_step_number(easy_env):
    assert easy_env._step_number == 0
    action = Action()
    easy_env.step(action)
    assert easy_env._step_number == 1
    easy_env.step(action)
    assert easy_env._step_number == 2
    easy_env.step(action)
    assert easy_env._step_number == 3
    assert easy_env._done is True


def test_step_done_after_three_steps(easy_env):
    action = Action()
    _, _, done1, _ = easy_env.step(action)
    _, _, done2, _ = easy_env.step(action)
    _, _, done3, _ = easy_env.step(action)
    assert done1 is False
    assert done2 is False
    assert done3 is True


def test_over_budget_action_applies_penalty(easy_env):
    # Exceed MGNREGA budget of 200 days
    action = Action(mgnrega_allocation={f"F{i:03d}": 50 for i in range(1, 6)})
    _, reward, _, _ = easy_env.step(action)
    assert any("Budget exceeded" in p for p in reward.penalties)
    assert reward.budget_adherence_score < 1.0


def test_state_returns_dict(easy_env):
    state = easy_env.state()
    assert isinstance(state, dict)
    assert "village" in state
    assert "task_id" in state
    assert "step_number" in state
    assert "done" in state


def test_step_after_done_raises(easy_env):
    action = Action()
    easy_env.step(action)
    easy_env.step(action)
    easy_env.step(action)
    with pytest.raises(RuntimeError):
        easy_env.step(action)


def test_reset_reinitialises_done_env():
    env = VillageWelfareEnv("easy")
    env.reset()
    action = Action()
    env.step(action)
    env.step(action)
    env.step(action)
    assert env._done is True
    # Should work after reset
    obs = env.reset()
    assert env._done is False
    assert obs.step_number == 0
