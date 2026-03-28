"""Tests for graders."""
import pytest

from environment.env import VillageWelfareEnv
from environment.models import Action
from environment.village_generator import generate_village
from graders import GRADER_MAP
from tasks import TASK_MAP


def _get_village_state(task_id: str) -> dict:
    env = VillageWelfareEnv(task_id)
    env.reset()
    return env.state()["village"]


def _perfect_action(village_state: dict) -> Action:
    """
    Build a near-perfect action:
    - MGNREGA: top landless families get days proportional to need_score, within budget
    - PM Awas: top houseless families
    - Ration: top non-pink families
    - No anomaly families included
    """
    families = village_state["families"]
    resources = village_state["available_resources"]

    sorted_fam = sorted(families, key=lambda f: f["need_score"], reverse=True)
    non_anomaly = [f for f in sorted_fam if not f.get("is_anomaly", False)]

    # MGNREGA
    mgnrega_budget = resources.get("mgnrega_days", 0)
    candidates = [f for f in non_anomaly if f["land_acres"] == 0.0] or non_anomaly
    total_need = sum(f["need_score"] for f in candidates) or 1.0
    mgnrega_alloc = {}
    remaining = mgnrega_budget
    for f in candidates:
        if remaining <= 0:
            break
        days = min(remaining, max(1, int((f["need_score"] / total_need) * mgnrega_budget)))
        if days > 0:
            mgnrega_alloc[f["id"]] = days
            remaining -= days

    # PM Awas
    pm_slots = resources.get("pm_awas_slots", 0)
    houseless = [f for f in non_anomaly if not f["has_house"]]
    pm_alloc = [f["id"] for f in houseless[:pm_slots]]

    # Ration
    ration_slots = resources.get("ration_upgrades", 0)
    upgradeable = [f for f in non_anomaly if f["current_ration_card"] != "pink"]
    ration_alloc = [f["id"] for f in upgradeable[:ration_slots]]

    return Action(
        mgnrega_allocation=mgnrega_alloc,
        pm_awas_allocation=pm_alloc,
        ration_upgrade_allocation=ration_alloc,
    )


def _random_action(village_state: dict) -> Action:
    """Deliberately bad action: give all budget to one family, violate eligibility."""
    families = village_state["families"]
    resources = village_state["available_resources"]

    # Give all MGNREGA days to one family (very unfair)
    first_id = families[0]["id"]
    mgnrega_alloc = {first_id: resources.get("mgnrega_days", 0)}

    # Allocate PM Awas to families who already have a house (ineligible)
    has_house = [f["id"] for f in families if f["has_house"]]
    pm_alloc = has_house[:resources.get("pm_awas_slots", 0)]

    # Allocate ration upgrades to families already on pink (ineligible)
    pink = [f["id"] for f in families if f["current_ration_card"] == "pink"]
    ration_alloc = pink[:resources.get("ration_upgrades", 0)]

    return Action(
        mgnrega_allocation=mgnrega_alloc,
        pm_awas_allocation=pm_alloc,
        ration_upgrade_allocation=ration_alloc,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
def test_perfect_allocation_scores_high(task_id):
    village = _get_village_state(task_id)
    action = _perfect_action(village)
    grader = GRADER_MAP[task_id]
    score = grader(action, village, TASK_MAP[task_id])
    assert score >= 0.65, f"{task_id}: expected >= 0.65, got {score}"


@pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
def test_bad_allocation_scores_low(task_id):
    village = _get_village_state(task_id)
    action = _random_action(village)
    grader = GRADER_MAP[task_id]
    score = grader(action, village, TASK_MAP[task_id])
    # Bad actions should score below 0.7
    assert score < 0.70, f"{task_id}: expected < 0.70, got {score}"


def test_anomaly_allocation_reduces_score():
    """Allocating to anomaly families must reduce the hard score."""
    village = _get_village_state("hard")
    families = village["families"]
    anomaly_ids = [f["id"] for f in families if f.get("is_anomaly", False)]
    assert anomaly_ids, "Hard village must have anomaly families"

    # Perfect action (avoids anomalies)
    good_action = _perfect_action(village)
    good_score = GRADER_MAP["hard"](good_action, village, TASK_MAP["hard"])

    # Bad action: include anomaly families in MGNREGA
    resources = village["available_resources"]
    mgnrega_alloc = {fid: 10 for fid in anomaly_ids}
    remaining = resources["mgnrega_days"] - sum(mgnrega_alloc.values())
    non_anomaly = [f for f in families if not f.get("is_anomaly", False)]
    for f in sorted(non_anomaly, key=lambda x: x["need_score"], reverse=True):
        if remaining <= 0:
            break
        mgnrega_alloc[f["id"]] = min(remaining, 10)
        remaining -= 10

    bad_action = Action(
        mgnrega_allocation=mgnrega_alloc,
        pm_awas_allocation=good_action.pm_awas_allocation,
        ration_upgrade_allocation=good_action.ration_upgrade_allocation,
    )
    bad_score = GRADER_MAP["hard"](bad_action, village, TASK_MAP["hard"])
    assert bad_score < good_score, (
        f"Anomaly allocation should reduce score: good={good_score}, bad={bad_score}"
    )


@pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
def test_graders_are_deterministic(task_id):
    """Same inputs must always return same score."""
    village = _get_village_state(task_id)
    action = _perfect_action(village)
    grader = GRADER_MAP[task_id]
    config = TASK_MAP[task_id]

    scores = [grader(action, village, config) for _ in range(5)]
    assert len(set(scores)) == 1, f"{task_id}: grader is not deterministic, got {scores}"


@pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
def test_score_in_valid_range(task_id):
    village = _get_village_state(task_id)
    action = _perfect_action(village)
    score = GRADER_MAP[task_id](action, village, TASK_MAP[task_id])
    assert 0.0 <= score <= 1.0
