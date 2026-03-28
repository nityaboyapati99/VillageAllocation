from __future__ import annotations

from typing import Any, Dict, List

from .models import Action, Family, Reward

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
W_NEED_COVERAGE = 0.40
W_FAIRNESS = 0.20
W_ELIGIBILITY = 0.20
W_ANOMALY = 0.10
W_BUDGET = 0.10

HIGHEST_RATION_TIER = "pink"


# ---------------------------------------------------------------------------
# Helper: Gini coefficient
# ---------------------------------------------------------------------------

def _gini(values: List[float]) -> float:
    """Return Gini coefficient (0 = perfect equality, 1 = perfect inequality)."""
    if not values or sum(values) == 0:
        return 0.0
    n = len(values)
    sorted_vals = sorted(values)
    cumulative = 0.0
    for i, v in enumerate(sorted_vals):
        cumulative += (2 * (i + 1) - n - 1) * v
    return cumulative / (n * sum(sorted_vals))


# ---------------------------------------------------------------------------
# Component scorers
# ---------------------------------------------------------------------------

def _need_coverage(
    action: Action,
    families: List[Family],
    available_resources: Dict[str, int],
) -> float:
    """
    Compare agent's allocation to ideal (top-N by need_score).
    Score = overlap / total_slots_used
    """
    family_map = {f.id: f for f in families}

    # Ideal: sort by need_score desc, pick top N for each scheme
    sorted_by_need = sorted(families, key=lambda f: f.need_score, reverse=True)

    scores: List[float] = []

    # MGNREGA — ideal is top families getting days proportional to need
    mgnrega_budget = available_resources.get("mgnrega_days", 0)
    if mgnrega_budget > 0 and action.mgnrega_allocation:
        agent_ids = set(action.mgnrega_allocation.keys())
        # Ideal recipients: landless families sorted by need
        eligible = [f for f in sorted_by_need if f.land_acres == 0.0] or sorted_by_need
        # How many families can realistically be served (avg ~10 days each)
        ideal_count = max(1, mgnrega_budget // 10)
        ideal_ids = {f.id for f in eligible[:ideal_count]}
        if ideal_ids:
            overlap = len(agent_ids & ideal_ids)
            scores.append(overlap / len(ideal_ids))

    # PM Awas — ideal is top-N families without a house
    pm_slots = available_resources.get("pm_awas_slots", 0)
    if pm_slots > 0:
        houseless = [f for f in sorted_by_need if not f.has_house] or sorted_by_need
        ideal_pm = {f.id for f in houseless[:pm_slots]}
        agent_pm = set(action.pm_awas_allocation)
        if ideal_pm:
            overlap = len(agent_pm & ideal_pm)
            scores.append(overlap / len(ideal_pm))

    # Ration upgrades — ideal is top-N families not already on pink
    ration_slots = available_resources.get("ration_upgrades", 0)
    if ration_slots > 0:
        upgradeable = [
            f for f in sorted_by_need
            if f.current_ration_card != HIGHEST_RATION_TIER
        ] or sorted_by_need
        ideal_ration = {f.id for f in upgradeable[:ration_slots]}
        agent_ration = set(action.ration_upgrade_allocation)
        if ideal_ration:
            overlap = len(agent_ration & ideal_ration)
            scores.append(overlap / len(ideal_ration))

    return round(sum(scores) / len(scores), 4) if scores else 0.0


def _fairness(action: Action, families: List[Family]) -> float:
    """Fairness via Gini coefficient on MGNREGA distribution among recipients."""
    if not action.mgnrega_allocation:
        return 0.5  # neutral — no days distributed

    days_list = list(action.mgnrega_allocation.values())
    gini = _gini([float(d) for d in days_list])
    # Convert: low Gini → high fairness
    return round(1.0 - gini, 4)


def _eligibility(
    action: Action,
    families: List[Family],
    penalties: List[str],
) -> float:
    family_map = {f.id: f for f in families}
    score = 1.0

    # PM Awas: penalise allocating to families who already have a house
    for fid in action.pm_awas_allocation:
        family = family_map.get(fid)
        if family and family.has_house:
            score -= 0.1
            penalties.append(
                f"PM Awas ineligible: {fid} ({family.name}) already has a house"
            )

    # Ration upgrade: penalise allocating to families already on pink tier
    for fid in action.ration_upgrade_allocation:
        family = family_map.get(fid)
        if family and family.current_ration_card == HIGHEST_RATION_TIER:
            score -= 0.1
            penalties.append(
                f"Ration ineligible: {fid} ({family.name}) already on pink tier"
            )

    return round(max(0.0, score), 4)


def _anomaly_detection(
    action: Action,
    families: List[Family],
    penalties: List[str],
) -> float:
    anomaly_ids = {f.id for f in families if f.is_anomaly}
    if not anomaly_ids:
        return 1.0  # no anomalies in this task → full score

    all_allocated = (
        set(action.mgnrega_allocation.keys())
        | set(action.pm_awas_allocation)
        | set(action.ration_upgrade_allocation)
    )

    allocated_anomalies = all_allocated & anomaly_ids
    score = 1.0

    for fid in allocated_anomalies:
        score -= 0.2
        penalties.append(f"Anomaly allocated: {fid} is a fraudulent application")

    if not allocated_anomalies:
        score = min(1.0, score + 0.1)  # bonus for catching all anomalies

    return round(max(0.0, score), 4)


def _budget_adherence(
    action: Action,
    available_resources: Dict[str, int],
    penalties: List[str],
) -> float:
    score = 1.0

    # MGNREGA days
    used_days = sum(action.mgnrega_allocation.values())
    budget_days = available_resources.get("mgnrega_days", 0)
    if used_days > budget_days:
        score -= 0.5
        penalties.append(
            f"Budget exceeded: MGNREGA used {used_days} > budget {budget_days}"
        )
    elif budget_days > 0 and used_days >= 0.9 * budget_days:
        score = min(1.0, score + 0.05)

    # PM Awas slots
    used_pm = len(action.pm_awas_allocation)
    budget_pm = available_resources.get("pm_awas_slots", 0)
    if used_pm > budget_pm:
        score -= 0.5
        penalties.append(
            f"Budget exceeded: PM Awas used {used_pm} > budget {budget_pm}"
        )

    # Ration upgrades
    used_ration = len(action.ration_upgrade_allocation)
    budget_ration = available_resources.get("ration_upgrades", 0)
    if used_ration > budget_ration:
        score -= 0.5
        penalties.append(
            f"Budget exceeded: Ration used {used_ration} > budget {budget_ration}"
        )

    return round(max(0.0, score), 4)


# ---------------------------------------------------------------------------
# Public: calculate_reward
# ---------------------------------------------------------------------------

def calculate_reward(
    action: Action,
    village_state: Dict[str, Any],
    task_config: Dict[str, Any] | None = None,
) -> Reward:
    """
    Compute all reward components and return a Reward object.

    Args:
        action:        Agent's allocation decisions.
        village_state: Dict with 'families' and 'available_resources'.
        task_config:   Optional task config dict (used to check task difficulty).
    """
    families = [
        f if isinstance(f, Family) else Family(**f)
        for f in village_state["families"]
    ]
    available_resources: Dict[str, int] = village_state["available_resources"]
    penalties: List[str] = []

    # Component scores
    need_cov = _need_coverage(action, families, available_resources)
    fairness = _fairness(action, families)
    eligibility = _eligibility(action, families, penalties)
    anomaly = _anomaly_detection(action, families, penalties)
    budget = _budget_adherence(action, available_resources, penalties)

    # Weighted total
    total = (
        W_NEED_COVERAGE * need_cov
        + W_FAIRNESS * fairness
        + W_ELIGIBILITY * eligibility
        + W_ANOMALY * anomaly
        + W_BUDGET * budget
    )
    total = round(max(0.0, min(1.0, total)), 4)

    breakdown = {
        "per_component": {
            "need_coverage": {"score": need_cov, "weight": W_NEED_COVERAGE},
            "fairness": {"score": fairness, "weight": W_FAIRNESS},
            "eligibility": {"score": eligibility, "weight": W_ELIGIBILITY},
            "anomaly_detection": {"score": anomaly, "weight": W_ANOMALY},
            "budget_adherence": {"score": budget, "weight": W_BUDGET},
        },
        "mgnrega_used": sum(action.mgnrega_allocation.values()),
        "mgnrega_budget": available_resources.get("mgnrega_days", 0),
        "pm_awas_used": len(action.pm_awas_allocation),
        "pm_awas_budget": available_resources.get("pm_awas_slots", 0),
        "ration_used": len(action.ration_upgrade_allocation),
        "ration_budget": available_resources.get("ration_upgrades", 0),
    }

    return Reward(
        total_reward=total,
        need_coverage_score=need_cov,
        fairness_score=fairness,
        eligibility_score=eligibility,
        anomaly_detection_score=anomaly,
        budget_adherence_score=budget,
        penalties=penalties,
        breakdown=breakdown,
    )
