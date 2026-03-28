"""
Grader for the Hard task: all three schemes, 80 families, 5 anomaly families.

Deterministic, rule-based, no LLM dependency.
Score range: 0.0 – 1.0
"""
from __future__ import annotations

from typing import Any, Dict

from environment.models import Action
from environment.reward import calculate_reward


def grade(action: Action, village_state: Dict[str, Any], task_config: Dict[str, Any]) -> float:
    """
    Grade a hard-task action.

    Fraud detection is heavily weighted here:
      - Each anomaly family allocated = -0.2
      - All anomalies avoided = +0.1 bonus

    All three schemes must be attempted; missing any scheme = penalty.

    Returns:
        float: score in [0.0, 1.0]
    """
    reward = calculate_reward(action, village_state, task_config)
    score = reward.total_reward

    # Penalise missing any scheme
    if not action.mgnrega_allocation:
        score -= 0.08
    if not action.pm_awas_allocation:
        score -= 0.08
    if not action.ration_upgrade_allocation:
        score -= 0.08

    # Anomaly detection already baked into reward; surface penalty count for clarity
    anomaly_penalty_count = sum(
        1 for p in reward.penalties if p.startswith("Anomaly allocated")
    )
    if anomaly_penalty_count == 0:
        # Perfect anomaly avoidance — small additional bonus
        score = min(1.0, score + 0.03)

    return round(max(0.0, min(1.0, score)), 4)
