"""
Grader for the Easy task: MGNREGA-only allocation, 20 families.

Deterministic, rule-based, no LLM dependency.
Score range: 0.0 – 1.0
"""
from __future__ import annotations

from typing import Any, Dict

from environment.models import Action
from environment.reward import calculate_reward


def grade(action: Action, village_state: Dict[str, Any], task_config: Dict[str, Any]) -> float:
    """
    Grade an easy-task action.

    Scoring focuses on:
      - Need coverage (top landless/low-income families getting MGNREGA days)
      - Fairness (equitable distribution, not all days to one family)
      - Budget adherence (must not exceed 200 days)
      - Eligibility is less critical for easy (no PM Awas / ration)

    Returns:
        float: score in [0.0, 1.0]
    """
    reward = calculate_reward(action, village_state, task_config)

    # For easy task: ignore anomaly score (none present), re-weight slightly
    # Base score from standard reward
    base = reward.total_reward

    # Bonus: if agent distributed days to >= 5 distinct families
    if len(action.mgnrega_allocation) >= 5:
        base = min(1.0, base + 0.02)

    # Penalty: if zero families were given MGNREGA days
    if not action.mgnrega_allocation:
        base = 0.0

    return round(max(0.0, min(1.0, base)), 4)
