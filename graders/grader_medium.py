"""
Grader for the Medium task: MGNREGA + ration upgrades, 40 families.

Deterministic, rule-based, no LLM dependency.
Score range: 0.0 – 1.0
"""
from __future__ import annotations

from typing import Any, Dict

from environment.models import Action
from environment.reward import calculate_reward


def grade(action: Action, village_state: Dict[str, Any], task_config: Dict[str, Any]) -> float:
    """
    Grade a medium-task action.

    Additional checks beyond base reward:
      - Penalise if ration upgrades list is empty (should attempt ration allocation)
      - Penalise if MGNREGA allocation is empty
      - Bonus for using both schemes

    Returns:
        float: score in [0.0, 1.0]
    """
    reward = calculate_reward(action, village_state, task_config)
    score = reward.total_reward

    # Penalise complete non-participation in a scheme
    if not action.mgnrega_allocation:
        score -= 0.10

    if not action.ration_upgrade_allocation:
        score -= 0.10

    # Bonus for attempting both schemes
    if action.mgnrega_allocation and action.ration_upgrade_allocation:
        score = min(1.0, score + 0.02)

    return round(max(0.0, min(1.0, score)), 4)
