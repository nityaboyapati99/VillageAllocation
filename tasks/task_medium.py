TASK_CONFIG = {
    "task_id": "medium",
    "name": "Multi-Scheme Allocation",
    "description": (
        "Allocate MGNREGA work days AND ration card upgrades across 40 families in a "
        "Warangal district village. Some families are eligible for one scheme but not both. "
        "Respect eligibility rules: do not upgrade ration cards of families already on the "
        "pink (highest) tier. Budget is tighter per head. No anomaly families present."
    ),
    "difficulty": "medium",
    "num_families": 40,
    "schemes": ["mgnrega", "ration_upgrades"],
    "include_anomalies": False,
    "max_steps": 3,
    "success_threshold": 0.65,
    "expected_agent_score": 0.74,
    "action_schema": {
        "mgnrega_allocation": "Dict[family_id: str, days: int]",
        "pm_awas_allocation": "List[family_id: str]",
        "ration_upgrade_allocation": "List[family_id: str]",
    },
    "hints": [
        "Do NOT allocate ration upgrades to families with current_ration_card == 'pink'.",
        "MGNREGA: prioritise landless families with low income.",
        "Ration upgrades: prioritise families on 'none' or 'white' with highest need_score.",
        "Total MGNREGA days must not exceed 350; ration upgrades must not exceed 8.",
        "Some high-need families may already be on pink ration — skip them for ration upgrade.",
    ],
}
