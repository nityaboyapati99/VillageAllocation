TASK_CONFIG = {
    "task_id": "easy",
    "name": "MGNREGA Allocation",
    "description": (
        "Allocate 200 MGNREGA work days fairly across 20 families in Kondapuram village, "
        "Nalgonda district. Focus on landless, low-income households with the highest need. "
        "Only MGNREGA days need to be allocated in this task. No anomaly families present."
    ),
    "difficulty": "easy",
    "num_families": 20,
    "schemes": ["mgnrega"],
    "include_anomalies": False,
    "max_steps": 3,
    "success_threshold": 0.7,
    "expected_agent_score": 0.82,
    "action_schema": {
        "mgnrega_allocation": "Dict[family_id: str, days: int]",
        "pm_awas_allocation": "List[family_id: str]",
        "ration_upgrade_allocation": "List[family_id: str]",
    },
    "hints": [
        "Prioritise families with land_acres == 0.0 (landless labourers).",
        "Families with monthly_income < 3000 INR should receive the most days.",
        "Distribute days proportionally to need_score, do not give all days to one family.",
        "Total MGNREGA days allocated must not exceed 200.",
    ],
}
