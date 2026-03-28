TASK_CONFIG = {
    "task_id": "hard",
    "name": "Full Village Allocation with Anomalies",
    "description": (
        "Allocate all three welfare schemes — MGNREGA work days, PM Awas Yojana housing slots, "
        "and ration card upgrades — across 80 families in a large Telangana village. "
        "WARNING: 5 families in the data have fraudulent/inflated applications (is_anomaly=True "
        "in ground truth, but they appear legitimate from need_score alone). "
        "The agent must allocate resources fairly while detecting and EXCLUDING these fraudulent "
        "applications. Budget is tighter relative to family count. All eligibility rules apply."
    ),
    "difficulty": "hard",
    "num_families": 80,
    "schemes": ["mgnrega", "pm_awas", "ration_upgrades"],
    "include_anomalies": True,
    "max_steps": 3,
    "success_threshold": 0.55,
    "expected_agent_score": 0.61,
    "action_schema": {
        "mgnrega_allocation": "Dict[family_id: str, days: int]",
        "pm_awas_allocation": "List[family_id: str]",
        "ration_upgrade_allocation": "List[family_id: str]",
    },
    "hints": [
        "Anomaly families have inflated need_score but are actually well-off (high income, own land, have house).",
        "Cross-check: if a family has high need_score but also land > 1 acre AND income > 6000, suspect anomaly.",
        "PM Awas: only allocate to families with has_house == False.",
        "Ration upgrades: skip families already on 'pink' tier.",
        "Total MGNREGA days must not exceed 600; PM Awas slots must not exceed 8; ration upgrades must not exceed 12.",
        "Allocating to an anomaly family causes a -0.2 penalty per family.",
    ],
}
