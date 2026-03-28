from __future__ import annotations

import json
import os
import random
from typing import Any, Dict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TELANGANA_DISTRICTS = [
    "Nalgonda", "Khammam", "Warangal", "Nizamabad", "Mahbubnagar",
    "Adilabad", "Karimnagar", "Medak", "Rangareddy", "Suryapet",
]

VILLAGE_NAMES = [
    "Kondapuram", "Rajampet", "Yellareddy", "Burgupalli", "Thimmapur",
    "Maddur", "Peddapalli", "Gangapur", "Sircilla", "Bhongir",
    "Kodangal", "Wanaparthy", "Gadwal", "Narayanpet", "Jogulamba",
]

INDIAN_FIRST_NAMES = [
    "Ramaiah", "Laxmaiah", "Venkatesh", "Suresh", "Naresh",
    "Ravi", "Srinivas", "Rajesh", "Mahesh", "Ganesh",
    "Lakshmi", "Savitri", "Padma", "Radha", "Anitha",
    "Kavitha", "Sunitha", "Rekha", "Usha", "Vijaya",
    "Yellaiah", "Balaiah", "Sailu", "Komuraiah", "Prakash",
    "Mallaiah", "Shankar", "Devaiah", "Narsaiah", "Tirupati",
    "Bhavani", "Saraswathi", "Kamala", "Meenakshi", "Bhagyalaxmi",
    "Kotaiah", "Ramulu", "Govindaiah", "Venkataramaiah", "Linga",
]

INDIAN_LAST_NAMES = [
    "Reddy", "Naidu", "Goud", "Yadav", "Sharma",
    "Verma", "Patel", "Nair", "Pillai", "Raju",
    "Babu", "Rao", "Kumar", "Singh", "Prasad",
    "Swamy", "Murthy", "Chary", "Gari", "Wala",
]

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

RATION_TIERS = ["none", "white", "yellow", "pink"]

# Resource scaling by task size
RESOURCE_TABLE = {
    20:  {"mgnrega_days": 200, "pm_awas_slots": 3,  "ration_upgrades": 5},
    40:  {"mgnrega_days": 350, "pm_awas_slots": 5,  "ration_upgrades": 8},
    80:  {"mgnrega_days": 600, "pm_awas_slots": 8,  "ration_upgrades": 12},
}


# ---------------------------------------------------------------------------
# Need score calculation
# ---------------------------------------------------------------------------

def compute_need_score(
    land_acres: float,
    monthly_income: int,
    dependents: int,
    has_sick_member: bool,
    is_widow_headed: bool,
    is_elderly_headed: bool,
    has_house: bool,
) -> float:
    """
    Compute need score (0.0 – 1.0) from family attributes.
    Maximum raw score = 1.00 (all flags triggered).
    """
    score = 0.0
    if land_acres == 0.0:
        score += 0.30
    if monthly_income < 3000:
        score += 0.25
    if dependents > 4:
        score += 0.15
    if has_sick_member:
        score += 0.10
    if is_widow_headed:
        score += 0.10
    if is_elderly_headed:
        score += 0.05
    if not has_house:
        score += 0.05
    # Clamp to [0, 1]
    return round(min(1.0, max(0.0, score)), 4)


# ---------------------------------------------------------------------------
# Family generator
# ---------------------------------------------------------------------------

def _random_name(rng: random.Random) -> str:
    return f"{rng.choice(INDIAN_FIRST_NAMES)} {rng.choice(INDIAN_LAST_NAMES)}"


def _generate_family(
    family_index: int,
    rng: random.Random,
    is_anomaly: bool = False,
) -> Dict[str, Any]:
    fid = f"F{family_index:03d}"

    land_acres = rng.choice([0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    monthly_income = rng.choice([
        1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 8000, 10000,
    ])
    dependents = rng.randint(1, 8)
    has_sick_member = rng.random() < 0.25
    is_widow_headed = rng.random() < 0.15
    is_elderly_headed = rng.random() < 0.20
    current_ration_card = rng.choice(RATION_TIERS)
    has_house = rng.random() < 0.55
    past_mgnrega_days = rng.randint(0, 90)
    debt_amount = rng.choice([0, 5000, 10000, 20000, 50000, 100000])

    raw_need = compute_need_score(
        land_acres, monthly_income, dependents,
        has_sick_member, is_widow_headed, is_elderly_headed, has_house,
    )

    # Anomaly: inflate need score to look very needy, but family is actually well-off
    if is_anomaly:
        # Override with well-off attributes but keep need_score high (fraudulent)
        land_acres = rng.choice([1.5, 2.0, 2.5, 3.0])
        monthly_income = rng.choice([7000, 8000, 9000, 10000])
        has_house = True
        current_ration_card = "pink"  # already on best tier
        raw_need = round(rng.uniform(0.72, 0.92), 4)  # inflated

    return {
        "id": fid,
        "name": _random_name(rng),
        "land_acres": land_acres,
        "monthly_income": monthly_income,
        "dependents": dependents,
        "has_sick_member": has_sick_member,
        "is_widow_headed": is_widow_headed,
        "is_elderly_headed": is_elderly_headed,
        "current_ration_card": current_ration_card,
        "has_house": has_house,
        "past_mgnrega_days_this_year": past_mgnrega_days,
        "debt_amount": debt_amount,
        "need_score": raw_need,
        "is_anomaly": is_anomaly,
    }


# ---------------------------------------------------------------------------
# Public: generate_village
# ---------------------------------------------------------------------------

def generate_village(
    num_families: int,
    include_anomalies: bool,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate a complete village state dict with families and resources.

    Args:
        num_families:      20, 40, or 80
        include_anomalies: inject fraudulent applications (hard task)
        seed:              RNG seed for reproducibility

    Returns:
        Full village state dict compatible with Observation model.
    """
    if num_families not in RESOURCE_TABLE:
        raise ValueError(f"num_families must be one of {list(RESOURCE_TABLE.keys())}")

    rng = random.Random(seed)

    village_name = rng.choice(VILLAGE_NAMES)
    district = rng.choice(TELANGANA_DISTRICTS)
    month = rng.choice(MONTHS)

    # Determine anomaly family indices (5 out of num_families for hard)
    anomaly_indices: set[int] = set()
    if include_anomalies:
        anomaly_count = 5
        anomaly_indices = set(rng.sample(range(1, num_families + 1), anomaly_count))

    families = []
    for i in range(1, num_families + 1):
        is_anomaly = i in anomaly_indices
        families.append(_generate_family(i, rng, is_anomaly=is_anomaly))

    resources = RESOURCE_TABLE[num_families]

    return {
        "village_name": village_name,
        "district": district,
        "state": "Telangana",
        "month": month,
        "available_resources": dict(resources),
        "families": families,
    }


# ---------------------------------------------------------------------------
# Public: generate_and_save_all  (called from Dockerfile & tests)
# ---------------------------------------------------------------------------

def generate_and_save_all(output_dir: str = "data/sample_villages") -> None:
    """Pre-generate all three village JSON files with seed=42."""
    os.makedirs(output_dir, exist_ok=True)

    configs = [
        ("easy_village.json",   20, False),
        ("medium_village.json", 40, False),
        ("hard_village.json",   80, True),
    ]

    for filename, num_families, include_anomalies in configs:
        village = generate_village(
            num_families=num_families,
            include_anomalies=include_anomalies,
            seed=42,
        )
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(village, f, indent=2, ensure_ascii=False)
        print(f"  Generated {path} ({num_families} families, anomalies={include_anomalies})")


if __name__ == "__main__":
    generate_and_save_all()
