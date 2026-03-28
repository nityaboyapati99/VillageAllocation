from __future__ import annotations

from typing import Dict, List
from pydantic import BaseModel, model_validator


class Family(BaseModel):
    id: str                               # e.g. "F001"
    name: str                             # Indian name
    land_acres: float                     # 0.0 = landless
    monthly_income: int                   # in INR
    dependents: int                       # total family members
    has_sick_member: bool
    is_widow_headed: bool
    is_elderly_headed: bool               # head of family > 60
    current_ration_card: str              # "none", "white", "yellow", "pink"
    has_house: bool
    past_mgnrega_days_this_year: int
    debt_amount: int                      # in INR
    need_score: float                     # computed: 0.0-1.0, higher = more needy
    is_anomaly: bool = False              # True = inflated/fraudulent application (hard task only)

    @model_validator(mode="after")
    def clamp_need_score(self) -> "Family":
        self.need_score = max(0.0, min(1.0, self.need_score))
        return self


class Observation(BaseModel):
    village_name: str
    district: str
    state: str
    month: str
    task_id: str
    available_resources: Dict[str, int]   # {"mgnrega_days": int, "pm_awas_slots": int, "ration_upgrades": int}
    families: List[Family]
    step_number: int
    episode_done: bool
    message: str                          # human-readable context


class Action(BaseModel):
    mgnrega_allocation: Dict[str, int] = {}      # family_id -> days allocated
    pm_awas_allocation: List[str] = []           # list of family_ids getting housing
    ration_upgrade_allocation: List[str] = []    # list of family_ids getting upgrade


class Reward(BaseModel):
    total_reward: float                   # final 0.0-1.0 score
    need_coverage_score: float            # did high-need families get priority?
    fairness_score: float                 # was distribution equitable?
    eligibility_score: float              # were scheme rules respected?
    anomaly_detection_score: float        # did agent catch fraudulent applications?
    budget_adherence_score: float         # did agent stay within limits?
    penalties: List[str]                  # list of penalty reasons
    breakdown: dict                       # detailed per-family scores
