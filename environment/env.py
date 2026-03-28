from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

from .models import Action, Observation, Reward
from .reward import calculate_reward
from .village_generator import generate_village

# Maps task_id → (num_families, include_anomalies)
TASK_SPEC: Dict[str, Dict[str, Any]] = {
    "easy":   {"num_families": 20, "include_anomalies": False},
    "medium": {"num_families": 40, "include_anomalies": False},
    "hard":   {"num_families": 80, "include_anomalies": True},
}

# File paths for pre-generated data
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample_villages")
VILLAGE_FILES = {
    "easy":   "easy_village.json",
    "medium": "medium_village.json",
    "hard":   "hard_village.json",
}

STEP_MESSAGES = [
    "Step 1: Allocate MGNREGA work days to eligible families.",
    "Step 2: Allocate PM Awas Yojana housing slots to homeless families.",
    "Step 3: Allocate ration card upgrades to needy families.",
]


class VillageWelfareEnv:
    """
    OpenEnv-compliant environment for village welfare resource allocation.

    Episode design:
      - 3 steps per episode (one per scheme)
      - Step 1: MGNREGA allocation
      - Step 2: PM Awas allocation
      - Step 3: Ration card upgrades
      - Episode ends (done=True) after step 3
    """

    def __init__(self, task_id: str) -> None:
        if task_id not in TASK_SPEC:
            raise ValueError(f"task_id must be one of {list(TASK_SPEC.keys())}")
        self.task_id = task_id
        self._village_state: Dict[str, Any] = {}
        self._step_number: int = 0
        self._done: bool = False
        self._last_action: Action | None = None

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """
        Load village data and return the initial observation.
        Tries to load from pre-generated JSON first; falls back to on-the-fly generation.
        """
        village_path = os.path.join(DATA_DIR, VILLAGE_FILES[self.task_id])

        if os.path.exists(village_path):
            with open(village_path, "r", encoding="utf-8") as f:
                self._village_state = json.load(f)
        else:
            spec = TASK_SPEC[self.task_id]
            self._village_state = generate_village(
                num_families=spec["num_families"],
                include_anomalies=spec["include_anomalies"],
                seed=42,
            )

        self._step_number = 0
        self._done = False
        self._last_action = None

        return self._build_observation(
            message=f"Episode started. Village: {self._village_state['village_name']}. "
                    f"{STEP_MESSAGES[0]}"
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Apply action, compute reward, advance step counter.

        Returns:
            (observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if not self._village_state:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        self._last_action = action
        reward = calculate_reward(action, self._village_state, {"task_id": self.task_id})

        self._step_number += 1
        self._done = self._step_number >= 3

        if self._done:
            msg = (
                f"Episode complete after 3 steps. "
                f"Total reward: {reward.total_reward:.3f}"
            )
        else:
            msg = STEP_MESSAGES[self._step_number]

        obs = self._build_observation(message=msg)

        info = {
            "task_id": self.task_id,
            "step": self._step_number,
            "penalties": reward.penalties,
            "breakdown": reward.breakdown,
        }

        return obs, reward, self._done, info

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def state(self) -> Dict[str, Any]:
        """Return full current environment state as a plain dict."""
        return {
            "task_id": self.task_id,
            "step_number": self._step_number,
            "done": self._done,
            "village": self._village_state,
            "last_action": self._last_action.model_dump() if self._last_action else None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self, message: str) -> Observation:
        vs = self._village_state
        return Observation(
            village_name=vs.get("village_name", ""),
            district=vs.get("district", ""),
            state=vs.get("state", "Telangana"),
            month=vs.get("month", ""),
            task_id=self.task_id,
            available_resources=vs.get("available_resources", {}),
            families=vs.get("families", []),
            step_number=self._step_number,
            episode_done=self._done,
            message=message,
        )
