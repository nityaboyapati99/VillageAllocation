"""
Baseline agent for Village Welfare Allocator.

Two modes:
  1. GPT-4o-mini (requires OPENAI_API_KEY in env)
  2. Greedy rule-based fallback (no API key needed)

Run:  python baseline.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

from environment.env import VillageWelfareEnv
from environment.models import Action
from graders import GRADER_MAP
from tasks import TASK_MAP

TASK_IDS = ["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# Greedy rule-based baseline (always available, no API key required)
# ---------------------------------------------------------------------------

def _greedy_action(observation_dict: Dict[str, Any]) -> Action:
    """
    Simple deterministic rule-based baseline:
    - MGNREGA: distribute days proportionally to need_score among landless families,
               within budget.
    - PM Awas: pick top-N houseless families by need_score.
    - Ration:  pick top-N non-pink families by need_score.
    """
    families = observation_dict["families"]
    resources = observation_dict["available_resources"]
    mgnrega_budget = resources.get("mgnrega_days", 0)
    pm_slots = resources.get("pm_awas_slots", 0)
    ration_slots = resources.get("ration_upgrades", 0)

    # Sort by need_score desc
    sorted_fam = sorted(families, key=lambda f: f["need_score"], reverse=True)

    # --- MGNREGA ---
    mgnrega_candidates = [f for f in sorted_fam if f["land_acres"] == 0.0] or sorted_fam
    total_need = sum(f["need_score"] for f in mgnrega_candidates) or 1.0
    mgnrega_alloc: Dict[str, int] = {}
    remaining = mgnrega_budget
    for f in mgnrega_candidates:
        if remaining <= 0:
            break
        share = f["need_score"] / total_need
        days = min(remaining, max(1, int(share * mgnrega_budget)))
        if days > 0:
            mgnrega_alloc[f["id"]] = days
            remaining -= days

    # --- PM Awas ---
    pm_candidates = [f for f in sorted_fam if not f["has_house"]]
    pm_alloc = [f["id"] for f in pm_candidates[:pm_slots]]

    # --- Ration upgrades ---
    ration_candidates = [
        f for f in sorted_fam if f["current_ration_card"] != "pink"
    ]
    ration_alloc = [f["id"] for f in ration_candidates[:ration_slots]]

    return Action(
        mgnrega_allocation=mgnrega_alloc,
        pm_awas_allocation=pm_alloc,
        ration_upgrade_allocation=ration_alloc,
    )


def run_greedy_baseline() -> Dict[str, Any]:
    """Run greedy baseline on all 3 tasks. Returns score dict."""
    scores: Dict[str, float] = {}

    for task_id in TASK_IDS:
        env = VillageWelfareEnv(task_id=task_id)
        obs = env.reset()
        action = _greedy_action(obs.model_dump())
        _, reward, _, _ = env.step(action)
        grader_fn = GRADER_MAP[task_id]
        task_config = TASK_MAP[task_id]
        village_state = env.state()["village"]
        grader_score = grader_fn(action, village_state, task_config)
        scores[task_id] = grader_score

    scores["average"] = round(sum(scores[t] for t in TASK_IDS) / len(TASK_IDS), 4)
    return scores


# ---------------------------------------------------------------------------
# GPT-4o-mini baseline
# ---------------------------------------------------------------------------

def _build_prompt(obs_dict: Dict[str, Any]) -> str:
    village = obs_dict["village_name"]
    district = obs_dict["district"]
    month = obs_dict["month"]
    resources = obs_dict["available_resources"]
    families = obs_dict["families"]

    family_rows = "\n".join(
        f"  {f['id']} | {f['name']:<30} | need={f['need_score']:.2f} | "
        f"income={f['monthly_income']:>6} | land={f['land_acres']} | "
        f"house={'Y' if f['has_house'] else 'N'} | "
        f"ration={f['current_ration_card']:<6} | "
        f"sick={'Y' if f['has_sick_member'] else 'N'} | "
        f"widow={'Y' if f['is_widow_headed'] else 'N'} | "
        f"elderly={'Y' if f['is_elderly_headed'] else 'N'}"
        for f in families
    )

    return f"""You are a Gram Panchayat officer allocating welfare resources fairly.

Village: {village}, {district} — {month}
Task: {obs_dict['task_id']}

Available resources:
  MGNREGA work days : {resources.get('mgnrega_days', 0)}
  PM Awas slots     : {resources.get('pm_awas_slots', 0)}
  Ration upgrades   : {resources.get('ration_upgrades', 0)}

Families (need_score: higher = more needy):
  ID    | Name                           | need  | income | land | house | ration | sick | widow | elderly
{family_rows}

Rules:
1. TOTAL mgnrega days allocated must NOT exceed {resources.get('mgnrega_days', 0)}.
2. PM Awas: only for families with has_house = N. Slots = {resources.get('pm_awas_slots', 0)}.
3. Ration upgrade: only for families NOT already on 'pink'. Upgrades = {resources.get('ration_upgrades', 0)}.
4. Prioritise families with highest need_score.
5. For hard task: some families may have inflated need_scores but are actually well-off
   (high income + own land + has house). Do NOT allocate to them.

Return ONLY valid JSON matching this exact schema:
{{
  "mgnrega_allocation": {{"F001": 15, "F003": 20, ...}},
  "pm_awas_allocation": ["F005", "F012", ...],
  "ration_upgrade_allocation": ["F002", "F008", ...]
}}

No explanation. JSON only."""


def _parse_action(gpt_text: str) -> Action:
    """Extract JSON from GPT response and parse into Action."""
    text = gpt_text.strip()
    # Strip markdown code fences if present
    if "```" in text:
        lines = text.split("\n")
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        )
    data = json.loads(text)
    return Action(**data)


def run_gpt_baseline() -> Dict[str, Any]:
    """Run GPT-4o-mini baseline on all 3 tasks. Requires OPENAI_API_KEY."""
    try:
        from openai import OpenAI
    except ImportError:
        print("openai package not installed. Run: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set. Falling back to greedy baseline.")
        return run_greedy_baseline()

    client = OpenAI(api_key=api_key)
    scores: Dict[str, float] = {}
    task_names = {
        "easy":   "MGNREGA Allocation",
        "medium": "Multi-Scheme Allocation",
        "hard":   "Full Village + Fraud Detection",
    }

    print("\nRunning baseline agent (GPT-4o-mini)...")

    for task_id in TASK_IDS:
        env = VillageWelfareEnv(task_id=task_id)
        obs = env.reset()
        obs_dict = obs.model_dump()

        prompt = _build_prompt(obs_dict)

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1500,
            )
            gpt_text = response.choices[0].message.content or ""
            action = _parse_action(gpt_text)
        except Exception as e:
            print(f"  [{task_id}] GPT error: {e}. Falling back to greedy.")
            obs_dict["families"] = [f.model_dump() if hasattr(f, "model_dump") else f for f in obs.families]
            action = _greedy_action(obs_dict)

        _, reward, _, _ = env.step(action)
        grader_fn = GRADER_MAP[task_id]
        task_config = TASK_MAP[task_id]
        village_state = env.state()["village"]
        grader_score = grader_fn(action, village_state, task_config)
        scores[task_id] = grader_score

        difficulty = task_config["difficulty"].capitalize()
        name = task_names[task_id]
        pad = 35 - len(name)
        print(f"  Task {TASK_IDS.index(task_id)+1} ({difficulty:<6} - {name}){' '*pad}: Score = {grader_score:.2f}")

    avg = round(sum(scores[t] for t in TASK_IDS) / len(TASK_IDS), 4)
    scores["average"] = avg
    print(f"  {'─'*55}")
    print(f"  Average Score:{' '*40}: {avg:.2f}\n")
    return scores


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        run_gpt_baseline()
    else:
        print("\nRunning greedy baseline (no OPENAI_API_KEY set)...")
        scores = run_greedy_baseline()
        print(f"  Task 1 (Easy   - MGNREGA Allocation):    Score = {scores['easy']:.2f}")
        print(f"  Task 2 (Medium - Multi-Scheme):          Score = {scores['medium']:.2f}")
        print(f"  Task 3 (Hard   - Full Village + Fraud):  Score = {scores['hard']:.2f}")
        print(f"  Average Score: {scores['average']:.2f}\n")
