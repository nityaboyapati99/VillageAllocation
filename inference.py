"""
inference.py — Village Welfare Allocator Baseline Inference Script

OpenEnv Hackathon Round 1 submission.
Authors: Shiva Chandra & Nitya Boyapati

Required environment variables:
    API_BASE_URL   — The API endpoint for the LLM (e.g. https://api.openai.com/v1)
    MODEL_NAME     — The model identifier (e.g. gpt-4o-mini)
    HF_TOKEN       — Your Hugging Face / API key

Optional:
    OPENAI_API_KEY — Falls back to HF_TOKEN if not set

Run:
    python inference.py
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
# Read required env vars (as specified by hackathon)
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME:   str = os.environ.get("MODEL_NAME",   "gpt-4o")
# HF_TOKEN is the primary key; fall back to OPENAI_API_KEY for local dev
HF_TOKEN:     str = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("OPENAI_API_KEY")
    or ""
)


# ---------------------------------------------------------------------------
# Greedy rule-based fallback (no API key required — used for smoke-testing)
# ---------------------------------------------------------------------------

def _greedy_action(observation_dict: Dict[str, Any]) -> Action:
    """
    Deterministic rule-based allocation.
    Distributes resources proportionally to need_score while respecting
    all eligibility rules. Used as fallback when no API key is available.
    """
    families  = observation_dict["families"]
    resources = observation_dict["available_resources"]

    mgnrega_budget = resources.get("mgnrega_days",    0)
    pm_slots       = resources.get("pm_awas_slots",   0)
    ration_slots   = resources.get("ration_upgrades", 0)

    sorted_fam = sorted(families, key=lambda f: f["need_score"], reverse=True)

    # MGNREGA — proportional to need among landless families
    candidates = [f for f in sorted_fam if f["land_acres"] == 0.0] or sorted_fam
    total_need = sum(f["need_score"] for f in candidates) or 1.0
    mgnrega_alloc: Dict[str, int] = {}
    remaining = mgnrega_budget
    for f in candidates:
        if remaining <= 0:
            break
        days = min(remaining, max(1, int((f["need_score"] / total_need) * mgnrega_budget)))
        if days > 0:
            mgnrega_alloc[f["id"]] = days
            remaining -= days

    # PM Awas — top-N houseless families
    pm_alloc = [f["id"] for f in sorted_fam if not f["has_house"]][:pm_slots]

    # Ration upgrades — top-N non-pink families
    ration_alloc = [
        f["id"] for f in sorted_fam if f["current_ration_card"] != "pink"
    ][:ration_slots]

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
        env.step(action)
        village_state = env.state()["village"]
        scores[task_id] = GRADER_MAP[task_id](action, village_state, TASK_MAP[task_id])
    scores["average"] = round(sum(scores[t] for t in TASK_IDS) / len(TASK_IDS), 4)
    return scores


# ---------------------------------------------------------------------------
# LLM prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(obs_dict: Dict[str, Any]) -> str:
    resources = obs_dict["available_resources"]
    families  = obs_dict["families"]

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

Village: {obs_dict['village_name']}, {obs_dict['district']} — {obs_dict['month']}
Task: {obs_dict['task_id']}

Available resources:
  MGNREGA work days : {resources.get('mgnrega_days', 0)}
  PM Awas slots     : {resources.get('pm_awas_slots', 0)}
  Ration upgrades   : {resources.get('ration_upgrades', 0)}

Families (need_score: higher = more needy):
  ID    | Name                           | need  | income | land | house | ration | sick | widow | elderly
{family_rows}

Rules:
1. Total MGNREGA days allocated must NOT exceed {resources.get('mgnrega_days', 0)}.
2. PM Awas: only allocate to families where house = N. Max slots = {resources.get('pm_awas_slots', 0)}.
3. Ration upgrade: only for families NOT already on 'pink'. Max = {resources.get('ration_upgrades', 0)}.
4. Prioritise families with highest need_score.
5. Watch for fraud: if a family has high need_score BUT also has high income (>6000), owns land (>1 acre),
   AND has a house — their application may be inflated. Do NOT allocate to them.
6. Distribute MGNREGA days across multiple families — do not give all days to one family.

Return ONLY valid JSON matching this exact schema (no explanation, no markdown):
{{
  "mgnrega_allocation": {{"F001": 15, "F003": 20}},
  "pm_awas_allocation": ["F005", "F012"],
  "ration_upgrade_allocation": ["F002", "F008"]
}}"""


def _parse_action(text: str) -> Action:
    """Parse JSON from LLM response into Action model."""
    text = text.strip()
    if "```" in text:
        text = "\n".join(
            line for line in text.split("\n")
            if not line.strip().startswith("```")
        )
    return Action(**json.loads(text))


# ---------------------------------------------------------------------------
# LLM-based inference using OpenAI client + hackathon env vars
# ---------------------------------------------------------------------------

def run_llm_inference() -> Dict[str, Any]:
    """
    Run LLM-based inference on all 3 tasks.
    Uses API_BASE_URL, MODEL_NAME, HF_TOKEN as required by the hackathon spec.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("openai package not installed. Run: pip install openai")
        sys.exit(1)

    # Use OpenAI client with hackathon-specified env vars
    client = OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL,
    )

    task_labels = {
        "easy":   "MGNREGA Allocation",
        "medium": "Multi-Scheme Allocation",
        "hard":   "Full Village + Fraud Detection",
    }

    scores: Dict[str, float] = {}
    print(f"\nRunning inference with model: {MODEL_NAME}")
    print(f"API base:                     {API_BASE_URL}\n")

    for i, task_id in enumerate(TASK_IDS, 1):
        env     = VillageWelfareEnv(task_id=task_id)
        obs     = env.reset()
        obs_dict = obs.model_dump()

        prompt = _build_prompt(obs_dict)

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1500,
            )
            llm_text = response.choices[0].message.content or ""
            action = _parse_action(llm_text)
        except Exception as exc:
            print(f"  [Task {task_id}] LLM error: {exc}. Falling back to greedy.")
            action = _greedy_action(obs_dict)

        env.step(action)
        village_state = env.state()["village"]
        score = GRADER_MAP[task_id](action, village_state, TASK_MAP[task_id])
        scores[task_id] = score

        label = task_labels[task_id]
        diff  = TASK_MAP[task_id]["difficulty"].capitalize()
        print(f"  Task {i} ({diff:<6} - {label:<30}): Score = {score:.2f}")

    avg = round(sum(scores[t] for t in TASK_IDS) / len(TASK_IDS), 4)
    scores["average"] = avg
    print(f"  {'─'*58}")
    print(f"  Average Score: {avg:.2f}\n")
    return scores


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if HF_TOKEN:
        run_llm_inference()
    else:
        print(
            "\n[WARNING] No HF_TOKEN or OPENAI_API_KEY found in environment.\n"
            "Running deterministic greedy baseline instead.\n"
            "Set HF_TOKEN (and optionally API_BASE_URL, MODEL_NAME) to run LLM inference.\n"
        )
        scores = run_greedy_baseline()
        labels = {
            "easy":   "MGNREGA Allocation",
            "medium": "Multi-Scheme Allocation",
            "hard":   "Full Village + Fraud",
        }
        for i, task_id in enumerate(TASK_IDS, 1):
            diff  = TASK_MAP[task_id]["difficulty"].capitalize()
            label = labels[task_id]
            print(f"  Task {i} ({diff:<6} - {label:<30}): Score = {scores[task_id]:.2f}")
        print(f"  Average Score: {scores['average']:.2f}\n")
