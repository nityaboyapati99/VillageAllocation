"""
FastAPI server for Village Welfare Allocator.
Runs on port 7860 for HuggingFace Spaces compatibility.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from environment.env import VillageWelfareEnv
from environment.models import Action, Reward
from environment.reward import calculate_reward
from graders import GRADER_MAP
from tasks import ALL_TASKS, TASK_MAP

app = FastAPI(
    title="Village Welfare Allocator",
    description=(
        "OpenEnv-compliant environment simulating welfare resource allocation "
        "in rural Indian Gram Panchayats."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# In-memory environment state (single session — sufficient for hackathon)
# ---------------------------------------------------------------------------
_env: VillageWelfareEnv | None = None


def _get_env() -> VillageWelfareEnv:
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call POST /reset first.")
    return _env


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str  # "easy" | "medium" | "hard"


class GraderRequest(BaseModel):
    action: Action
    task_id: str


class BaselineResponse(BaseModel):
    easy: float
    medium: float
    hard: float
    average: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def health_check() -> Dict[str, str]:
    return {"status": "ok", "env": "village-welfare-allocator"}


@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "task_id": t["task_id"],
                "name": t["name"],
                "description": t["description"],
                "difficulty": t["difficulty"],
                "num_families": t["num_families"],
                "success_threshold": t["success_threshold"],
                "expected_agent_score": t["expected_agent_score"],
                "action_schema": t["action_schema"],
            }
            for t in ALL_TASKS
        ]
    }


@app.post("/reset")
async def reset(request: ResetRequest) -> Dict[str, Any]:
    global _env

    if request.task_id not in TASK_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id '{request.task_id}'. Must be one of: easy, medium, hard",
        )

    _env = VillageWelfareEnv(task_id=request.task_id)
    observation = _env.reset()
    return observation.model_dump()


@app.post("/step")
async def step(action: Action) -> Dict[str, Any]:
    env = _get_env()

    try:
        observation, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": observation.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
async def get_state() -> Dict[str, Any]:
    env = _get_env()
    return env.state()


@app.post("/grader")
async def grade(request: GraderRequest) -> Dict[str, Any]:
    if request.task_id not in GRADER_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id '{request.task_id}'. Must be one of: easy, medium, hard",
        )

    env = VillageWelfareEnv(task_id=request.task_id)
    env.reset()
    village_state = env.state()["village"]

    grader_fn = GRADER_MAP[request.task_id]
    task_config = TASK_MAP[request.task_id]
    score = grader_fn(request.action, village_state, task_config)

    reward = calculate_reward(request.action, village_state, task_config)
    result = reward.model_dump()
    result["grader_score"] = score
    return result


@app.post("/baseline")
async def run_baseline() -> Dict[str, Any]:
    """
    Triggers the greedy baseline (no LLM) against all 3 tasks.
    Returns scores without requiring an API key.
    """
    from inference import run_greedy_baseline

    loop = asyncio.get_event_loop()
    scores = await loop.run_in_executor(None, run_greedy_baseline)
    return scores
