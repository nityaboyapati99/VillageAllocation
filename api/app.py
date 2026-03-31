"""
FastAPI server for Village Welfare Allocator.
Runs on port 7860 for HuggingFace Spaces compatibility.
OpenEnv-compliant: /health, /metadata, /schema, /mcp, /reset, /step, /state
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
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
    task_id: str = "easy"  # "easy" | "medium" | "hard"


class GraderRequest(BaseModel):
    action: Action
    task_id: str


# ---------------------------------------------------------------------------
# OpenEnv Required Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root() -> Dict[str, str]:
    return {"status": "ok", "env": "village-welfare-allocator"}


@app.get("/health")
async def health() -> Dict[str, str]:
    """OpenEnv required: health check returning status=healthy."""
    return {"status": "healthy"}


@app.get("/metadata")
async def metadata() -> Dict[str, Any]:
    """OpenEnv required: environment metadata."""
    return {
        "name": "village-welfare-allocator",
        "description": (
            "An OpenEnv environment simulating welfare resource allocation in rural Indian "
            "Gram Panchayats. An AI agent must fairly distribute MGNREGA work days, PM Awas "
            "housing slots, and ration card upgrades across village families based on need, "
            "eligibility, and equity — while detecting fraudulent applications in hard mode."
        ),
        "version": "1.0.0",
        "author": "Shiva Chandra & Nitya Boyapati",
        "tags": ["rural-india", "welfare", "resource-allocation", "fairness", "openenv"],
        "tasks": [
            {"id": t["task_id"], "difficulty": t["difficulty"], "name": t["name"]}
            for t in ALL_TASKS
        ],
    }


@app.get("/schema")
async def schema() -> Dict[str, Any]:
    """OpenEnv required: action, observation, and state schemas."""
    return {
        "action": {
            "type": "object",
            "description": "Welfare resource allocation decisions",
            "properties": {
                "mgnrega_allocation": {
                    "type": "object",
                    "description": "Dict mapping family_id to MGNREGA work days allocated",
                    "additionalProperties": {"type": "integer"},
                },
                "pm_awas_allocation": {
                    "type": "array",
                    "description": "List of family_ids receiving PM Awas housing slot",
                    "items": {"type": "string"},
                },
                "ration_upgrade_allocation": {
                    "type": "array",
                    "description": "List of family_ids receiving ration card upgrade",
                    "items": {"type": "string"},
                },
            },
        },
        "observation": {
            "type": "object",
            "description": "Village state visible to the agent",
            "properties": {
                "village_name": {"type": "string"},
                "district": {"type": "string"},
                "state": {"type": "string"},
                "month": {"type": "string"},
                "task_id": {"type": "string", "enum": ["easy", "medium", "hard"]},
                "available_resources": {
                    "type": "object",
                    "properties": {
                        "mgnrega_days": {"type": "integer"},
                        "pm_awas_slots": {"type": "integer"},
                        "ration_upgrades": {"type": "integer"},
                    },
                },
                "families": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "land_acres": {"type": "number"},
                            "monthly_income": {"type": "integer"},
                            "dependents": {"type": "integer"},
                            "has_sick_member": {"type": "boolean"},
                            "is_widow_headed": {"type": "boolean"},
                            "is_elderly_headed": {"type": "boolean"},
                            "current_ration_card": {"type": "string"},
                            "has_house": {"type": "boolean"},
                            "past_mgnrega_days_this_year": {"type": "integer"},
                            "debt_amount": {"type": "integer"},
                            "need_score": {"type": "number"},
                            "is_anomaly": {"type": "boolean"},
                        },
                    },
                },
                "step_number": {"type": "integer"},
                "episode_done": {"type": "boolean"},
                "message": {"type": "string"},
            },
        },
        "state": {
            "type": "object",
            "description": "Full internal environment state",
            "properties": {
                "task_id": {"type": "string"},
                "step_number": {"type": "integer"},
                "done": {"type": "boolean"},
                "village": {"type": "object"},
                "last_action": {"type": "object", "nullable": True},
            },
        },
    }


@app.post("/mcp")
async def mcp(request: Request) -> Dict[str, Any]:
    """OpenEnv required: MCP (Model Context Protocol) JSON-RPC 2.0 endpoint."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    method  = body.get("method", "")
    req_id  = body.get("id", 1)
    jsonrpc = body.get("jsonrpc", "2.0")

    # Handle standard MCP methods
    if method == "initialize":
        result = {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": "village-welfare-allocator",
                "version": "1.0.0",
            },
        }
    elif method == "tools/list":
        result = {
            "tools": [
                {
                    "name": "reset",
                    "description": "Reset the environment for a given task",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"task_id": {"type": "string"}},
                        "required": ["task_id"],
                    },
                },
                {
                    "name": "step",
                    "description": "Submit a welfare allocation action",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "mgnrega_allocation": {"type": "object"},
                            "pm_awas_allocation": {"type": "array"},
                            "ration_upgrade_allocation": {"type": "array"},
                        },
                    },
                },
                {
                    "name": "state",
                    "description": "Get current environment state",
                    "inputSchema": {"type": "object", "properties": {}},
                },
            ]
        }
    else:
        result = {
            "name": "village-welfare-allocator",
            "version": "1.0.0",
            "description": "OpenEnv welfare allocation environment",
        }

    return {"jsonrpc": "2.0", "id": req_id, "result": result}


# ---------------------------------------------------------------------------
# Core OpenEnv Endpoints
# ---------------------------------------------------------------------------

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
    """Triggers greedy baseline against all 3 tasks. No API key needed."""
    from inference import run_greedy_baseline

    loop = asyncio.get_event_loop()
    scores = await loop.run_in_executor(None, run_greedy_baseline)
    return scores
