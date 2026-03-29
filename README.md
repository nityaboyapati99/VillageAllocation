---
title: Village Welfare Allocator
emoji: 🏡
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - rural-india
  - welfare
  - resource-allocation
  - fastapi
---

# Village Welfare Allocator

**OpenEnv Hackathon Round 1 Submission**
Authors: Shiva Chandra & Nitya Boyapati

---

## Overview

In rural India, Gram Panchayat officers manually decide which families receive government welfare benefits — MGNREGA work days, PM Awas Yojana housing, and ration card upgrades. This process is often slow, opaque, and vulnerable to corruption and unconscious bias.

**Village Welfare Allocator** is an OpenEnv-compliant AI environment that simulates this decision problem. An AI agent is given a village's family-level data and must allocate limited welfare resources as fairly and need-accurately as possible — across three difficulty levels, with the hardest level introducing fraudulent applications the agent must detect.

This environment addresses a real governance challenge affecting hundreds of millions of rural Indians.

---

## Environment Description

Each episode simulates one month of welfare allocation in a Telangana village. The agent receives a full observation of all families (income, land, health, housing, existing benefits) and the available resource budget, then submits a single allocation action covering all three schemes. The environment scores the allocation across five dimensions: need coverage, fairness, eligibility, anomaly detection, and budget adherence.

**Episode flow:**
1. `reset(task_id)` → returns village observation
2. `step(action)` → returns (observation, reward, done, info)
3. Episode ends after 3 steps (one per scheme type)

---

## Observation Space

```
village_name          : str    — Name of the village
district              : str    — Telangana district
state                 : str    — Always "Telangana"
month                 : str    — Current month
task_id               : str    — "easy" | "medium" | "hard"
available_resources   : dict   — {"mgnrega_days": int, "pm_awas_slots": int, "ration_upgrades": int}
families              : list   — List of Family objects (see below)
step_number           : int    — Current step (0-2)
episode_done          : bool   — True after step 3
message               : str    — Human-readable context

Family fields:
  id                       : str    — e.g. "F001"
  name                     : str    — Indian name
  land_acres               : float  — 0.0 = landless
  monthly_income           : int    — INR
  dependents               : int    — family size
  has_sick_member          : bool
  is_widow_headed          : bool
  is_elderly_headed        : bool   — head > 60 years old
  current_ration_card      : str    — "none" | "white" | "yellow" | "pink"
  has_house                : bool
  past_mgnrega_days_this_year : int
  debt_amount              : int    — INR
  need_score               : float  — 0.0–1.0 (pre-computed, higher = more needy)
  is_anomaly               : bool   — (hard task only) True = fraudulent application
```

---

## Action Space

```python
{
  "mgnrega_allocation":        Dict[family_id: str, days: int],
  "pm_awas_allocation":        List[family_id: str],
  "ration_upgrade_allocation": List[family_id: str]
}
```

Constraints:
- Total MGNREGA days must not exceed budget
- PM Awas: max slots allocated = `pm_awas_slots`
- Ration upgrades: max allocated = `ration_upgrades`

---

## Reward Function

| Component | Weight | Description |
|---|---|---|
| need_coverage_score | 0.40 | Did high-need families get priority? |
| fairness_score | 0.20 | Equitable MGNREGA distribution (Gini-based) |
| eligibility_score | 0.20 | Scheme eligibility rules respected |
| anomaly_detection_score | 0.10 | Fraudulent applications excluded |
| budget_adherence_score | 0.10 | Stayed within resource limits |

Final score = weighted sum, clipped to [0.0, 1.0]

**Penalties:**
- Allocating PM Awas to family that already has house: -0.1 per violation
- Allocating ration upgrade to family already on pink tier: -0.1 per violation
- Exceeding any resource budget: -0.5
- Allocating to anomaly (fraudulent) family: -0.2 per family

---

## Tasks

| Task | Families | Schemes | Anomalies | Success Threshold |
|---|---|---|---|---|
| easy | 20 | MGNREGA only | None | 0.70 |
| medium | 40 | MGNREGA + Ration | None | 0.65 |
| hard | 80 | All three | 5 hidden | 0.55 |

### Easy — MGNREGA Allocation
Allocate 200 MGNREGA work days across 20 families. Clear need signals, no tricks. Focus on landless, low-income families.

### Medium — Multi-Scheme Allocation
Allocate MGNREGA days and ration card upgrades across 40 families. Some families are eligible for one scheme but not both. Eligibility rules must be respected.

### Hard — Full Village with Fraud Detection
All three schemes across 80 families. Five families have inflated `need_score` but are actually well-off (anomaly families). The agent must detect and exclude them while correctly serving genuinely needy families.

---

## Baseline Scores

| Task | Agent | Score |
|---|---|---|
| Easy — MGNREGA Allocation | GPT-4o | 0.78 |
| Medium — Multi-Scheme | GPT-4o | 0.70 |
| Hard — Full Village + Fraud | GPT-4o | 0.75 |
| **Average** | | **0.75** |

---

## Setup & Installation

```bash
# Clone the repo
git clone https://github.com/your-username/village-welfare-allocator
cd village-welfare-allocator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy env file and add your OpenAI key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=your_key

# Pre-generate village data
python -c "from environment.village_generator import generate_and_save_all; generate_and_save_all()"
```

---

## Running Locally

```bash
uvicorn api.app:app --host 0.0.0.0 --port 7860
```

Then open: http://localhost:7860

---

## Running Baseline

```bash
# With OpenAI key (GPT-4o-mini)
python baseline.py

# Without key (greedy rule-based)
python baseline.py
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Docker

```bash
# Build
docker build -t village-welfare-allocator .

# Run (with OpenAI key)
docker run -p 7860:7860 -e OPENAI_API_KEY=your_key village-welfare-allocator

# Run (without key — greedy baseline only)
docker run -p 7860:7860 village-welfare-allocator
```

---

## API Reference

### `GET /`
Health check.
```json
{"status": "ok", "env": "village-welfare-allocator"}
```

### `GET /tasks`
List all 3 tasks with action schemas.
```json
{
  "tasks": [
    {
      "task_id": "easy",
      "name": "MGNREGA Allocation",
      "difficulty": "easy",
      "action_schema": { ... }
    }
  ]
}
```

### `POST /reset`
Start a new episode.
```json
// Request
{"task_id": "easy"}

// Response: Observation object
{"village_name": "Kondapuram", "families": [...], "available_resources": {...}, ...}
```

### `POST /step`
Submit an allocation action.
```json
// Request
{
  "mgnrega_allocation": {"F001": 20, "F003": 15},
  "pm_awas_allocation": ["F005"],
  "ration_upgrade_allocation": ["F002", "F007"]
}

// Response
{
  "observation": {...},
  "reward": {"total_reward": 0.78, "need_coverage_score": 0.85, ...},
  "done": false,
  "info": {"penalties": [], "breakdown": {...}}
}
```

### `GET /state`
Current environment state.

### `POST /grader`
Score an action without running a full episode.
```json
// Request
{"action": {...}, "task_id": "easy"}

// Response: Reward + grader_score
```

### `POST /baseline`
Run the greedy baseline on all 3 tasks. Returns scores.

---

## Real-World Impact

India has 250,000+ Gram Panchayats serving 800 million rural residents. Welfare allocation decisions — who gets MGNREGA work, who gets a house, who gets subsidised food — are made manually by local officers, often without data tools, under political pressure, and with limited accountability.

AI-assisted allocation tools trained on environments like this could:
- Reduce time to identify eligible beneficiaries from weeks to seconds
- Create auditable, need-based allocation records
- Surface anomalous applications for human review
- Help understaffed Panchayat offices manage larger beneficiary pools fairly

This environment is designed to produce agents that generalise across village sizes and demographic compositions — making it a realistic training ground for real-world welfare AI.
