from .task_easy import TASK_CONFIG as EASY_CONFIG
from .task_medium import TASK_CONFIG as MEDIUM_CONFIG
from .task_hard import TASK_CONFIG as HARD_CONFIG

ALL_TASKS = [EASY_CONFIG, MEDIUM_CONFIG, HARD_CONFIG]

TASK_MAP = {
    "easy": EASY_CONFIG,
    "medium": MEDIUM_CONFIG,
    "hard": HARD_CONFIG,
}

__all__ = ["ALL_TASKS", "TASK_MAP", "EASY_CONFIG", "MEDIUM_CONFIG", "HARD_CONFIG"]
