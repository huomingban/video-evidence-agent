"""Small persistent task runner used when a queue service is unavailable."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable

from .config import logger
from .storage import get_connection, get_media_task, update_media_task

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="media-task")


def submit_task(task_id: str, handler: Callable[[], dict[str, Any]]) -> None:
    _executor.submit(_run_task, task_id, handler)


def _run_task(task_id: str, handler: Callable[[], dict[str, Any]]) -> None:
    update_media_task(
        task_id,
        state="RUNNING",
        started_at=datetime.now(timezone.utc).isoformat(),
    )
    try:
        result = handler()
        update_media_task(
            task_id,
            state="COMPLETED",
            result_json=json.dumps(result, ensure_ascii=False),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    except Exception as error:
        logger.exception("media task failed task_id=%s", task_id)
        task = get_media_task(task_id)
        if task is not None and task.get("task_type") == "TRANSCRIPTION":
            with get_connection() as connection:
                connection.execute(
                    "UPDATE videos SET status = 'FAILED', updated_at = CURRENT_TIMESTAMP WHERE video_id = ?",
                    (task["video_id"],),
                )
        update_media_task(
            task_id,
            state="FAILED",
            error=str(error)[:2000],
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
