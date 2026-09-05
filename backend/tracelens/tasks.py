"""Small persistent task runner used when a queue service is unavailable."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable

from .config import logger
from .storage import get_connection, get_media_task, update_media_task

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="media-task")
_producer = None


def _publish(task_id: str) -> bool:
    global _producer
    if os.getenv("ROCKETMQ_ENABLED", "false").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    nameserver = os.getenv("ROCKETMQ_NAMESERVER", "").strip()
    topic = os.getenv("ROCKETMQ_TOPIC", "tracelens-video-tasks").strip()
    if not nameserver or not topic:
        return False
    try:
        from rocketmq.client import Message, Producer
        if _producer is None:
            _producer = Producer(os.getenv("ROCKETMQ_PRODUCER_GROUP", "tracelens-python-producer"))
            _producer.set_name_server_address(nameserver)
            _producer.start()
        message = Message(topic)
        message.set_keys(task_id)
        message.set_body(json.dumps({"taskId": task_id}, ensure_ascii=False).encode("utf-8"))
        _producer.send_sync(message)
        return True
    except Exception:
        logger.exception("rocketmq publish failed task_id=%s", task_id)
        return False


def submit_task(task_id: str, handler: Callable[[], dict[str, Any]]) -> None:
    if _publish(task_id):
        return
    _executor.submit(_run_task, task_id, handler)


def run_task(task_id: str, handler: Callable[[], dict[str, Any]]) -> None:
    """Run a persisted task in a queue worker."""
    _run_task(task_id, handler)


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
        if task is not None and task.get("task_type") in {"TRANSCRIPTION", "BILIBILI_IMPORT"}:
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
