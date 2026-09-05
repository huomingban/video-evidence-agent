"""RocketMQ worker that reconstructs persisted TraceLens tasks by task type."""
from __future__ import annotations

import json
import os
import signal
import threading

from . import api
from .storage import get_connection, get_media_task
from .tasks import run_task


def _handler(task_id: str):
    task = get_media_task(task_id)
    if task is None:
        raise RuntimeError(f"task not found: {task_id}")
    task_type = str(task.get("task_type", "")).upper()
    if task_type == "ANALYSIS":
        return lambda: api._run_analysis_task(
            task_id, str(task.get("question") or ""), task.get("video_id") or None, str(task.get("session_id") or "")
        )
    if task_type == "BILIBILI_IMPORT":
        return lambda: api._run_bilibili_import_task(task_id, str(task["video_id"]))
    if task_type == "TRANSCRIPTION":
        with get_connection() as connection:
            video = connection.execute(
                "SELECT stored_path, filename FROM videos WHERE video_id = ?", (task["video_id"],)
            ).fetchone()
        if video is None:
            raise RuntimeError("transcription video not found")
        from pathlib import Path
        return lambda: api._run_transcription_task(
            task_id, str(task["video_id"]), Path(video["stored_path"]), str(video["filename"])
        )
    raise RuntimeError(f"unsupported task type: {task_type}")


def main() -> None:
    from rocketmq.client import ConsumeStatus, PushConsumer

    nameserver = os.getenv("ROCKETMQ_NAMESERVER", "127.0.0.1:9876")
    topic = os.getenv("ROCKETMQ_TOPIC", "tracelens-video-tasks")
    group = os.getenv("ROCKETMQ_CONSUMER_GROUP", "tracelens-python-consumer")
    consumer = PushConsumer(group)
    consumer.set_name_server_address(nameserver)
    stopped = threading.Event()

    def handle(message):
        try:
            payload = json.loads(message.body.decode("utf-8"))
            task_id = str(payload["taskId"])
            run_task(task_id, _handler(task_id))
            return ConsumeStatus.CONSUME_SUCCESS
        except Exception:
            api.logger.exception("rocketmq task failed")
            return ConsumeStatus.RECONSUME_LATER

    consumer.subscribe(topic, handle)
    consumer.start()
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    api.logger.info("rocketmq worker started nameserver=%s topic=%s", nameserver, topic)
    stopped.wait()
    consumer.shutdown()


if __name__ == "__main__":
    main()
