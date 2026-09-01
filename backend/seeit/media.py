"""Media resource validation helpers.

Audio extraction and ASR live in ``ocr_runner.py`` to mirror the reference
project's media-processing boundary.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import HTTPException


def validate_video_id(video_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", video_id):
        raise HTTPException(
            status_code=422,
            detail="video_id may contain only letters, numbers, '.', '_' and '-'",
        )
    return video_id


def validate_video_filename(filename: str) -> str:
    safe_name = Path(filename.replace("\\", "/")).name
    allowed_extensions = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg"}
    if not safe_name or Path(safe_name).suffix.lower() not in allowed_extensions:
        raise HTTPException(status_code=415, detail="unsupported video file type")
    return safe_name
