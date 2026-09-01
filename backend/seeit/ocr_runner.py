"""Video media processing and timestamped ASR evidence extraction."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

from .config import direct_connection_if_configured

_WHISPER_MODEL = None

def resolve_ffmpeg_path() -> str | None:
    candidates = [
        shutil.which("ffmpeg"),
        r"C:\Users\21854\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\FFmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def extract_transcript_from_video(video_path: Path, file_name: str) -> list[tuple[float, float, str]]:
    ffmpeg_path = resolve_ffmpeg_path()
    if not ffmpeg_path:
        raise RuntimeError("FFmpeg is not installed or cannot be found")

    try:
        with NamedTemporaryFile(suffix=".wav", delete=False) as audio_temp:
            audio_tmp_path = Path(audio_temp.name)

        command = [
            ffmpeg_path,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            str(audio_tmp_path),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)

        global _WHISPER_MODEL
        try:
            from faster_whisper import WhisperModel
        except Exception as error:
            raise RuntimeError("faster-whisper is not installed") from error

        if _WHISPER_MODEL is None:
            with direct_connection_if_configured():
                _WHISPER_MODEL = WhisperModel(
                    os.getenv("WHISPER_MODEL", "tiny"),
                    device=os.getenv("WHISPER_DEVICE", "cpu"),
                    compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
                )
        model = _WHISPER_MODEL
        segments, _ = model.transcribe(str(audio_tmp_path), language="zh", beam_size=1, vad_filter=True)
        chunks: list[tuple[float, float, str]] = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                chunks.append((float(segment.start), float(segment.end), text))

        if chunks:
            return merge_transcript_chunks(
                chunks,
                max_duration=float(os.getenv("TRANSCRIPT_CHUNK_SECONDS", "30")),
                max_chars=int(os.getenv("TRANSCRIPT_CHUNK_MAX_CHARS", "240")),
                max_gap=float(os.getenv("TRANSCRIPT_MAX_GAP_SECONDS", "2")),
            )
    except RuntimeError:
        raise
    except Exception as error:
        logger.exception("Video transcription failed")
        raise RuntimeError(f"Video transcription failed: {error}") from error
    finally:
        try:
            if "audio_tmp_path" in locals():
                audio_tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    return []


def merge_transcript_chunks(
    segments: list[tuple[float, float, str]],
    max_duration: float = 30.0,
    max_chars: int = 240,
    max_gap: float = 2.0,
) -> list[tuple[float, float, str]]:
    """Group short Whisper segments into readable, retrieval-friendly evidence blocks."""
    merged: list[tuple[float, float, str]] = []
    current_start: float | None = None
    current_end = 0.0
    current_text = ""

    for start, end, text in segments:
        normalized = " ".join(text.split())
        if not normalized or end <= start:
            continue

        candidate_text = f"{current_text}{normalized}" if current_text else normalized
        should_flush = (
            current_start is not None
            and (start - current_end > max_gap
                 or end - current_start > max_duration
                 or len(candidate_text) > max_chars)
        )
        if should_flush:
            merged.append((current_start, current_end, current_text))
            current_start = None
            current_text = ""

        if current_start is None:
            current_start = start
        current_end = end
        current_text += normalized

    if current_start is not None and current_text:
        merged.append((current_start, current_end, current_text))
    return merged
