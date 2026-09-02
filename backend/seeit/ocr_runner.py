"""Video media processing and timestamped ASR evidence extraction."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import gc
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path
from tempfile import NamedTemporaryFile

from .config import PROJECT_ROOT, direct_connection_if_configured, logger

_WHISPER_MODEL = None
_OPENCC = None


def normalize_transcript_text(text: str) -> str:
    """Normalize ASR text for display while preserving the original meaning."""
    normalized = " ".join(str(text).split()).strip()
    if not normalized:
        return ""
    global _OPENCC
    if _OPENCC is None:
        try:
            from opencc import OpenCC
            _OPENCC = OpenCC("t2s")
        except Exception:
            _OPENCC = False
    if _OPENCC:
        return _OPENCC.convert(normalized)
    return normalized


def _paddle_ocr_content(result: object, confidence_threshold: float, min_length: int) -> str:
    payload = getattr(result, "json", result)
    if isinstance(payload, dict) and isinstance(payload.get("res"), dict):
        payload = payload["res"]
    if not isinstance(payload, dict):
        return ""
    texts = list(payload.get("rec_texts") or [])
    scores = list(payload.get("rec_scores") or [])
    accepted: list[str] = []
    for index, raw_text in enumerate(texts):
        text = normalize_transcript_text(str(raw_text))
        try:
            score = float(scores[index]) if index < len(scores) else 1.0
        except (TypeError, ValueError):
            score = 0.0
        if score >= confidence_threshold and len(text) >= min_length:
            accepted.append(text)
    return " ".join(accepted)[:2000]

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


def _cached_whisper_snapshot(model_name: str) -> str | None:
    """Find a usable faster-whisper snapshot without contacting Hugging Face."""
    candidate = Path(model_name).expanduser()
    if candidate.is_dir() and all((candidate / name).is_file() for name in (
        "config.json", "model.bin", "tokenizer.json", "vocabulary.txt"
    )):
        return str(candidate)
    if "/" in model_name or "\\" in model_name:
        repo_name = model_name.replace("/", "--").replace("\\", "--")
    else:
        repo_name = f"models--Systran--faster-whisper-{model_name}"
    cache_root = Path(os.getenv("HF_HUB_CACHE", str(Path.home() / ".cache" / "huggingface" / "hub")))
    repo_root = cache_root / repo_name
    snapshots = repo_root / "snapshots"
    if not snapshots.is_dir():
        return None
    for snapshot in sorted(snapshots.iterdir(), reverse=True):
        if snapshot.is_dir() and all((snapshot / name).is_file() for name in (
            "config.json", "model.bin", "tokenizer.json", "vocabulary.txt"
        )):
            return str(snapshot)
    return None


def _whisper_model_kwargs() -> dict[str, object]:
    return {
        "device": os.getenv("WHISPER_DEVICE", "cpu"),
        "compute_type": os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        "cpu_threads": max(1, int(os.getenv("WHISPER_CPU_THREADS", "4"))),
    }


def _load_whisper_model() -> object:
    from faster_whisper import WhisperModel

    configured = os.getenv("WHISPER_MODEL", "base").strip() or "base"
    cached = _cached_whisper_snapshot(configured)
    if cached:
        return WhisperModel(cached, local_files_only=True, **_whisper_model_kwargs())

    try:
        return WhisperModel(
            configured,
            download_root=_resolve_model_root(),
            **_whisper_model_kwargs(),
        )
    except Exception as primary_error:
        fallback = os.getenv("WHISPER_FALLBACK_MODEL", "tiny").strip()
        fallback_path = _cached_whisper_snapshot(fallback)
        if fallback_path and fallback != configured:
            logger.warning(
                "Whisper model %s unavailable; using cached fallback %s",
                configured,
                fallback,
            )
            return WhisperModel(
                fallback_path,
                local_files_only=True,
                **_whisper_model_kwargs(),
            )
        raise RuntimeError(
            f"Whisper 模型 {configured} 尚未下载，且无法连接 Hugging Face。"
            "请检查网络/代理，或先手动下载模型后重试。"
            f" 原始错误：{str(primary_error)[:300]}"
        ) from primary_error


def release_whisper_model() -> None:
    """Release ASR memory before starting PaddleOCR in the same worker."""
    global _WHISPER_MODEL
    _WHISPER_MODEL = None
    gc.collect()


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
        subprocess.run(
            command, check=True, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )

        global _WHISPER_MODEL
        try:
            from faster_whisper import WhisperModel
        except Exception as error:
            raise RuntimeError("faster-whisper is not installed") from error

        if _WHISPER_MODEL is None:
            with direct_connection_if_configured():
                _WHISPER_MODEL = _load_whisper_model()
        model = _WHISPER_MODEL
        segments, _ = model.transcribe(
            str(audio_tmp_path),
            language=os.getenv("WHISPER_LANGUAGE", "zh").strip() or None,
            beam_size=max(1, int(os.getenv("WHISPER_BEAM_SIZE", "5"))),
            vad_filter=os.getenv("WHISPER_VAD_FILTER", "true").lower() in {"1", "true", "yes"},
            condition_on_previous_text=True,
            initial_prompt=os.getenv("WHISPER_INITIAL_PROMPT", "").strip() or None,
            hotwords=os.getenv("WHISPER_HOTWORDS", "").strip() or None,
        )
        chunks: list[tuple[float, float, str]] = []
        for segment in segments:
            text = normalize_transcript_text(segment.text)
            if text:
                chunks.append((float(segment.start), float(segment.end), text))

        if chunks:
            return merge_transcript_chunks(
                chunks,
                max_duration=float(os.getenv("TRANSCRIPT_CHUNK_SECONDS", "0")),
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


def _resolve_model_root() -> str:
    configured = os.getenv("WHISPER_MODEL_ROOT", "data/models/whisper").strip()
    path = Path(configured)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def extract_ocr_evidence(video_path: Path, file_name: str = "") -> list[tuple[float, float, str]]:
    """Extract deduplicated on-screen text from periodic video keyframes."""
    if os.getenv("OCR_ENABLED", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        return []
    ffmpeg_path = resolve_ffmpeg_path()
    if not ffmpeg_path:
        raise RuntimeError("OCR requires FFmpeg, but FFmpeg was not found")

    interval = max(1, int(os.getenv("OCR_INTERVAL_SECONDS", "15")))
    max_frames = max(1, int(os.getenv("PADDLEOCR_MAX_FRAMES", "40")))
    max_width = max(320, int(os.getenv("PADDLEOCR_FRAME_MAX_WIDTH", "960")))
    threshold = min(1.0, max(0.0, float(os.getenv("PADDLEOCR_CONFIDENCE_THRESHOLD", "0.65"))))
    min_length = max(1, int(os.getenv("PADDLEOCR_MIN_TEXT_LENGTH", "1")))
    dedup_threshold = min(1.0, max(0.0, float(os.getenv("PADDLEOCR_DEDUP_THRESHOLD", "0.88"))))
    temp_root = PROJECT_ROOT / ".tmp"
    temp_root.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(prefix="ocr-", dir=temp_root) as directory:
            frame_dir = Path(directory)
            frame_pattern = str(frame_dir / "frame-%06d.png")
            subprocess.run(
                [
                    ffmpeg_path, "-y", "-i", str(video_path),
                    "-vf", (
                        f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{interval})',"
                        f"scale='min({max_width},iw)':-2"
                    ),
                    "-fps_mode", "vfr", "-frames:v", str(max_frames), frame_pattern,
                ],
                check=True, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=900,
            )
            frames = sorted(frame_dir.glob("frame-*.png"))
            if not frames:
                logger.warning("OCR extracted no video frames for %s", file_name or video_path.name)
                return []
            payload = _run_ocr_subprocess(frame_dir)
            if payload.get("fatalError"):
                raise RuntimeError(f"OCR runner failed: {payload['fatalError']}")
            frame_errors = list(payload.get("errors") or [])
            if frame_errors and not any(
                str(item.get("content", "")).strip()
                for item in payload.get("results", [])
            ):
                first_error = frame_errors[0].get("error", "unknown frame error")
                raise RuntimeError(
                    f"OCR failed on all {len(frame_errors)} frames: {first_error}"
                )
            results: list[tuple[float, float, str]] = []
            previous = ""
            for item in payload.get("results", []):
                index = max(0, int(item.get("index", 0)))
                content = normalize_transcript_text(str(item.get("content", ""))).strip()
                if not content or (previous and SequenceMatcher(None, previous, content).ratio() >= dedup_threshold):
                    continue
                start = float(index * interval)
                results.append((start, start + interval, content))
                previous = content
            logger.info("OCR completed file=%s frames=%s evidence=%s", file_name, payload.get("frameCount", len(frames)), len(results))
            return results
    except RuntimeError:
        raise
    except Exception as error:
        logger.exception("OCR failed for %s", file_name or video_path.name)
        raise RuntimeError(f"OCR failed: {error}") from error


def _run_ocr_subprocess(directory: Path) -> dict[str, object]:
    """Run Paddle in a clean process, matching the reference project boundary."""
    output_path = directory / "paddle-results.json"
    project_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    python_executable = str(project_python) if project_python.is_file() else sys.executable
    command = [
        python_executable, "-m", "seeit.ocr_runner",
        "--input-dir", str(directory), "--output", str(output_path),
    ]
    child_env = os.environ.copy()
    model_root = Path(os.getenv("PADDLEOCR_MODEL_ROOT", "data/models/paddlex"))
    if not model_root.is_absolute():
        model_root = PROJECT_ROOT / model_root
    paddle_home = Path(os.getenv("PADDLE_HOME", "data/models/paddle"))
    if not paddle_home.is_absolute():
        paddle_home = PROJECT_ROOT / paddle_home
    model_root.mkdir(parents=True, exist_ok=True)
    paddle_home.mkdir(parents=True, exist_ok=True)
    child_env.update({
        "PADDLE_PDX_CACHE_HOME": str(model_root),
        "PADDLE_PDX_MODEL_SOURCE": os.getenv("PADDLEOCR_MODEL_SOURCE", "bos"),
        "PADDLE_HOME": str(paddle_home),
        "PADDLEX_HOME": str(model_root),
        # Paddle's legacy hub still resolves ~/.cache/paddle through the
        # Windows user profile even when PaddleX cache variables are set.
        "USERPROFILE": str(paddle_home),
        "HOME": str(paddle_home),
    })
    completed = subprocess.run(
        command, cwd=str(PROJECT_ROOT / "backend"), env=child_env,
        check=False, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=max(60, int(os.getenv("PADDLEOCR_PROCESS_TIMEOUT_SECONDS", "600"))),
    )
    if not output_path.exists():
        raise RuntimeError(
            f"OCR runner exited with code {completed.returncode}: {completed.stderr[-1000:]}"
        )
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"OCR runner returned invalid JSON: {error}") from error
    if completed.returncode != 0 or payload.get("fatalError"):
        logger.error("OCR runner failed returncode=%s fatal=%s stderr=%s", completed.returncode, payload.get("fatalError"), completed.stderr[-2000:])
    return payload


def _ocr_runner_main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    try:
        model_root = Path(os.getenv("PADDLEOCR_MODEL_ROOT", "data/models/paddlex"))
        if not model_root.is_absolute():
            model_root = PROJECT_ROOT / model_root
        model_root.mkdir(parents=True, exist_ok=True)
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(model_root)
        os.environ["PADDLE_PDX_MODEL_SOURCE"] = os.getenv("PADDLEOCR_MODEL_SOURCE", "bos")
        os.environ["PADDLEX_HOME"] = str(model_root)
        paddle_home = Path(os.getenv("PADDLE_HOME", "data/models/paddle"))
        if not paddle_home.is_absolute():
            paddle_home = PROJECT_ROOT / paddle_home
        paddle_home.mkdir(parents=True, exist_ok=True)
        os.environ["USERPROFILE"] = str(paddle_home)
        os.environ["HOME"] = str(paddle_home)
        from paddleocr import PaddleOCR
        confidence = min(1.0, max(0.0, float(os.getenv("PADDLEOCR_CONFIDENCE_THRESHOLD", "0.65"))))
        min_length = max(1, int(os.getenv("PADDLEOCR_MIN_TEXT_LENGTH", "1")))
        frames = sorted(Path(args.input_dir).glob("frame-*.png"))
        model = PaddleOCR(
            text_detection_model_name=os.getenv("PADDLEOCR_DETECTION_MODEL", "PP-OCRv5_mobile_det"),
            text_recognition_model_name=os.getenv("PADDLEOCR_RECOGNITION_MODEL", "PP-OCRv5_mobile_rec"),
            use_doc_orientation_classify=False, use_doc_unwarping=False,
            use_textline_orientation=False, text_rec_score_thresh=confidence,
            device=os.getenv("PADDLEOCR_DEVICE", "cpu"),
            cpu_threads=max(1, int(os.getenv("PADDLEOCR_CPU_THREADS", "4"))),
            enable_mkldnn=os.getenv(
                "PADDLEOCR_ENABLE_MKLDNN",
                "false" if os.name == "nt" else "true",
            ).lower() in {"1", "true", "yes"},
        )
        results = []
        errors = []
        for index, frame in enumerate(frames):
            try:
                predictions = model.predict(str(frame), use_doc_orientation_classify=False,
                                            use_doc_unwarping=False, use_textline_orientation=False,
                                            text_rec_score_thresh=confidence)
                content = " ".join(filter(None, (_paddle_ocr_content(item, confidence, min_length) for item in predictions)))
                results.append({"index": index, "frame": frame.name, "content": content[:2000]})
            except Exception as error:
                errors.append({"frame": frame.name, "error": f"{type(error).__name__}: {error}"})
        payload = {"frameCount": len(frames), "results": results, "errors": errors}
        output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return 0
    except Exception as error:
        output.write_text(json.dumps({"frameCount": 0, "results": [], "errors": [], "fatalError": f"{type(error).__name__}: {error}"}, ensure_ascii=False), encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(_ocr_runner_main())


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
        ends_sentence = bool(re.search(r"[。！？!?；;]$", normalized))
        duration_reached = max_duration > 0 and end - current_start > max_duration if current_start is not None else False
        should_flush = current_start is not None and (
            start - current_end > max_gap
            or duration_reached
            or len(candidate_text) > max_chars
            or (max_duration <= 0 and ends_sentence and end - current_start >= 8)
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
