from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
try:
    import yt_dlp
except ImportError:  # pragma: no cover - optional until Bilibili import is used
    yt_dlp = None

BVID_PATTERN = re.compile(r"BV[0-9A-Za-z]{10}", re.IGNORECASE)
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".flv", ".mov", ".m4v"}
BILIBILI_API_BASE = "https://api.bilibili.com"
BILIBILI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131.0 Safari/537.36",
    "Referer": "https://www.bilibili.com/",
    "Origin": "https://www.bilibili.com",
}


class _QuietYtdlpLogger:
    def debug(self, _: str) -> None:
        return

    def warning(self, _: str) -> None:
        return

    def error(self, _: str) -> None:
        return


class BilibiliImportError(RuntimeError):
    pass


@dataclass(frozen=True)
class DownloadedVideo:
    bvid: str
    title: str
    uploader: str
    duration_seconds: int
    cover_url: str
    path: Path


def normalize_bvid(value: str) -> str:
    match = BVID_PATTERN.search(str(value).strip())
    if not match:
        raise ValueError("请输入正确的 BV 号")
    return "BV" + match.group(0)[2:]


def bilibili_video_url(bvid: str) -> str:
    return f"https://www.bilibili.com/video/{normalize_bvid(bvid)}"


def _options() -> dict[str, Any]:
    return {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "socket_timeout": 20,
        "retries": 3,
        "fragment_retries": 3,
        "extractor_retries": 2,
        "http_headers": BILIBILI_HEADERS,
        "logger": _QuietYtdlpLogger(),
    }


def _metadata(info: dict[str, Any], bvid: str) -> dict[str, Any]:
    cover = str(info.get("thumbnail") or info.get("pic") or "")[:1024]
    if cover.startswith("http://"):
        cover = "https://" + cover.removeprefix("http://")
    return {
        "bvid": bvid,
        "title": str(info.get("title") or bvid)[:255],
        "uploader": str(info.get("uploader") or info.get("channel") or "")[:100],
        "durationSeconds": max(0, int(info.get("duration") or 0)),
        "coverUrl": cover,
        "webpageUrl": bilibili_video_url(bvid),
    }


def _api_payload(path: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        response = httpx.get(
            f"{BILIBILI_API_BASE}{path}", params=params, headers=BILIBILI_HEADERS,
            follow_redirects=True, timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError, TypeError) as exc:
        raise BilibiliImportError("B 站公开接口暂时不可用") from exc
    if not isinstance(payload, dict) or payload.get("code") != 0:
        message = str(payload.get("message") or "未知错误") if isinstance(payload, dict) else "响应格式异常"
        raise BilibiliImportError(f"B 站公开接口返回异常：{message[:120]}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise BilibiliImportError("B 站公开接口未返回视频信息")
    return data


def _download_stream(url: str, destination: Path) -> None:
    try:
        with httpx.stream("GET", url, headers=BILIBILI_HEADERS, follow_redirects=True,
                          timeout=httpx.Timeout(120, connect=20)) as response:
            response.raise_for_status()
            with destination.open("wb") as output:
                for chunk in response.iter_bytes(1024 * 1024):
                    output.write(chunk)
    except (httpx.HTTPError, OSError) as exc:
        raise BilibiliImportError("B 站视频流下载失败") from exc


def _stream_url(stream: dict[str, Any]) -> str:
    return str(stream.get("baseUrl") or stream.get("base_url") or "")


def _download_via_public_api(bvid: str, directory: Path) -> DownloadedVideo:
    view = _api_payload("/x/web-interface/view", {"bvid": bvid})
    pages = view.get("pages") if isinstance(view.get("pages"), list) else []
    first_page = pages[0] if pages and isinstance(pages[0], dict) else {}
    aid, cid = int(view.get("aid") or 0), int(first_page.get("cid") or view.get("cid") or 0)
    if not aid or not cid:
        raise BilibiliImportError("B 站公开接口未返回视频分集信息")
    play = _api_payload("/x/player/playurl", {
        "avid": aid, "cid": cid, "fnval": 16, "fnver": 0, "fourk": 0, "qn": 80,
    })
    dash = play.get("dash") if isinstance(play.get("dash"), dict) else {}
    videos = [item for item in dash.get("video", []) if isinstance(item, dict) and _stream_url(item)]
    audios = [item for item in dash.get("audio", []) if isinstance(item, dict) and _stream_url(item)]
    if not videos or not audios:
        raise BilibiliImportError("B 站公开接口未返回可用的音视频流")
    video = max(videos, key=lambda item: int(item.get("bandwidth") or 0))
    audio = max(audios, key=lambda item: int(item.get("bandwidth") or 0))
    video_path, audio_path = directory / f"{bvid}.video.m4s", directory / f"{bvid}.audio.m4s"
    output_path = directory / f"{bvid}.mp4"
    try:
        _download_stream(_stream_url(video), video_path)
        _download_stream(_stream_url(audio), audio_path)
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_path),
            "-i", str(audio_path), "-c", "copy", "-movflags", "+faststart", str(output_path),
        ], check=True, capture_output=True, text=True, timeout=180)
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            raise BilibiliImportError("FFmpeg 未生成可用的 B 站视频文件")
    except (subprocess.SubprocessError, OSError) as exc:
        raise BilibiliImportError("FFmpeg 合并 B 站音视频流失败") from exc
    finally:
        video_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)
    owner = view.get("owner") if isinstance(view.get("owner"), dict) else {}
    metadata = _metadata({**view, "uploader": owner.get("name")}, bvid)
    return DownloadedVideo(bvid=bvid, title=metadata["title"], uploader=metadata["uploader"],
                           duration_seconds=metadata["durationSeconds"], cover_url=metadata["coverUrl"],
                           path=output_path)


def fetch_bilibili_metadata(value: str) -> dict[str, Any]:
    bvid = normalize_bvid(value)
    try:
        info = _api_payload("/x/web-interface/view", {"bvid": bvid})
        owner = info.get("owner") if isinstance(info.get("owner"), dict) else {}
        return _metadata({**info, "uploader": owner.get("name")}, bvid)
    except BilibiliImportError as api_error:
        if yt_dlp is None:
            raise BilibiliImportError("Bilibili 导入需要安装 yt-dlp")
        try:
            with yt_dlp.YoutubeDL({**_options(), "skip_download": True}) as downloader:
                info = downloader.extract_info(bilibili_video_url(bvid), download=False)
        except yt_dlp.utils.DownloadError as exc:
            try:
                info = _api_payload("/x/web-interface/view", {"bvid": bvid})
                owner = info.get("owner") if isinstance(info.get("owner"), dict) else {}
                return _metadata({**info, "uploader": owner.get("name")}, bvid)
            except BilibiliImportError:
                raise BilibiliImportError("B 站暂时拒绝了网页解析，请稍后重试") from exc
        if not isinstance(info, dict):
            raise BilibiliImportError("B 站返回了无法识别的视频信息")
        return _metadata(info, bvid)


def download_bilibili_video(value: str, directory: Path) -> DownloadedVideo:
    bvid = normalize_bvid(value)
    if yt_dlp is None:
        raise BilibiliImportError("Bilibili 导入需要安装 yt-dlp")
    directory.mkdir(parents=True, exist_ok=True)
    options = {
        **_options(),
        "format": "bv*[height<=1080][vcodec^=avc]+ba/b[height<=1080][vcodec^=avc]/bv*[height<=1080]+ba/b[height<=1080]",
        "merge_output_format": "mp4",
        "outtmpl": str(directory / bvid) + ".%(ext)s",
        "concurrent_fragment_downloads": 2,
    }
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(bilibili_video_url(bvid), download=True)
    except yt_dlp.utils.DownloadError:
        return _download_via_public_api(bvid, directory)
    candidates = [p for p in directory.glob(f"{bvid}.*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    if not candidates:
        return _download_via_public_api(bvid, directory)
    path = max(candidates, key=lambda item: item.stat().st_size)
    metadata = _metadata(info if isinstance(info, dict) else {}, bvid)
    return DownloadedVideo(
        bvid=bvid, title=metadata["title"], uploader=metadata["uploader"],
        duration_seconds=metadata["durationSeconds"], cover_url=metadata["coverUrl"], path=path,
    )
