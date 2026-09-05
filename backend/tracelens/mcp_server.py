"""MCP tools for the TraceLens evidence API.

Run with: python -m tracelens.mcp_server --transport streamable-http
"""
from __future__ import annotations

import argparse
import os
from typing import Any, Literal

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

API_URL = os.getenv("TRACELENS_API_URL", "http://127.0.0.1:9090").rstrip("/")

mcp = FastMCP(
    name="TraceLens",
    instructions="先检索带时间戳的 ASR/OCR 证据，再回答视频事实问题。不得把没有证据支持的内容当作结论。",
    host=os.getenv("MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("MCP_PORT", "8001")),
    streamable_http_path="/mcp",
    stateless_http=True,
    json_response=True,
)


class Citation(BaseModel):
    evidenceId: str = Field(min_length=1)
    timestamp: str = Field(min_length=1)
    content: str = Field(min_length=1, max_length=2000)
    source: Literal["ASR", "OCR", "SYSTEM"] = "ASR"


def _request(method: str, path: str, **kwargs: Any) -> Any:
    headers = {}
    token = os.getenv("TRACELENS_MCP_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = httpx.request(method, f"{API_URL}{path}", headers=headers, timeout=130, **kwargs)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"TraceLens API 请求失败: {exc.__class__.__name__}") from exc
    return response.json() if response.content else None


@mcp.tool(title="列出视频")
def list_videos(limit: int = 50) -> dict[str, Any]:
    """列出当前 TraceLens 视频库。"""
    items = _request("GET", "/api/videos").get("items", [])
    limit = max(1, min(int(limit), 100))
    return {"items": items[:limit], "count": min(len(items), limit), "total": len(items)}


@mcp.tool(title="检索视频证据")
def search_video_evidence(video_id: str, query: str, top_k: int = 8, sources: list[Literal["ASR", "OCR"]] | None = None) -> dict[str, Any]:
    """按问题检索带时间戳的 ASR/OCR 证据。"""
    return _request("GET", "/api/evidence/search", params={
        "video_id": video_id, "query": query, "top_k": max(1, min(int(top_k), 40)),
        "sources": ",".join(sources or []),
    })


@mcp.tool(title="展开证据上下文")
def get_evidence_window(video_id: str, timestamp_ms: int, before_ms: int = 15000, after_ms: int = 15000) -> dict[str, Any]:
    """读取指定时间戳前后的连续证据。"""
    return _request("GET", "/api/evidence/window", params={
        "video_id": video_id, "timestamp_ms": max(0, int(timestamp_ms)),
        "before_ms": max(0, int(before_ms)), "after_ms": max(0, int(after_ms)),
    })


@mcp.tool(title="读取视频时间轴")
def get_video_timeline(video_id: str, start_ms: int = 0, end_ms: int | None = None, limit: int = 200) -> dict[str, Any]:
    """读取视频的 ASR/OCR 时间轴。"""
    payload = _request("GET", "/api/evidence", params={"video_id": video_id})
    items = payload.get("items", [])
    upper = int(end_ms) if end_ms is not None else None
    segments = [
        {"startMs": round(float(item["start_seconds"]) * 1000), "endMs": round(float(item["end_seconds"]) * 1000),
         "source": item.get("source", "ASR"), "content": item["text"], "evidenceId": str(item["id"])}
        for item in items
        if float(item["end_seconds"]) * 1000 >= max(0, int(start_ms))
        and (upper is None or float(item["start_seconds"]) * 1000 <= upper)
    ][:max(1, min(int(limit), 500))]
    return {"videoId": video_id, "segments": segments, "count": len(segments)}


@mcp.tool(title="基于视频追问")
def ask_video(video_id: str, question: str, session_id: str | None = None) -> dict[str, Any]:
    """基于已保存证据回答问题并返回引用。"""
    return _request("POST", "/api/ask", json={"video_id": video_id, "question": question, "session_id": session_id})


@mcp.tool(title="预览哔哩哔哩视频")
def preview_bilibili_video(bvid: str) -> dict[str, Any]:
    """读取公开视频元数据，不下载视频。"""
    return _request("POST", "/api/media/bilibili/preview", json={"bvid": bvid})


@mcp.tool(title="导入哔哩哔哩视频")
def import_bilibili_video(bvid: str) -> dict[str, Any]:
    """异步下载并转录公开视频。"""
    return _request("POST", "/api/media/bilibili/import", json={"bvid": bvid})


@mcp.tool(title="查询导入任务")
def get_import_status(task_id: str) -> dict[str, Any]:
    """查询 Bilibili 导入及转录任务状态。"""
    return _request("GET", "/api/media/bilibili/import-status", params={"task_id": task_id})


def main() -> None:
    parser = argparse.ArgumentParser(description="TraceLens MCP server")
    parser.add_argument("--transport", choices=["stdio", "sse", "streamable-http"], default=os.getenv("MCP_TRANSPORT", "stdio"))
    parser.add_argument("--host", default=os.getenv("MCP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "8001")))
    args = parser.parse_args()
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
