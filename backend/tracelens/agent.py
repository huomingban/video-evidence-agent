"""Agent tools, DeepSeek orchestration, reports, and citation validation."""
from __future__ import annotations
import json
import os
import re
import time
from typing import Any
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
try:
    import httpx
except Exception:
    httpx = None
from .config import UPLOADS_DIR, env_flag, llm_settings, logger
from .models import Evidence
from .retrieval import (
    format_timestamp,
    load_video_evidence,
    parse_time_hints,
    plan_evidence_requirements,
    public_evidence,
    search_evidence,
    search_qdrant,
    tokenize,
)
from .runtime_retrieval import build_runtime_retriever
from .storage import get_connection


def deepseek_failure_result(question: str, error: Exception | str) -> dict[str, Any]:
    """Return a useful provider error without exposing credentials or huge traces."""
    detail = str(error).replace("\n", " ").strip()
    settings = llm_settings()
    if settings.get("api_key"):
        detail = detail.replace(settings["api_key"], "<redacted>")
    detail = re.sub(r"\b(?:sk|ak)-[A-Za-z0-9_-]{8,}\b", "<redacted>", detail)
    detail = re.sub(r"\borg-[A-Za-z0-9_-]{8,}\b", "<org-redacted>", detail)
    detail = detail[:500] or "unknown DeepSeek request error"
    return {
        "question": question,
        "answer": f"DeepSeek 请求失败：{detail}",
        "grounded": False,
        "citations": [],
        "provider": "DeepSeek error",
        "error": detail,
        "trace": [f"DeepSeek request failed: {detail}"],
    }


# Preserve the old function for downstream callers while provider names move to DeepSeek.
kimi_failure_result = deepseek_failure_result


def _create_llm_completion(client: Any, **kwargs: Any) -> Any:
    """Create a completion with DeepSeek-compatible Thinking parameters."""
    settings = llm_settings()
    # The OpenAI SDK does not accept ``thinking`` as a top-level argument.
    # DeepSeek expects it inside extra_body, and Thinking cannot be combined
    # with the forced tool_choice used by the structured Agent. In the normal
    # workflow we therefore omit Thinking entirely.
    extra_body = dict(kwargs.get("extra_body") or {})
    extra_body.pop("thinking", None)
    thinking_enabled = bool(settings.get("thinking_enabled")) and not kwargs.get("tool_choice")
    if kwargs.get("tool_choice") == "auto":
        kwargs.pop("tool_choice", None)
    if thinking_enabled:
        extra_body["thinking"] = {"type": "enabled"}
    if extra_body:
        kwargs["extra_body"] = extra_body
    else:
        kwargs.pop("extra_body", None)
    kwargs.pop("thinking", None)
    for attempt in range(4):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as error:
            status_code = getattr(error, "status_code", None)
            if status_code != 429 or attempt == 3:
                raise
            match = re.search(r"after\s+(\d+)\s+seconds?", str(error), flags=re.I)
            delay = int(match.group(1)) if match else min(30, 2 ** (attempt + 1))
            time.sleep(max(1, min(delay + 1, 30)))


_create_kimi_completion = _create_llm_completion


def is_summary_goal(goal: str) -> bool:
    """Return whether a question asks for a video-level synthesis.

    Summary questions are intentionally detected by meaning rather than only
    by words such as ``总结``. Natural questions like ``视频主要讲了什么``
    must use the representative timeline path as well.
    """
    normalized = re.sub(r"\s+", "", str(goal)).lower()
    patterns = (
        # Canonical UTF-8 forms. Older revisions stored several patterns as
        # mojibake, causing real Chinese summary questions to miss this path.
        r"(?:这(?:个|段)?视频)?(?:主要|大致)?(?:在)?(?:讲|说|介绍|讨论)了?(?:些)?什么",
        r"(?:这(?:个|段)?视频)?(?:的)?(?:主要|核心)?内容(?:是|有什么|包括)?什么?",
        r"(?:总结|概括|综述|梳理).{0,10}(?:这(?:个|段)?视频|视频内容|主要内容|核心观点)",
        r"(?:这(?:个|段)?视频|视频内容).{0,8}(?:总结|概括|梳理)",
        r"提炼(?:这(?:个|段)?视频)?(?:的)?(?:核心|主要)(?:观点|内容|结论)",
        r"(?:summari[sz]e|overview|main\s+(?:content|point|topic))",
        r"(?:这个|该|本|这段)?视频(?:最)?(?:主要)?(?:在)?(?:讲|说|介绍|讨论|分析)(?:了)?(?:些)?什么",
        r"(?:这个|该|本|这段)?视频(?:的)?(?:最)?(?:主要|核心)?内容(?:是|有)?什么",
        r"(?:总结|概括|综述|梳理|理解).{0,10}(?:视频|主要内容|核心内容|核心观点)",
        r"(?:生成|整理)(?:一份)?(?:学习笔记|视频摘要)",
        r"提炼(?:视频)?(?:核心观点|主要观点|关键结论)",
        r"总结|概括|综述|梳理|核心内容|主要内容|关键观点|学习笔记|视频摘要|summari[sz]e|overview",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def build_agent_plan(goal: str) -> dict[str, Any]:
    """Return the human-readable intent plan shown in Agent traces."""
    normalized = " ".join(str(goal).split())
    if is_summary_goal(normalized):
        intent = "STRUCTURED_SUMMARY"
        tasks = [
            "读取视频元数据并确定总结范围",
            "抽取时间轴上具有代表性的 ASR/OCR 证据",
            "综合主题、观点和示例生成结构化摘要",
            "校验结论与时间戳证据",
        ]
    elif re.search(r"会议|决策|待办|负责人|风险", normalized):
        intent = "MEETING_REVIEW"
        tasks = [
            "识别会议分析范围",
            "检索决策、待办、负责人和风险证据",
            "展开关键证据窗口",
            "校验每条结论的时间戳引用",
        ]
    elif re.search(r"步骤|操作|教程|流程|怎么做|如何(?:操作|完成|实现|使用|配置|部署)", normalized):
        intent = "OPERATION_GUIDE"
        tasks = [
            "识别操作目标",
            "检索步骤、界面动作和前后依赖",
            "展开关键证据窗口并按顺序组织",
            "校验步骤引用和时间戳",
        ]
    else:
        intent = "EVIDENCE_QA"
        tasks = [
            "理解问题中的主体、关系和答案范围",
            "检索最相关的 ASR/OCR 时间轴证据",
            "展开命中片段前后的上下文",
            "校验回答是否完全由视频证据支持",
        ]
    return {
        "understoodGoal": normalized,
        "intent": intent,
        "tasks": tasks,
        "steps": [
            {"stage": "CONTEXT", "tools": ["get_video_metadata"]},
            {"stage": "RETRIEVAL", "tools": ["search_timeline", "get_evidence_window"]},
            {"stage": "CRITIC", "tools": ["verify_citations", "generate_report"]},
        ],
    }


class AgentToolbox:
    """Tools the model may choose while investigating one video."""

    def __init__(self, video_id: str | None, question: str) -> None:
        self.video_id = video_id
        self.question = question
        self.evidence = load_video_evidence(video_id)
        self.retriever = build_runtime_retriever(video_id)
        self._trace: list[dict[str, Any]] = []
        self._goal_coverage_plan: dict[str, Any] | None = None
        self._goal_evidence_sufficiency: dict[str, Any] | None = None

    def plan(self) -> dict[str, Any]:
        normalized = re.sub(r"\s+", "", self.question)
        is_summary = is_summary_goal(self.question)
        is_time_question = bool(re.search(r"\d{1,3}:\d{2}|\d+秒|什么时候|何时|哪里", normalized))
        if is_summary:
            intent = "STRUCTURED_SUMMARY"
            retrieval_tools = ["get_video_metadata", "get_timeline_overview"]
        elif is_time_question:
            intent = "TIMESTAMP_QA"
            retrieval_tools = ["get_video_metadata", "search_timeline", "get_evidence_window"]
        else:
            intent = "EVIDENCE_QA"
            retrieval_tools = ["get_video_metadata", "search_timeline"]
        return {
            "intent": intent,
            "stages": [
                {"name": "CONTEXT", "tools": ["get_video_metadata"]},
                {"name": "RETRIEVAL", "tools": retrieval_tools},
                {"name": "VALIDATION", "tools": ["verify_citations"]},
                {"name": "REPORT", "tools": ["generate_report"]},
            ],
        }

    @staticmethod
    def _evenly_spaced(items: list[Evidence], count: int) -> list[Evidence]:
        if len(items) <= count:
            return items
        if count <= 1:
            return [items[len(items) // 2]]
        indexes = {
            round(index * (len(items) - 1) / (count - 1))
            for index in range(count)
        }
        return [items[index] for index in sorted(indexes)]

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_video_metadata",
                    "description": "读取当前视频的文件状态、证据数量和时间范围。开始分析时调用。",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_timeline_overview",
                    "description": "从完整时间轴均匀抽取代表性证据，用于总结视频主题。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "max_segments": {"type": "integer", "minimum": 4, "maximum": 20},
                        },
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_timeline",
                    "description": "按问题检索最相关的带时间戳 ASR 证据。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "minLength": 1},
                            "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                            "sources": {
                                "type": "array",
                                "items": {"type": "string", "enum": ["ASR", "OCR"]},
                                "maxItems": 2,
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_evidence_window",
                    "description": "展开某个时间点前后的连续证据，避免脱离上下文理解。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timestamp_ms": {"type": "integer", "minimum": 0},
                            "before_ms": {"type": "integer", "minimum": 0, "maximum": 120000},
                            "after_ms": {"type": "integer", "minimum": 0, "maximum": 120000},
                        },
                        "required": ["timestamp_ms"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "verify_citations",
                    "description": "检查候选引用是否真实存在并且能覆盖当前问题。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "citation_ids": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "maxItems": 20,
                            },
                        },
                        "required": ["citation_ids"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_report",
                    "description": "提交最终结构化报告。必须是最后一步；证据不足时 answerable=false。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answerable": {"type": "boolean"},
                            "final_answer": {"type": "string", "minLength": 1, "maxLength": 2000},
                            "citation_ids": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "maxItems": 20,
                            },
                            "support_level": {
                                "type": "string",
                                "enum": ["DIRECT", "SUMMARY", "INFERENCE", "INSUFFICIENT"],
                            },
                        },
                        "required": ["answerable", "final_answer", "citation_ids", "support_level"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    # Reference-project naming. Keeping this alias makes the structured
    # Planner/Retriever/Verifier/Writer/Critic pipeline independently usable.
    def tool_schemas(self) -> list[dict[str, Any]]:
        return self.schemas()

    def prefetch_goal_evidence(self, goal: str, top_k: int = 8) -> dict[str, Any]:
        """Run deterministic coverage retrieval once and cache its result."""
        coverage_plan = plan_evidence_requirements(goal)
        requirements = coverage_plan.get("requirements", [])[:6]
        merged: dict[str, dict[str, Any]] = {}
        requirement_status: list[dict[str, Any]] = []

        def add(item: dict[str, Any], requirement_id: str, rank: int) -> None:
            key = str(item.get("segmentId") or item.get("evidenceId") or item.get("evidence_id"))
            if not key or key in {"None", "0"}:
                return
            current = merged.setdefault(key, dict(item))
            ids = current.setdefault("coverageRequirementIds", [])
            if requirement_id not in ids:
                ids.append(requirement_id)
            reasons = current.setdefault("coverageSelectionReasons", [])
            reason = "REQUIREMENT_PRIMARY" if rank == 1 else "REQUIREMENT_CANDIDATE"
            if reason not in reasons:
                reasons.append(reason)

        for index, requirement in enumerate(requirements, start=1):
            requirement_id = str(requirement.get("requirementId") or f"R{index}")
            query = str(requirement.get("query") or goal)
            # Search each source independently so OCR cannot be crowded out by ASR.
            source_matches: list[dict[str, Any]] = []
            for source in ("OCR", "ASR"):
                source_result = self._search({"query": query, "top_k": max(2, top_k // 2), "sources": [source]})
                source_matches.extend(dict(item) for item in source_result.get("matches", []))
            by_key = {
                str(item.get("segmentId") or item.get("evidenceId") or item.get("evidence_id")): item
                for item in source_matches
            }
            candidates = sorted(
                by_key.values(),
                key=lambda item: (
                    0 if str(item.get("source", "")).upper() == "OCR" else 1,
                    int(item.get("startMs", 0)),
                ),
            )[:top_k]
            for rank, item in enumerate(candidates, start=1):
                add(item, requirement_id, rank)
            requirement_status.append({
                "requirementId": requirement_id,
                "query": query,
                "candidateCount": len(candidates),
                "satisfied": bool(candidates),
                "candidateEvidenceIds": [
                    str(item.get("evidenceId", item.get("evidence_id"))) for item in candidates
                ],
            })

        matches = sorted(
            merged.values(),
            key=lambda item: (
                0 if str(item.get("source", "")).upper() == "OCR" else 1,
                int(item.get("startMs", 0)),
                str(item.get("segmentId", "")),
            ),
        )
        visible_limit = max(top_k, 12)
        ocr = [item for item in matches if str(item.get("source", "")).upper() == "OCR"]
        asr = [item for item in matches if str(item.get("source", "")).upper() != "OCR"]
        ocr_quota = min(len(ocr), max(1, (visible_limit * 3 + 4) // 5))
        matches = (ocr[:ocr_quota] + asr)[:visible_limit]
        result = {
            "ok": True,
            "query": goal,
            "matches": matches,
            "coveragePlan": coverage_plan,
            "evidenceSufficiency": {
                "decision": "SUFFICIENT_CANDIDATES" if all(item["satisfied"] for item in requirement_status) else "PARTIAL_EVIDENCE",
                "fullyCovered": bool(requirement_status) and all(item["satisfied"] for item in requirement_status),
                "requirementCount": len(requirement_status),
                "satisfiedRequirementCount": sum(bool(item["satisfied"]) for item in requirement_status),
                "requirements": requirement_status,
            },
        }
        matches = result.get("matches", [])
        self._goal_coverage_plan = dict(result.get("coveragePlan") or {}) or None
        self._goal_evidence_sufficiency = dict(result.get("evidenceSufficiency") or {}) or None
        return {
            **result,
        }

    def goal_evidence_sufficiency(self) -> dict[str, Any] | None:
        return json.loads(json.dumps(self._goal_evidence_sufficiency, ensure_ascii=False)) if self._goal_evidence_sufficiency else None

    def evidence_for_question(self, question: str, max_segments: int = 18) -> list[dict[str, Any]]:
        """Select a small question-specific context for follow-up answers."""
        if is_summary_goal(question):
            return self._overview({"max_segments": max_segments}).get("segments", [])
        result = self._search({"query": question, "top_k": min(8, max_segments)})
        meaningful = [
            dict(item) for item in result.get("matches", [])
            if float(item.get("score", 0.0)) > 0
            or str(item.get("selectionReason", "")) == "TIME_ANCHOR"
        ]
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for match in meaningful[:4]:
            window = self._window({
                "timestamp_ms": int(match.get("startMs", 0)),
                "before_ms": 15000,
                "after_ms": 15000,
            })
            for item in window.get("segments", []):
                key = str(item.get("segmentId"))
                if key in seen:
                    continue
                selected.append(item)
                seen.add(key)
                if len(selected) >= max_segments:
                    return selected
        return selected or meaningful[:max_segments]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            if name == "get_video_metadata":
                result = self._metadata()
            elif name == "get_timeline_overview":
                result = self._overview(arguments)
            elif name == "search_timeline":
                result = self._search(arguments)
            elif name == "get_evidence_window":
                result = self._window(arguments)
            elif name == "verify_citations":
                result = self._verify(arguments)
            elif name == "generate_report":
                result = self._report(arguments)
            else:
                result = {"ok": False, "error": f"未知 Agent 工具: {name}"}
        except (TypeError, ValueError, KeyError) as error:
            result = {"ok": False, "error": str(error)}
        self._trace.append({
            "tool": name,
            "arguments": arguments,
            "success": bool(result.get("ok")),
            "result_preview": json.dumps(result, ensure_ascii=False)[:500],
        })
        return result

    def trace(self) -> list[dict[str, Any]]:
        return list(self._trace)

    def _metadata(self) -> dict[str, Any]:
        files = []
        if self.video_id:
            directory = UPLOADS_DIR / self.video_id
            if directory.is_dir():
                files = [path.name for path in directory.iterdir() if path.is_file()]
        return {
            "ok": True,
            "video_id": self.video_id,
            "uploaded_files": files,
            "evidence_count": len(self.evidence),
            "timeline_end_ms": round(max((item.end_seconds for item in self.evidence), default=0) * 1000),
        }

    def _overview(self, arguments: dict[str, Any]) -> dict[str, Any]:
        count = max(4, min(int(arguments.get("max_segments", 12)), 20))
        # ASR usually has many more rows than OCR. Reserve part of the
        # overview for each available source so OCR remains visible to the
        # Planner and Writer instead of being crowded out by ASR rows.
        sources = sorted({item.source.upper() for item in self.evidence})
        selected: list[Evidence] = []
        if len(sources) > 1:
            per_source = max(1, count // len(sources))
            for source in sources:
                selected.extend(self._evenly_spaced(
                    [item for item in self.evidence if item.source.upper() == source],
                    per_source,
                ))
            selected = sorted(selected, key=lambda item: (item.start_seconds, item.id))[:count]
        else:
            selected = self._evenly_spaced(self.evidence, count)
        return {
            "ok": True,
            "sampling": "even_timeline",
            "segments": [self._structured_evidence(item) for item in selected],
        }

    def _search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or self.question).strip()
        if not query:
            raise ValueError("query 不能为空")
        count = max(1, min(int(arguments.get("top_k", 6)), 10))
        requested_sources = {
            str(source).upper() for source in (arguments.get("sources") or [])
            if str(source).upper() in {"ASR", "OCR"}
        }
        retrieval = self.retriever.search(
            query,
            top_k=max(count * 3, 12),
            sources=sorted(requested_sources) if requested_sources else None,
        )
        # Search each source independently when both modalities are requested.
        # This is the important part that prevents a long ASR transcript from
        # filling the candidate window before slide text is considered.
        if requested_sources == {"ASR", "OCR"}:
            source_items: dict[str, dict[str, Any]] = {}
            for source in ("OCR", "ASR"):
                source_result = self.retriever.search(query, top_k=max(count * 2, 12), sources=[source])
                for item in source_result.get("matches", []):
                    source_items.setdefault(str(item.get("segmentId") or item.get("evidenceId")), dict(item))
            time_hints = parse_time_hints(query)
            ranked = sorted(source_items.values(), key=lambda item: (
                min((abs(int(item.get("startMs", 0)) - hint) for hint in time_hints), default=10**12),
                0 if str(item.get("source", "")).upper() == "OCR" else 1,
                -float(item.get("score", 0.0)),
                int(item.get("startMs", 0)),
            ))
            ocr = [item for item in ranked if str(item.get("source", "")).upper() == "OCR"]
            asr = [item for item in ranked if str(item.get("source", "")).upper() != "OCR"]
            ocr_quota = min(len(ocr), max(1, (count * 3 + 4) // 5))
            matches = (ocr[:ocr_quota] + asr)[:count]
        else:
            matches = [dict(item) for item in retrieval.get("matches", [])][:count]
        return {
            "ok": True,
            "query": query,
            "matches": matches,
            "retrievalMode": retrieval.get("retrievalMode", "COVERAGE_AWARE"),
            "coveragePlan": retrieval.get("coveragePlan"),
            "evidenceSufficiency": retrieval.get("evidenceSufficiency"),
            "timeHintsMs": retrieval.get("timeHintsMs", []),
        }

    def _window(self, arguments: dict[str, Any]) -> dict[str, Any]:
        timestamp = max(0, int(arguments["timestamp_ms"]))
        before = max(0, min(int(arguments.get("before_ms", 15000)), 120000))
        after = max(0, min(int(arguments.get("after_ms", 15000)), 120000))
        start, end = timestamp - before, timestamp + after
        matches = [
            item for item in self.evidence
            if item.end_seconds * 1000 >= start and item.start_seconds * 1000 <= end
        ][:40]
        return {
            "ok": True,
            "window_start_ms": max(0, start),
            "window_end_ms": end,
            "segments": [self._structured_evidence(item) for item in matches],
        }

    @staticmethod
    def _structured_evidence(item: Evidence) -> dict[str, Any]:
        """Expose both legacy snake_case and reference-style fields."""
        result = public_evidence(item)
        result.update({
            "evidenceId": str(item.id),
            "segmentId": f"evidence-{item.id}",
            "startMs": round(item.start_seconds * 1000),
            "endMs": round(item.end_seconds * 1000),
            "content": item.text,
            "source": item.source,
        })
        return result

    def _ids(self, arguments: dict[str, Any]) -> list[int]:
        raw_ids = (
            arguments.get("citation_ids")
            or arguments.get("evidenceIds")
            or arguments.get("citations")
            or arguments.get("evidence")
            or []
        )
        if not isinstance(raw_ids, list):
            return []
        ids = []
        for value in raw_ids:
            if isinstance(value, dict):
                value = value.get(
                    "dbEvidenceId",
                    value.get("evidence_id", value.get("evidenceId", value.get("id"))),
                )
            if isinstance(value, int) and value not in ids:
                ids.append(value)
            elif isinstance(value, str) and value.isdigit() and int(value) not in ids:
                ids.append(int(value))
        return ids

    def _verify(self, arguments: dict[str, Any]) -> dict[str, Any]:
        ids = self._ids(arguments)
        by_id = {item.id: item for item in self.evidence}
        selected = [by_id[item_id] for item_id in ids if item_id in by_id]
        coverage = verify_coverage_tool(self.question, selected)
        return {
            "ok": True,
            "valid": len(selected) == len(ids) and bool(selected),
            "valid_ids": [item.id for item in selected],
            "coverage": coverage,
        }

    def _report(self, arguments: dict[str, Any]) -> dict[str, Any]:
        answerable = bool(arguments.get("answerable"))
        final_answer = str(
            arguments.get("final_answer") or arguments.get("finalAnswer") or ""
        ).strip()
        ids = self._ids(arguments)
        by_id = {item.id: item for item in self.evidence}
        selected = [by_id[item_id] for item_id in ids if item_id in by_id]
        invalid_ids = [item_id for item_id in ids if item_id not in by_id]
        coverage = verify_coverage_tool(self.question, selected)
        # The structured Verifier already checked requirement-level entailment.
        # Do not let the legacy whole-question token ratio reject a valid
        # timestamp answer merely because the question contains words such as
        # "around" or "what" that cannot appear in the transcript.
        if answerable and selected and (
            parse_time_hints(self.question) or is_summary_goal(self.question)
        ):
            coverage["adequate"] = True
            coverage["reason"] = (
                "Representative timeline evidence covers the video-level synthesis"
                if is_summary_goal(self.question)
                else "Timestamp requirement is covered by selected evidence"
            )
        if not final_answer:
            return {"ok": False, "accepted": False, "error": "final_answer 不能为空"}
        if answerable and (not selected or invalid_ids or not coverage["adequate"]):
            return {
                "ok": True,
                "accepted": False,
                "error": "报告未通过引用或证据覆盖校验，请继续检索；证据不足则提交 answerable=false",
                "valid_ids": [item.id for item in selected],
                "coverage": coverage,
            }
        if not answerable:
            selected = []
        return {
            "ok": True,
            "accepted": True,
            "answerable": answerable,
            "final_answer": final_answer,
            "support_level": str(arguments.get("support_level") or ("DIRECT" if answerable else "INSUFFICIENT")),
            "citations": [self._structured_evidence(item) for item in selected],
        }

def search_semantic_tool(question: str, video_id: str | None) -> list[Evidence]:
    """Search using vector similarity"""
    qdrant_hits = search_qdrant(question, video_id, limit=3)
    return qdrant_hits


def search_keyword_tool(question: str, video_id: str | None) -> list[Evidence]:
    """Search using keyword matching"""
    question_tokens = tokenize(question)
    with get_connection() as connection:
        rows = connection.execute(
            "SELECT id, video_id, start_seconds, end_seconds, text, source FROM evidence "
            "WHERE (? IS NULL OR video_id = ?) ORDER BY start_seconds",
            (video_id, video_id),
        ).fetchall()

    scored: list[tuple[int, Evidence]] = []
    for row in rows:
        evidence = Evidence(**dict(row))
        overlap = len(question_tokens & tokenize(evidence.text))
        if overlap:
            scored.append((overlap, evidence))
    scored.sort(key=lambda item: (-item[0], item[1].start_seconds))
    return [evidence for _, evidence in scored[:3]]


def verify_coverage_tool(question: str, evidence: list[Evidence]) -> dict[str, Any]:
    """Verify if evidence covers question requirements"""
    if not evidence:
        return {"adequate": False, "reason": "No evidence found"}
    
    q_tokens = tokenize(question)
    e_tokens = set()
    for e in evidence:
        e_tokens.update(tokenize(e.text))
    
    overlap_ratio = len(q_tokens & e_tokens) / max(len(q_tokens), 1)
    adequate = overlap_ratio >= float(os.getenv("MIN_EVIDENCE_COVERAGE", "0.3"))
    
    return {
        "adequate": adequate,
        "reason": f"Coverage: {overlap_ratio:.1%} token overlap" if overlap_ratio > 0 else "Fallback retrieval",
        "evidence_count": len(evidence),
    }

def evidence_citations(evidence: list[Evidence]) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": item.id,
            "timestamp": f"{format_timestamp(item.start_seconds)} - {format_timestamp(item.end_seconds)}",
            "text": item.text,
            "source": item.source,
        }
        for item in evidence
    ]

def parse_kimi_json(content: str) -> dict[str, Any] | None:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _message_value(message: Any, name: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(name, default)
    return getattr(message, name, default)


def _tool_call_value(call: Any, name: str, default: Any = None) -> Any:
    if isinstance(call, dict):
        return call.get(name, default)
    return getattr(call, name, default)


def _function_value(call: Any, name: str, default: Any = None) -> Any:
    function = _tool_call_value(call, "function", {}) or {}
    if isinstance(function, dict):
        return function.get(name, default)
    return getattr(function, name, default)


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _assistant_tool_message(message: Any, tool_calls: list[Any]) -> dict[str, Any]:
    normalized_calls = []
    for call in tool_calls:
        normalized_calls.append({
            "id": str(_tool_call_value(call, "id", "tool-call")),
            "type": "function",
            "function": {
                "name": str(_function_value(call, "name", "")),
                "arguments": str(_function_value(call, "arguments", "{}")),
            },
        })
    return {
        "role": "assistant",
        "content": _message_value(message, "content"),
        "tool_calls": normalized_calls,
    }

def run_kimi_agent(question: str, video_id: str | None, history: list[dict[str, str]] | None = None) -> dict[str, Any] | None:
    """Let Kimi choose evidence tools until it submits an accepted report."""
    if not kimi_is_configured():
        return None

    settings = llm_settings()
    toolbox = AgentToolbox(video_id, question)
    plan = toolbox.plan()
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "你是一个严谨的视频证据 Agent。只能使用工具返回的证据回答，不能补充外部知识。"
                "先根据问题选择合适的工具：概括问题使用时间轴概览，具体问题使用时间轴检索，"
                "需要上下文时展开证据窗口。完成调查后必须调用 generate_report。"
                "如果证据不足，必须提交 answerable=false，并明确说明视频没有提供足够依据。"
            ),
        },
    ]
    if env_flag("DEEPSEEK_SEND_SESSION_HISTORY", env_flag("KIMI_SEND_SESSION_HISTORY", False)) and history:
        messages.extend(
            {
                "role": item["role"],
                "content": item["content"],
            }
            for item in history[-12:]
            if item.get("role") in {"user", "assistant"} and item.get("content", "").strip()
        )
    messages.append(
        {
            "role": "user",
            "content": (
                f"视频 ID：{video_id or '未指定'}\n问题：{question}\n"
                f"证据需求计划：{json.dumps(plan, ensure_ascii=False)}"
            ),
        }
    )
    try:
        http_client = None
        if httpx is not None:
            http_client = httpx.Client(
                trust_env=settings["trust_env"],
                proxy=settings["proxy"],
                timeout=settings["timeout"],
            )
        client = OpenAI(
            api_key=settings["api_key"],
            base_url=settings["base_url"],
            timeout=settings["timeout"],
            max_retries=0,
            http_client=http_client,
        )
        max_steps = max(2, min(int(os.getenv("AGENT_MAX_TOOL_STEPS", "8")), 12))
        for step in range(max_steps):
            response = _create_llm_completion(client,
                model=settings["model"],
                temperature=settings["temperature"],
                messages=messages,
                tools=toolbox.schemas(),
                tool_choice="auto",
            )
            message = response.choices[0].message
            tool_calls = list(_message_value(message, "tool_calls", []) or [])
            content = str(_message_value(message, "content", "") or "")

            if tool_calls:
                messages.append(_assistant_tool_message(message, tool_calls))
                for call in tool_calls:
                    call_id = str(_tool_call_value(call, "id", f"tool-call-{step}"))
                    name = str(_function_value(call, "name", ""))
                    arguments = _parse_tool_arguments(_function_value(call, "arguments", "{}"))
                    result = toolbox.execute(name, arguments)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                    if name == "generate_report" and result.get("accepted"):
                        return {
                            "question": question,
                            "answer": result["final_answer"],
                            "grounded": bool(result["answerable"]),
                            "citations": result["citations"],
                            "provider": "DeepSeek",
                            "agent_plan": plan,
                            "support_level": result["support_level"],
                            "trace": [
                                f"Agent step {step + 1}: DeepSeek selected {item['tool']}"
                                for item in toolbox.trace()
                            ],
                            "tool_trace": toolbox.trace(),
                        }
                continue

            # Some OpenAI-compatible endpoints do not return tool_calls reliably.
            # Accept a JSON report as a compatibility fallback, then keep the same gate.
            parsed = parse_kimi_json(content)
            if parsed:
                arguments = {
                    "answerable": parsed.get("answerable", True),
                    "final_answer": parsed.get("final_answer", parsed.get("answer", "")),
                    "citation_ids": parsed.get("citation_ids", parsed.get("citations", [])),
                    "support_level": parsed.get("support_level", "DIRECT"),
                }
                result = toolbox.execute("generate_report", arguments)
                if result.get("accepted"):
                    return {
                        "question": question,
                        "answer": result["final_answer"],
                        "grounded": bool(result["answerable"]),
                        "citations": result["citations"],
                        "provider": "DeepSeek",
                        "agent_plan": plan,
                        "support_level": result["support_level"],
                        "trace": [
                            f"Agent step {step + 1}: compatibility report submitted"
                        ],
                        "tool_trace": toolbox.trace(),
                    }
            messages.extend([
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": "请继续调用证据工具，并通过 generate_report 提交最终报告，不要只输出普通文本。",
                },
            ])
        raise RuntimeError("DeepSeek Agent exceeded its tool-step budget without an accepted report")
    except Exception as error:
        logger.exception("DeepSeek Agent execution failed")
        return kimi_failure_result(question, error)


def generate_kimi_answer(question: str, evidence: list[Evidence]) -> tuple[str, list[Evidence]] | None:
    if not kimi_is_configured():
        return None

    settings = llm_settings()
    context = [
        {
            "evidence_id": item.id,
            "timestamp": f"{format_timestamp(item.start_seconds)} - {format_timestamp(item.end_seconds)}",
            "text": item.text,
        }
        for item in evidence
    ]
    prompt = (
        "请只根据给定的视频证据回答问题。不要补充证据中没有的信息。"
        "必须返回 JSON，不要使用 Markdown，格式为："
        '{"answer":"...","citation_ids":[1,2]}。'
        "citation_ids 只能填写真正支持答案的 evidence_id；如果证据不足，answer 写明证据不足，"
        "并将 citation_ids 设为空数组。请用中文回答。\n\n"
        f"问题：{question}\n证据：{json.dumps(context, ensure_ascii=False)}"
    )
    try:
        http_client = None
        if httpx is not None:
            http_client = httpx.Client(
                trust_env=settings["trust_env"],
                proxy=settings["proxy"],
                timeout=settings["timeout"],
            )
        client = OpenAI(
            api_key=settings["api_key"],
            base_url=settings["base_url"],
            timeout=settings["timeout"],
            max_retries=0,
            http_client=http_client,
        )
        response = _create_llm_completion(client,
            model=settings["model"],
            temperature=settings["temperature"],
            messages=[
                {
                    "role": "system",
                    "content": "你是一个严谨的视频证据问答助手。回答必须可由提供的证据支持。",
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        parsed = parse_kimi_json(content)
        if not parsed or not isinstance(parsed.get("answer"), str):
            raise ValueError("DeepSeek returned an invalid JSON answer")
        ids = parsed.get("citation_ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError("DeepSeek returned no valid citation ids")
        evidence_by_id = {item.id: item for item in evidence}
        selected = [evidence_by_id[int(item_id)] for item_id in ids if str(item_id).isdigit() and int(item_id) in evidence_by_id]
        if not selected:
            raise ValueError("DeepSeek cited evidence outside the retrieved set")
        return parsed["answer"].strip(), selected
    except Exception:
        logger.exception("Kimi answer generation failed")
        return None


def answer_from_evidence(question: str, evidence: list[Evidence]) -> dict[str, Any]:
    if not evidence:
        return {
            "answer": "当前证据中没有找到足够信息，暂时无法可靠回答。",
            "grounded": False,
            "citations": [],
            "provider": "refusal",
        }

    kimi_result = generate_kimi_answer(question, evidence)
    if kimi_result is not None:
        answer, cited_evidence = kimi_result
        return {
            "answer": answer,
            "grounded": True,
            "citations": evidence_citations(cited_evidence),
            "provider": "DeepSeek",
        }

    if kimi_is_configured():
        return {
            "answer": "DeepSeek 暂时不可用，未生成未经验证的回答。请检查 API Key、模型名称和网络连接。",
            "grounded": False,
            "citations": [],
            "provider": "DeepSeek error",
        }

    answer = "根据检索到的视频证据，相关内容包括：" + "；".join(
        item.text for item in evidence
    )
    return {
        "answer": answer,
        "grounded": True,
        "citations": evidence_citations(evidence),
        "provider": "local fallback",
    }
    
def deepseek_is_configured() -> bool:
    settings = llm_settings()
    return bool(OpenAI is not None and settings["enabled"] and settings["api_key"])


kimi_is_configured = deepseek_is_configured


class DeepSeekStructuredProvider:
    """OpenAI-compatible DeepSeek adapter used by the structured LangGraph."""

    def __init__(self) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed")
        settings = llm_settings()
        http_client = None
        if httpx is not None:
            http_client = httpx.Client(
                trust_env=settings["trust_env"],
                proxy=settings["proxy"],
                timeout=settings["timeout"],
            )
        self._client = OpenAI(
            api_key=settings["api_key"],
            base_url=settings["base_url"],
            timeout=settings["timeout"],
            max_retries=0,
            http_client=http_client,
        )
        self._settings = settings

    @staticmethod
    def _message_dict(message: Any) -> dict[str, Any]:
        calls = []
        for call in list(_message_value(message, "tool_calls", []) or []):
            calls.append({
                "id": str(_tool_call_value(call, "id", "tool-call")),
                "type": "function",
                "function": {
                    "name": str(_function_value(call, "name", "")),
                    "arguments": str(_function_value(call, "arguments", "{}")),
                },
            })
        return {
            "role": "assistant",
            "content": _message_value(message, "content"),
            "tool_calls": calls,
        }

    def _completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: dict[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._settings["model"],
            "temperature": self._settings["temperature"],
            "messages": messages,
            "tools": tools,
        }
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        # The OpenAI-compatible SDK returns a ChatCompletion object.  The
        # structured graph consumes a plain message mapping, so normalize the
        # SDK object at this boundary instead of leaking it into the graph.
        try:
            response = _create_llm_completion(self._client, **kwargs)
        except Exception as error:
            unsupported_choice = tool_choice is not None and (
                "does not support this tool_choice" in str(error)
                or re.search(r"unsupported.*tool_choice", str(error), flags=re.I) is not None
            )
            if not unsupported_choice:
                raise
            # Some DeepSeek thinking configurations accept tools but reject
            # a forced function choice. Let the model choose the only schema.
            kwargs.pop("tool_choice", None)
            response = _create_llm_completion(self._client, **kwargs)
        choices = _message_value(response, "choices", []) or []
        if not choices:
            raise RuntimeError("DeepSeek response did not contain choices")
        return self._message_dict(_message_value(choices[0], "message", {}))

    def follow_up(
        self,
        report: str,
        question: str,
        *,
        history: list[dict[str, str]] | None = None,
        memory_summary: str = "",
        evidence_timeline: str = "",
    ) -> str:
        """Answer a follow-up as prose without rebuilding a structured report."""
        messages: list[dict[str, Any]] = [{
            "role": "system",
            "content": (
                "你是视频证据问答 Agent 的追问模块。只根据已有报告、会话上下文和本次检索到的证据回答。"
                "输出一段简洁自然的中文，不要输出 JSON、标题、证据列表或时间轴。"
                "如果证据不足，明确说明无法从视频确定，不要补充外部知识。"
            ),
        }]
        messages.append({
            "role": "user",
            "content": (
                f"已有初次报告：\n{report[:10000]}\n\n"
                f"本轮相关视频证据：\n{evidence_timeline[:10000]}\n\n"
                f"会话记忆摘要：\n{memory_summary[:3000]}"
            ),
        })
        if history:
            messages.extend(
                {"role": item["role"], "content": item["content"]}
                for item in history[-20:]
                if item.get("role") in {"user", "assistant"} and item.get("content", "").strip()
            )
        messages.append({
            "role": "user",
            "content": f"追问：{question}",
        })
        response = _create_llm_completion(
            self._client,
            model=self._settings["model"],
            temperature=self._settings["temperature"],
            messages=messages,
        )
        choices = _message_value(response, "choices", []) or []
        if not choices:
            raise RuntimeError("DeepSeek follow-up response did not contain choices")
        content = _message_value(choices[0], "message", {})
        answer = str(_message_value(content, "content", "") or "").strip()
        if not answer:
            raise RuntimeError("DeepSeek follow-up returned an empty answer")
        return answer


def run_structured_deepseek_agent(
    question: str,
    video_id: str | None,
) -> dict[str, Any] | None:
    """Run the reference-style five-stage Agent when explicitly enabled."""
    if not deepseek_is_configured():
        return None
    try:
        from .agent_structured import run_structured_evidence_agent

        provider = DeepSeekStructuredProvider()
        toolbox = AgentToolbox(video_id, question)
        result = run_structured_evidence_agent(provider, toolbox, question)
        report = result.get("report") or result
        report_answer = report.get("finalAnswer") or report.get("final_answer")
        report_answerable = report.get("answerable", result.get("answerable", False))
        report_evidence = report.get("evidence", result.get("citations", []))
        return {
            "question": question,
            "answer": str(report_answer or result.get("answer") or ""),
            "grounded": bool(report_answerable),
            "citations": [
                {
                    "evidence_id": int(item.get(
                        "dbEvidenceId",
                        item.get("evidence_id", item.get("id", 0)),
                    ) or 0),
                    "timestamp": (
                        format_timestamp(float(item.get("timestampMs", item.get("startMs", 0))) / 1000)
                        if "timestampMs" in item or "startMs" in item else
                        str(item.get("timestamp", ""))
                    ),
                    "text": item.get("content", ""),
                    "source": item.get("source", "ASR"),
                }
                for item in report_evidence[:9]
                if int(item.get(
                    "dbEvidenceId",
                    item.get("evidence_id", item.get("id", 0)),
                ) or 0) > 0
            ],
            "provider": "DeepSeek structured Agent",
            "kind": "initial_report",
            "report": {
                "answerable": bool(report_answerable),
                "title": str(report.get("title") or "视频分析报告"),
                "finalAnswer": str(report_answer or result.get("answer") or ""),
                "conclusions": list(report.get("conclusions") or []),
                "evidence": [
                    {
                        "evidence_id": int(item.get("dbEvidenceId", item.get("evidence_id", item.get("id", 0))) or 0),
                        "timestamp": format_timestamp(float(item.get("timestampMs", item.get("startMs", 0))) / 1000),
                        "text": item.get("content", item.get("text", "")),
                        "source": item.get("source", "ASR"),
                    }
                    for item in report_evidence[:9]
                ],
                "suggestions": list(report.get("suggestions") or []),
            },
            "support_level": "DIRECT" if report_answerable else "INSUFFICIENT",
            "agent_graph": result.get("agentGraph"),
            "trace": [
                f"Node: {node}"
                for node in (result.get("agentGraph") or {}).get("nodes", [])
            ],
            "tool_trace": toolbox.trace(),
        }
    except Exception as error:
        logger.exception("Structured DeepSeek Agent execution failed")
        return deepseek_failure_result(question, error)


KimiStructuredProvider = DeepSeekStructuredProvider
run_structured_kimi_agent = run_structured_deepseek_agent


def run_deepseek_follow_up(
    question: str,
    video_id: str | None,
    previous_report: dict[str, Any],
    history: list[dict[str, str]] | None = None,
    memory_summary: str = "",
) -> dict[str, Any] | None:
    if not deepseek_is_configured():
        return None
    try:
        toolbox = AgentToolbox(video_id, question)
        selected = toolbox.evidence_for_question(question, max_segments=18)
        evidence_context = "\n".join(
            f"[{item.get('source', 'ASR')}] {item.get('timestamp', '')} {item.get('content', item.get('text', ''))}"
            for item in selected
        )
        provider = DeepSeekStructuredProvider()
        answer = provider.follow_up(
            str(previous_report.get("answer") or previous_report.get("finalAnswer") or ""),
            question,
            history=history,
            memory_summary=memory_summary,
            evidence_timeline=evidence_context,
        )
        return {
            "question": question,
            "answer": answer,
            "grounded": bool(selected),
            "citations": [],
            "provider": "DeepSeek follow-up",
            "kind": "follow_up",
            "support_level": "DIRECT" if selected else "INSUFFICIENT",
            "trace": ["Follow-up retrieval: selected question-specific evidence", "Follow-up writer: prose answer"],
            "tool_trace": toolbox.trace(),
        }
    except Exception as error:
        logger.exception("DeepSeek follow-up failed")
        return deepseek_failure_result(question, error)
