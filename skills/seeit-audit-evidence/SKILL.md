---
name: tracelens-audit-evidence
description: 使用 TraceLens 原始 ASR/OCR 时间轴审计报告中的结论、引用和时间戳，识别无证据、弱证据或错引内容。
---

# TraceLens 证据审计

使用 MCP 工具逐条验证报告结论，不以报告本身作为事实来源。

1. 确定 `video_id`，调用 `list_videos` 选择视频。
2. 将报告拆成可独立验证的事实性结论。
3. 对每条结论调用 `search_video_evidence`，必要时用 `get_evidence_window` 检查上下文。
4. 标记每条结论为支持、部分支持、不支持或无法判断。

只根据工具返回的证据写作。证据不足时明确说明，不得用外部知识补齐结论。
