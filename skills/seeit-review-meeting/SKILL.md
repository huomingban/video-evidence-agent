---
name: tracelens-review-meeting
description: 基于 TraceLens 视频证据复盘会议，提取决策、待办、负责人、截止时间、风险和争议。
---

# TraceLens 会议复盘

用 `search_video_evidence` 分别检索决策、行动项和风险，再用 `get_evidence_window` 检查上下文。合并重复事项，按时间顺序保留直接证据。

输出会议概览、已确认决策、行动项、风险与未决事项。行动项必须包含事项、负责人、截止时间、状态和时间戳；未被视频明确说出的字段写“未明确”，不得猜测。
