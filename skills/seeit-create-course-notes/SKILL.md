---
name: tracelens-create-course-notes
description: 基于 TraceLens 视频证据生成带时间戳的课程笔记、章节摘要、概念解释和复习题。
---

# TraceLens 课程笔记

使用 `list_videos`、`search_video_evidence` 和 `get_evidence_window` 检索课程主题、概念、示例和总结；对不同时间段的命中补齐上下文。证据不足时使用“视频中未明确说明”。

输出顺序：课程概览、章节笔记、核心概念、示例或易错点、复习清单、复习题。每个事实性结论至少附 `[MM:SS]` 时间戳，并保留 ASR/OCR 来源。不要补写视频未明确讲述的定义或答案。
