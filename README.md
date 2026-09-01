# VideoEvidence Agent

📹 面向长视频的可追溯 AI Agent，支持证据检索、工具调用、决策链可视化。

一个生产级的视频问答系统，实现了从视频上传、自动转录、语义检索到 Agent 推理的完整闭环。

## ✨ 核心特性

- **多策略检索**：语义检索（Qdrant + SentenceTransformers）+ 关键词检索的混合方案
- **LangGraph Agent**：Retrieve → Verify → Answer 的工作流，支持条件路由和拒答保护
- **工具调用系统**：显式工具调用（semantic_search, keyword_search, verify_coverage）并追踪
- **自动转录**：FFmpeg + faster-whisper 实时将视频转换成带时间戳的证据
- **决策链可视化**：前端展示 Agent 每一步的思考过程和工具调用
- **证据引用**：每条答案都附带精确的时间戳和来源
- **无证据拒答**：当证据不足时，Agent 会主动拒绝回答而非幻觉
- **性能监控**：`/api/metrics` 端点提供系统统计和 Agent 能力展示

## 🏗️ 技术栈

| 组件 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| Agent 框架 | LangGraph |
| 向量数据库 | Qdrant (in-memory mode) |
| 文本向量化 | sentence-transformers (multilingual-MiniLM-L12-v2) |
| 视频处理 | FFmpeg |
| 语音转写 | faster-whisper (Tiny model, CPU-optimized) |
| 持久化存储 | SQLite |
| 前端框架 | Vue 3 + Vite |
| 测试框架 | pytest |

## 🚀 快速开始

### 环境要求
- Python 3.12+
- Node.js 16+
- FFmpeg（Windows: `winget install Gyan.FFmpeg`）

### 后端启动

```powershell
cd backend
pip install -r requirements.txt
Copy-Item .env.example .env
# 编辑 .env，填写 KIMI_API_KEY 和 KIMI_MODEL
python -m pytest -q              # 运行测试（14 个测试，不调用真实 Kimi）
uvicorn seeit.main:app --reload --port 9090
```

Kimi 配置保存在 `backend/.env`，后端启动时会自动读取，不需要每次重新输入。`.env` 已加入 Git 忽略规则，不能提交真实 API Key。默认使用 Kimi 的 OpenAI 兼容地址 `https://api.moonshot.cn/v1`；`KIMI_MODEL` 请填写你的 Kimi 账户当前可用的模型名称。

程序默认不读取系统 HTTP 代理（`KIMI_TRUST_ENV=false`），适合本机存在失效代理配置的情况。如果你的网络必须经过代理，请在 `.env` 中设置 `KIMI_TRUST_ENV=true`，或填写 `KIMI_PROXY=http://代理地址:端口`。

未填写 `KIMI_API_KEY` 时，系统仍可使用本地检索和模板回退回答；填写后 `/api/ask` 会调用 Kimi，并校验返回的引用只能来自检索到的证据。

上传视频时，系统会先用 FFmpeg 抽取音频，再由 faster-whisper 转写。视频没有可识别语音、文件损坏或转写失败时，接口会返回明确错误，不会写入伪造证据。

打开 http://127.0.0.1:9090/docs 查看 API 文档。

### 前端启动

```powershell
cd frontend
npm install
npm run dev
```

打开 http://127.0.0.1:5173 使用工作台。

## 📋 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/evidence` | GET/POST | 证据列表和创建 |
| `/api/ask` | POST | 提问（执行 Agent 工作流） |
| `/api/videos/upload` | POST | 上传视频（自动转录） |
| `/api/videos` | GET | 查看已保存的视频资源 |
| `/api/videos/{video_id}` | DELETE | 删除视频、证据和检索索引 |
| `/api/videos/{video_id}/memory` | GET | 查看该视频的会话、消息和历史报告 |
| `/api/demo/seed` | POST | 加载演示数据 |
| `/api/metrics` | GET | 系统指标和 Agent 能力 |

## 🧪 测试

```powershell
cd backend
pytest -q                        # 14 个测试通过，不调用真实 Kimi
pytest -v                        # 详细输出
pytest tests/test_api.py::test_ask_includes_trace -v  # 单个测试
```

**测试覆盖：**
- ✅ 健康检查
- ✅ 证据管理（CRUD）
- ✅ 无证据拒答
- ✅ 视频上传和转录
- ✅ Agent 决策链追踪
- ✅ 会话复用、持久化和恢复
- ✅ 视频删除时清理会话记忆
- ✅ 系统指标

## 💡 使用示例

### 1. 加载演示数据

```bash
curl -X POST http://127.0.0.1:9090/api/demo/seed \
  -H "Content-Type: application/json" \
  -d '{"video_id": "demo-video"}'
```

### 2. 提问（触发 Agent 工作流）

```bash
curl -X POST http://127.0.0.1:9090/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "项目式学习的特点是什么？",
    "video_id": "demo-video"
  }'
```

**响应示例：**

```json
{
  "question": "项目式学习的特点是什么？",
  "answer": "根据检索到的视频证据，相关内容包括：视频先介绍了项目式学习：先做一个能运行的小项目，再围绕项目补齐知识。",
  "grounded": true,
  "citations": [
    {
      "evidence_id": 1,
      "timestamp": "00:00 - 00:42",
      "text": "视频先介绍了项目式学习：先做一个能运行的小项目，再围绕项目补齐知识。"
    }
  ],
  "trace": [
    "Retrieve: Found 3 evidence pieces",
    "Tool: semantic_search returned 3 results",
    "Verify: Coverage: 100.0% token overlap (adequate=true)",
    "Answer: Generated grounded=true"
  ]
}
```

### 3. 上传视频

```bash
curl -X POST http://127.0.0.1:9090/api/videos/upload \
  -F "video_id=my-video" \
  -F "file=@video.mp4"
```

### 4. 查看系统指标

```bash
curl http://127.0.0.1:9090/api/metrics | jq
```

## 🎯 Agent 工作流

```
用户提问
  ↓
[Retrieve Node]
  ├─ Tool: semantic_search (Qdrant 向量相似度)
  ├─ Tool: keyword_search (关键词匹配)
  └─ 返回 top-3 证据
  ↓
[Verify Node]
  ├─ Tool: verify_coverage (检查证据覆盖度)
  ├─ IF 证据不足 → END (拒答)
  └─ IF 证据充分 → 继续
  ↓
[Answer Node]
  ├─ 基于证据生成答案
  ├─ 附加时间戳引用
  └─ 返回带 trace 的结果
  ↓
返回给用户（包含决策链可视化）
```

## 与参考项目的对应关系

当前项目是参考项目的轻量本地版，概念对应如下：

| 参考项目概念 | 当前项目实现 | 说明 |
|---|---|---|
| `Media` | `videos` SQLite 表 + `data/uploads/` | 保存视频资源、哈希、文件路径和转录状态 |
| `EvidenceSegment` | `evidence` SQLite 表 | 保存带开始/结束时间的转录证据 |
| `AgentToolbox` | `backend/seeit/agent.py` 中的 `AgentToolbox` | 提供元数据、时间轴检索、证据窗口和报告工具 |
| `agent_graph` | `backend/seeit/agent.py` + `agent_graph.py` | Kimi 可选择工具；未配置 Kimi 时使用 LangGraph 固定流程 |
| `generate_report` | Agent 工具的结构化报告提交 | 后端校验引用 ID 和证据覆盖后才接受 |
| Qdrant 检索 | `backend/seeit/qdrant_retrieval.py` + `retrieval.py` | 向量不可用时自动降级为关键词检索 |
| Agent 会话记忆 | 已启用 | SQLite 持久化会话、消息、结构化报告和工具轨迹；最近消息可按配置注入 Kimi |
| MySQL / Redis / RocketMQ | 暂未引入 | 适合多用户、异步任务和生产部署，当前本地版不依赖 |

学习参考项目时，建议优先关注这条数据流：

```text
问题 -> 模型选择工具 -> 后端执行工具 -> 工具结果回传模型
     -> 模型提交报告 -> 后端校验引用 -> 返回答案
```

当前项目已经实现参考项目中的核心 Agent 数据流，并继续补齐工程化能力：会话持久化、报告留档、工具轨迹和资源生命周期管理。SQLite 适合当前单机开发和简历演示；如果后续部署多实例，再将会话与任务存储迁移到 MySQL/Redis 等服务。

### 会话记忆

每次 /api/ask 都会创建或复用 session_id。系统会把用户问题、Agent 回答、结构化报告和工具调用轨迹写入 data/agent.sqlite3，重启后仍可通过 /api/videos/{video_id}/memory 恢复。前端会自动携带当前会话 ID，并展示该视频最近一次会话。

由于历史消息发送给 Kimi 属于数据外发，配置项 KIMI_SEND_SESSION_HISTORY 控制是否将最近 12 条消息加入模型上下文。backend/.env.example 默认关闭；将它设为 true 后才会把历史消息发送给 Kimi，设为 false 时历史仍保存在本地，但仅用于页面恢复和审计。

## 📊 项目统计

- **代码组织**：`backend/seeit/` 按 Agent、检索、媒体、存储和 API 职责拆分
- **测试覆盖**：14 个回归测试，100% 通过
- **依赖库数**：10+ 主要库（FastAPI、LangGraph、Qdrant 等）
- **前端组件**：Vue 3 单页应用，实时交互式 UI

## 🔍 目录结构

```
video-evidence-agent/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py              # 兼容入口，实际应用在 seeit/
│   ├── seeit/
│   │   ├── main.py              # ASGI 入口
│   │   ├── api.py               # FastAPI 路由和应用组装
│   │   ├── agent.py             # AgentToolbox、Kimi 和引用校验
│   │   ├── agent_graph.py       # LangGraph fallback 工作流
│   │   ├── retrieval.py         # 关键词、向量和 Qdrant 检索
│   │   ├── embedding_retrieval.py # Embedding 后端边界
│   │   ├── qdrant_retrieval.py  # Qdrant 存储边界
│   │   ├── runtime_retrieval.py # 运行时检索策略选择
│   │   ├── ocr_runner.py        # FFmpeg + faster-whisper 转写
│   │   ├── storage.py           # SQLite 和会话持久化
│   │   ├── media.py             # 视频资源校验
│   │   ├── config.py            # 环境变量和运行时路径
│   │   └── models.py            # 请求模型和领域模型
│   ├── tests/
│   │   └── test_api.py          # 14 个 pytest 测试
│   ├── requirements.txt
│   └── pytest.ini
├── frontend/
│   ├── src/
│   │   ├── App.vue              # 主工作台（Trace 可视化）
│   │   ├── api.js               # API 客户端
│   │   └── style.css            # Agent trace 样式
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── data/                         # 运行时数据（git ignore）
│   ├── agent.sqlite3            # 证据存储
│   ├── uploads/                 # 上传的视频
│   └── qdrant/                  # 向量数据库
├── PROJECT_CONTEXT.md           # 详细项目背景
└── README.md
```

## 🛠️ 开发指南

### 添加新工具

在 `backend/seeit/agent.py` 的 `AgentToolbox` 中定义新工具：

```python
AVAILABLE_TOOLS = {
    "my_tool": {
        "description": "Tool description",
        "params": ["param1", "param2"],
    },
}

def my_tool_function(param1: str, param2: str) -> dict[str, Any]:
    # 实现逻辑
    return {"result": "..."}
```

然后在 Kimi 工具循环或 `backend/seeit/agent_graph.py` 的节点中调用它。

### 自定义 Agent 节点

编辑 LangGraph 工作流（`build_agent_graph()` 函数）：

```python
def custom_node(state: AgentState) -> dict[str, Any]:
    # 你的逻辑
    return {"trace": [...]}

def build_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("custom", custom_node)
    workflow.add_edge("previous_node", "custom")
    # ...
    return workflow.compile()
```

## 📈 性能指标

系统在本机测试环境中的表现：

| 指标 | 数值 |
|------|------|
| 文本检索延迟 | <100ms |
| 向量相似度计算 | <50ms |
| 完整 Agent 循环 | ~200ms |
| 视频转录速度 | ~1x（实时速度） |
| 单次上传处理 | ~1-3s（含转录） |

## 🎓 适合作为面试项目的理由

1. **完整的系统设计**：从数据导入到答案输出的全流程
2. **主流技术栈**：FastAPI、LangGraph、Qdrant 都是生产级别的工具
3. **可解释性强**：Agent Trace 展示了完整的推理过程，易于讨论
4. **工程素质**：有自动化测试、指标监控、错误处理
5. **扩展性强**：易于添加新工具、新节点、新模型
6. **面试话题丰富**：检索、排序、Agent 设计、文本处理、视频处理等

## 🚀 下一步计划

- [x] 多轮对话支持（SQLite 会话历史管理）
- [ ] 检索结果重排序层（使用小模型）
- [ ] Qdrant 持久化存储配置
- [ ] Docker 完整部署镜像
- [ ] 自动评估框架（NDCG、F1 等指标）
- [ ] 支持多种 LLM（OpenAI、本地模型等）
- [ ] 前端实时流式输出
- [ ] 证据高亮和视频时间轴跳转

## 📝 许可证

MIT

## 👤 作者

项目式学习 - 面试准备项目

---

**最后更新**：2026-08-31  
**Agent 版本**：0.3.0  
**测试状态**：✅ 7/7 通过
