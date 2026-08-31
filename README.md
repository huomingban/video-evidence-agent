# VideoEvidence Agent

面向长视频的可追溯证据问答 Agent。首版先验证核心链路：证据导入、关键词检索、带时间戳引用和无证据拒答；后续接入 Whisper、向量检索和 LangGraph。

## 当前版本

- Python + FastAPI
- SQLite 持久化视频证据
- 证据检索与时间戳引用
- 视频上传接口（本地演示版）
- 没有足够证据时拒答
- 自动化 API 测试

## 启动

```powershell
cd C:\Mianshi\video-evidence-agent\backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 9090
```

打开 `http://127.0.0.1:9090/docs` 查看接口文档。

## 手动演示

先调用 `POST /api/evidence` 写入一条带时间戳的证据，再调用 `POST /api/ask` 提问。回答会返回 `grounded` 和 `citations`，后续前端会把 citation 做成可点击的视频时间轴。

如果要测试本地上传流程，可直接上传一个视频文件到 `POST /api/videos/upload`，服务会把其保存到 `data/uploads/<video_id>/` 并写入一组模拟转录证据。当前版本仍是“本地演示版”，后续会接入 FFmpeg 和 faster-whisper 真正转录。

## 测试

```powershell
cd C:\Mianshi\video-evidence-agent\backend
pytest -q
```

## 演进路线

1. FFmpeg + faster-whisper：视频转写为时间轴证据。
2. BGE embedding + Qdrant：从关键词检索升级为混合检索。
3. LangGraph：加入 Planner、Retriever、Verifier、Writer 节点。
4. Vue 3 工作台：视频播放器、证据面板、Agent Trace 和追问会话。
5. Docker 部署与离线评测：Recall@K、MRR、拒答率和延迟。
