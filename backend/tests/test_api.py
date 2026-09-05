from types import SimpleNamespace

import tracelens.api as main
import pytest
from fastapi.testclient import TestClient

from tracelens.api import DB_PATH, app
from tracelens.agent import AgentToolbox, is_summary_goal
from tracelens.agent_structured import run_structured_evidence_agent
from tracelens.retrieval import plan_evidence_requirements


client = TestClient(app)


@pytest.fixture(autouse=True)
def disable_external_kimi(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_ENABLED", "false")
    monkeypatch.setenv("DEEPSEEK_AGENT_WORKFLOW", "legacy")


def setup_function() -> None:
    main.init_db()
    with __import__("sqlite3").connect(DB_PATH) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS evidence ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, video_id TEXT NOT NULL, "
            "start_seconds REAL NOT NULL, end_seconds REAL NOT NULL, text TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS videos ("
            "video_id TEXT PRIMARY KEY, filename TEXT NOT NULL, stored_path TEXT NOT NULL, "
            "content_hash TEXT NOT NULL, status TEXT NOT NULL, transcript_text TEXT, "
            "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, "
            "updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        connection.execute("DELETE FROM agent_messages")
        connection.execute("DELETE FROM agent_reports")
        connection.execute("DELETE FROM agent_sessions")
        connection.execute("DELETE FROM media_tasks")
        connection.execute("DELETE FROM evidence")
        connection.execute("DELETE FROM videos")


def test_health() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_protected_mode_isolates_videos_and_revokes_logout_token(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    import uuid
    suffix = uuid.uuid4().hex[:8]
    owner = client.post("/api/auth/register", json={"username": f"owner_{suffix}", "password": "secure-pass-1"})
    other = client.post("/api/auth/register", json={"username": f"other_{suffix}", "password": "secure-pass-2"})
    assert owner.status_code == 200
    assert other.status_code == 200
    owner_headers = {"Authorization": f"Bearer {owner.json()['access_token']}"}
    other_headers = {"Authorization": f"Bearer {other.json()['access_token']}"}

    seeded = client.post("/api/demo/seed", json={"video_id": "private-auth-video"}, headers=owner_headers)
    assert seeded.status_code == 200
    assert [item["video_id"] for item in client.get("/api/videos", headers=owner_headers).json()["items"]] == [
        "private-auth-video"
    ]
    assert client.get("/api/videos", headers=other_headers).json()["items"] == []
    assert client.get("/api/evidence", params={"video_id": "private-auth-video"}, headers=other_headers).status_code == 404
    assert client.get("/api/videos").status_code == 401

    assert client.post("/api/auth/logout", headers=owner_headers).status_code == 200
    assert client.get("/api/auth/me", headers=owner_headers).status_code == 401


def test_evidence_time_hints_and_window() -> None:
    with __import__("sqlite3").connect(DB_PATH) as connection:
        connection.executemany(
            "INSERT INTO evidence(video_id, start_seconds, end_seconds, text, source) VALUES (?, ?, ?, ?, ?)",
            [
                ("timeline-demo", 58, 65, "ASR 时间点附近的说明", "ASR"),
                ("timeline-demo", 64, 70, "OCR 画面中的关键词", "OCR"),
                ("timeline-demo", 120, 130, "无关内容", "ASR"),
            ],
        )
    hints = client.get(
        "/api/evidence/time-hints",
        params={"query": "请看 01:02 以及 2分30秒和 4秒"},
    )
    assert hints.status_code == 200
    assert hints.json()["timestampsMs"] == [62000, 150000, 4000]

    search = client.get(
        "/api/evidence/search",
        params={"query": "关键词", "video_id": "timeline-demo", "sources": "OCR"},
    )
    assert search.status_code == 200
    assert search.json()["matches"][0]["source"] == "OCR"

    window = client.get(
        "/api/evidence/window",
        params={"video_id": "timeline-demo", "timestamp_ms": 62000, "before_ms": 5000, "after_ms": 5000},
    )
    assert window.status_code == 200
    assert [item["source"] for item in window.json()["segments"]] == ["ASR", "OCR"]


def test_runtime_retrieval_prioritizes_time_anchor_and_ocr() -> None:
    with __import__("sqlite3").connect(DB_PATH) as connection:
        connection.executemany(
            "INSERT INTO evidence(video_id, start_seconds, end_seconds, text, source) VALUES (?, ?, ?, ?, ?)",
            [
                ("anchor-demo", 0, 8, "开场介绍", "ASR"),
                ("anchor-demo", 165, 175, "前一个章节", "OCR"),
                ("anchor-demo", 180, 190, "第三分钟的核心观点", "OCR"),
                ("anchor-demo", 181, 184, "第三分钟讲解核心观点", "ASR"),
                ("anchor-demo", 240, 250, "结尾总结", "OCR"),
            ],
        )
    question = "3分钟左右讲了什么？"
    result = AgentToolbox("anchor-demo", question).execute(
        "search_timeline", {"query": question, "top_k": 6, "sources": ["ASR", "OCR"]}
    )
    assert result["timeHintsMs"] == [180000]
    assert result["matches"][0]["startMs"] == 180000
    assert result["matches"][0]["source"] == "OCR"
    assert any(item["source"] == "ASR" and 180000 <= item["startMs"] <= 190000
               for item in result["matches"])


def test_requirement_planner_splits_multi_part_questions() -> None:
    plan = plan_evidence_requirements("视频讲了什么？另外核心案例是什么？")
    assert plan["strategy"] == "CLAUSE_DECOMPOSITION"
    assert len(plan["requirements"]) == 2


def test_evidence_answer_contains_citation(monkeypatch) -> None:
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(main, "extract_ocr_evidence", lambda video_path, file_name: [(0.0, 15.0, "视频标题 OCR")])
    response = client.post(
        "/api/evidence",
        json={
            "video_id": "demo",
            "start_seconds": 12,
            "end_seconds": 35,
            "text": "视频介绍了三个学习方法：刻意练习、复盘和项目实践。",
        },
    )
    assert response.status_code == 200

    response = client.post(
        "/api/ask",
        json={"video_id": "demo", "question": "视频介绍了哪些学习方法？"},
    )
    body = response.json()
    assert response.status_code == 200
    assert body["grounded"] is True
    assert body["citations"][0]["timestamp"] == "00:12 - 00:35"
    monkeypatch.undo()


def test_unknown_question_refuses() -> None:
    response = client.post("/api/ask", json={"question": "视频讲了什么天气预报？"})
    assert response.status_code == 200
    assert response.json()["grounded"] is False


@pytest.mark.parametrize(
    "question",
    ["这个视频最主要在说什么？", "视频主要讲了什么？", "视频的主要内容是什么？"],
)
def test_summary_questions_are_classified_as_video_synthesis(question: str) -> None:
    assert is_summary_goal(question) is True
    assert AgentToolbox("summary-classification", question).plan()["intent"] == "STRUCTURED_SUMMARY"


@pytest.mark.parametrize(
    "question",
    [
        "这个视频主要在说什么？",
        "这段视频的主要内容是什么？",
        "请概括一下视频内容",
    ],
)
def test_real_chinese_summary_questions_use_synthesis_path(question: str) -> None:
    assert is_summary_goal(question) is True
    assert AgentToolbox("summary-chinese-classification", question).plan()["intent"] == "STRUCTURED_SUMMARY"


def test_summary_report_accepts_representative_timeline_evidence() -> None:
    video_id = "summary-report-gate"
    evidence_ids = []
    for index, text in enumerate([
        "开场说明视频讨论科技创新的意义。",
        "中段以国产大飞机的研发作为案例。",
        "结尾强调青年要通过实践承担责任。",
    ]):
        response = client.post(
            "/api/evidence",
            json={
                "video_id": video_id,
                "start_seconds": index * 60,
                "end_seconds": index * 60 + 10,
                "text": text,
            },
        )
        assert response.status_code == 200
        evidence_ids.append(response.json()["id"])

    result = AgentToolbox(video_id, "这个视频主要在说什么？").execute(
        "generate_report",
        {
            "answerable": True,
            "finalAnswer": "视频围绕科技创新展开，并通过案例和结尾号召串联主题。",
            "evidence": [{"dbEvidenceId": item} for item in evidence_ids],
            "support_level": "SYNTHESIS",
        },
    )

    assert result["accepted"] is True
    assert [item["evidenceId"] for item in result["citations"]] == [str(item) for item in evidence_ids]


def test_local_summary_fallback_samples_the_whole_timeline() -> None:
    video_id = "summary-fallback-video"
    for index in range(6):
        response = client.post(
            "/api/evidence",
            json={
                "video_id": video_id,
                "start_seconds": index * 60,
                "end_seconds": index * 60 + 10,
                "text": f"视频第 {index + 1} 个阶段介绍了一个核心观点。",
            },
        )
        assert response.status_code == 200

    response = client.post(
        "/api/ask",
        json={"video_id": video_id, "question": "这个视频最主要在说什么？"},
    )
    body = response.json()
    assert response.status_code == 200
    assert body["grounded"] is True
    assert body["trace"] == ["Fallback: summary goal uses representative timeline overview"]
    assert len(body["citations"]) == 6

def test_upload_video_creates_transcript_evidence(monkeypatch) -> None:
    monkeypatch.setattr(main, "extract_ocr_evidence", lambda video_path, file_name: [(0.0, 15.0, "ocr evidence")])
    monkeypatch.setattr(
        main,
        "extract_transcript_from_video",
        lambda video_path, file_name: [(0.0, 2.0, "视频处理流程测试")],
    )
    response = client.post(
        '/api/videos/upload',
        data={'video_id': 'uploaded-demo'},
        files={'file': ('demo-clip.mp4', b'fake mp4 bytes', 'video/mp4')},
    )
    assert response.status_code == 200
    body = response.json()
    assert body['video_id'] == 'uploaded-demo'
    assert body['stored_path'].endswith('demo-clip.mp4')
    assert body['asr_count'] == 1
    assert body['ocr_count'] == 1
    sources = [item['source'] for item in client.get('/api/evidence', params={'video_id': 'uploaded-demo'}).json()['items']]
    assert sources == ['ASR', 'OCR']

    response = client.post('/api/ask', json={
        'video_id': 'uploaded-demo',
        'question': '这个视频说明了什么处理流程？',
    })
    assert response.status_code == 200
    assert response.json()['grounded'] is True


def test_background_upload_returns_transcription_task(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "extract_transcript_from_video",
        lambda video_path, file_name: [(0.0, 2.0, "异步转录测试")],
    )
    monkeypatch.setattr(main, "extract_ocr_evidence", lambda video_path, file_name: [])
    response = client.post(
        "/api/videos/upload",
        data={"video_id": "background-demo", "background": "true"},
        files={"file": ("background.mp4", b"background bytes", "video/mp4")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    task_id = body["transcription_task_id"]

    import time
    for _ in range(500):
        task = client.get(f"/api/tasks/{task_id}").json()
        if task["state"] in {"COMPLETED", "FAILED"}:
            break
        time.sleep(0.02)
    assert task["state"] == "COMPLETED"
    assert task["result"]["asr_count"] == 1
    assert client.get("/api/evidence", params={"video_id": "background-demo"}).json()["items"]


def test_ask_rejects_video_while_transcription_is_pending(monkeypatch) -> None:
    monkeypatch.setattr(main, "extract_transcript_from_video", lambda video_path, file_name: [])
    monkeypatch.setattr(main, "extract_ocr_evidence", lambda video_path, file_name: [])
    response = client.post(
        "/api/videos/upload",
        data={"video_id": "pending-demo", "background": "true"},
        files={"file": ("pending.mp4", b"pending bytes", "video/mp4")},
    )
    assert response.status_code == 200
    task_id = response.json()["transcription_task_id"]
    response = client.post(
        "/api/ask",
        json={"video_id": "pending-demo", "question": "视频讲了什么？"},
    )
    assert response.status_code == 409
    assert "not ready" in response.json()["detail"]
    assert client.get(f"/api/tasks/{task_id}").status_code == 200


def test_upload_deduplicates_and_delete_cleans_video(monkeypatch) -> None:
    transcribe_calls = []

    def fake_transcribe(video_path, file_name):
        transcribe_calls.append(video_path)
        return [(0.0, 2.0, "deduplicated upload evidence")]

    monkeypatch.setattr(main, "extract_transcript_from_video", fake_transcribe)
    payload = {
        "video_id": "managed-video",
        "file": ("managed.mp4", b"same video bytes", "video/mp4"),
    }
    first = client.post("/api/videos/upload", data={"video_id": payload["video_id"]}, files={"file": payload["file"]})
    second = client.post("/api/videos/upload", data={"video_id": payload["video_id"]}, files={"file": payload["file"]})

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["deduplicated"] is True
    assert len(transcribe_calls) == 1
    assert client.get("/api/videos").json()["items"][0]["video_id"] == "managed-video"

    deleted = client.delete("/api/videos/managed-video")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert client.get("/api/videos").json()["items"] == []
    assert client.get("/api/evidence", params={"video_id": "managed-video"}).json()["items"] == []


def test_kimi_answer_validates_citations(monkeypatch) -> None:
    cited_evidence_id = 0

    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=f'{{"answer":"项目式学习先做项目再补齐知识。","citation_ids":[{cited_evidence_id}]}}'
                        )
                    )
                ]
            )

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_ENABLED", "true")
    monkeypatch.setattr(main, "OpenAI", FakeClient)
    response = client.post(
        "/api/evidence",
        json={
            "video_id": "kimi-test",
            "start_seconds": 0,
            "end_seconds": 10,
            "text": "项目式学习先做一个能运行的小项目，再围绕项目补齐知识。",
        },
    )
    assert response.status_code == 200
    cited_evidence_id = response.json()["id"]

    response = client.post(
        "/api/ask",
        json={"video_id": "kimi-test", "question": "项目式学习如何开始？"},
    )
    body = response.json()
    assert response.status_code == 200
    assert body["provider"] == "DeepSeek"
    assert body["citations"][0]["evidence_id"] == cited_evidence_id


def test_provider_failure_is_persisted_as_visible_report(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_ENABLED", "true")
    monkeypatch.setattr(main, "run_deepseek_agent", lambda question, video_id, history=None: {
        "question": question,
        "answer": "DeepSeek 请求失败：认证失败",
        "grounded": False,
        "provider": "DeepSeek error",
        "error": "authentication failed",
    })
    response = client.post(
        "/api/ask",
        json={"video_id": "failure-demo", "question": "首次报告"},
    )
    assert response.status_code == 200
    with __import__("sqlite3").connect(DB_PATH) as connection:
        count = connection.execute("SELECT COUNT(*) FROM agent_reports").fetchone()[0]
    assert count == 1
    reports = client.get("/api/videos/failure-demo/reports").json()["items"]
    assert len(reports) == 1
    assert reports[0]["answerable"] is False


def test_kimi_agent_selects_tools_and_submits_report(monkeypatch) -> None:
    calls = []
    requested_tools = []
    cited_evidence_id = [0]

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            step = len(calls)
            if step == 1:
                name, arguments = "get_video_metadata", "{}"
            elif step == 2:
                name, arguments = "search_timeline", '{"query":"project learning process"}'
            else:
                name, arguments = "generate_report", (
                    '{"answerable":true,"final_answer":"The project learning process includes practice.",'
                    f'"citation_ids":[{cited_evidence_id[0]}],"support_level":"DIRECT"}}'
                )
            requested_tools.append(name)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(
                        id=f"call-{step}",
                        function=SimpleNamespace(name=name, arguments=arguments),
                    )],
                ))]
            )

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_ENABLED", "true")
    monkeypatch.setattr(main, "OpenAI", FakeClient)
    response = client.post(
        "/api/evidence",
        json={
            "video_id": "agent-tool-test",
            "start_seconds": 0,
            "end_seconds": 10,
            "text": "The project learning process includes practice.",
        },
    )
    assert response.status_code == 200
    cited_evidence_id[0] = response.json()["id"]

    response = client.post(
        "/api/ask",
        json={"video_id": "agent-tool-test", "question": "What does the project learning process include?"},
    )
    body = response.json()
    assert response.status_code == 200
    assert body["provider"] == "DeepSeek"
    assert body["grounded"] is True
    assert requested_tools == [
        "get_video_metadata",
        "search_timeline",
        "generate_report",
    ]
    assert body["tool_trace"][-1]["tool"] == "generate_report"


def test_merge_transcript_chunks() -> None:
    segments = [
        (0.0, 4.0, "第一句。"),
        (4.2, 8.0, "第二句。"),
        (40.0, 44.0, "第三句。"),
    ]
    merged = main.merge_transcript_chunks(segments, max_duration=30, max_gap=2)
    assert merged == [
        (0.0, 8.0, "第一句。第二句。"),
        (40.0, 44.0, "第三句。"),
    ]

def test_seed_and_list_demo_evidence() -> None:
    response = client.post("/api/demo/seed", json={"video_id": "demo-video"})
    assert response.status_code == 200
    assert response.json()["seeded"] == 3

    response = client.get("/api/evidence", params={"video_id": "demo-video"})
    assert response.status_code == 200
    assert len(response.json()["items"]) == 3


def test_ask_includes_trace() -> None:
    response = client.post("/api/demo/seed", json={"video_id": "demo-video"})
    assert response.status_code == 200

    response = client.post(
        "/api/ask",
        json={"video_id": "demo-video", "question": "项目式学习的特点是什么？"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "trace" in body
    assert isinstance(body["trace"], list)
    assert len(body["trace"]) > 0
    assert body["grounded"] is True


def test_session_memory_is_reused_and_restored() -> None:
    client.post("/api/demo/seed", json={"video_id": "memory-video"})
    first = client.post(
        "/api/ask",
        json={"video_id": "memory-video", "question": "项目式学习的特点是什么？"},
    )
    first_body = first.json()
    session_id = first_body["session_id"]

    second = client.post(
        "/api/ask",
        json={
            "video_id": "memory-video",
            "question": "请再概括一次。",
            "session_id": session_id,
        },
    )
    assert second.status_code == 200
    assert second.json()["session_id"] == session_id

    memory = client.get("/api/videos/memory-video/memory")
    assert memory.status_code == 200
    session = memory.json()["sessions"][0]
    assert session["session_id"] == session_id
    assert len(session["messages"]) == 4
    assert len(session["reports"]) == 2


def test_follow_up_uses_prose_path_and_hides_citations(monkeypatch) -> None:
    client.post("/api/demo/seed", json={"video_id": "follow-up-video"})
    first = client.post(
        "/api/ask",
        json={"video_id": "follow-up-video", "question": "项目式学习的特点是什么？"},
    )
    session_id = first.json()["session_id"]
    called = False

    def fail_structured(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("follow-up must not run the structured workflow")

    monkeypatch.setattr(main, "run_deepseek_agent", fail_structured)
    second = client.post(
        "/api/ask",
        json={
            "video_id": "follow-up-video",
            "question": "能再解释一下吗？",
            "session_id": session_id,
        },
    )
    assert second.status_code == 200
    assert second.json()["citations"] == []
    assert called is False


def test_delete_video_removes_session_memory() -> None:
    client.post("/api/demo/seed", json={"video_id": "delete-memory-video"})
    response = client.post(
        "/api/ask",
        json={"video_id": "delete-memory-video", "question": "视频讲了什么？"},
    )
    assert response.status_code == 200
    assert client.get("/api/videos/delete-memory-video/memory").json()["sessions"]

    deleted = client.delete("/api/videos/delete-memory-video")
    assert deleted.status_code == 200
    assert client.get("/api/videos/delete-memory-video/memory").json()["sessions"] == []


def test_kimi_session_history_is_opt_in(monkeypatch) -> None:
    evidence = client.post(
        "/api/evidence",
        json={
            "video_id": "history-video",
            "start_seconds": 0,
            "end_seconds": 10,
            "text": "The video explains project learning.",
        },
    )
    assert evidence.status_code == 200
    evidence_id = evidence.json()["id"]
    requests = []

    class FakeCompletions:
        def create(self, **kwargs):
            requests.append({"messages": list(kwargs["messages"])})
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(
                        id="report-call",
                        function=SimpleNamespace(
                            name="generate_report",
                            arguments=(
                                '{"answerable":true,"final_answer":"Project learning.",'
                                f'"citation_ids":[{evidence_id}],"support_level":"DIRECT"}}'
                            ),
                        ),
                    )],
                ))]
            )

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_ENABLED", "true")
    monkeypatch.setenv("DEEPSEEK_SEND_SESSION_HISTORY", "true")
    monkeypatch.setattr(main, "OpenAI", FakeClient)
    result = main.run_kimi_agent(
        "What is project learning?",
        "history-video",
        history=[
            {"role": "user", "content": "What did the speaker introduce?"},
            {"role": "assistant", "content": "The speaker introduced project learning."},
        ],
    )

    assert result is not None
    messages = requests[0]["messages"]
    assert {"role": "user", "content": "What did the speaker introduce?"} in messages
    assert {"role": "assistant", "content": "The speaker introduced project learning."} in messages
    assert "What is project learning?" in messages[-1]["content"]


def test_structured_agent_runs_planner_retriever_verifier_writer_critic() -> None:
    response = client.post(
        "/api/evidence",
        json={
            "video_id": "structured-agent-test",
            "start_seconds": 0,
            "end_seconds": 5,
            "text": "The project uses retrieval and verification.",
        },
    )
    assert response.status_code == 200
    evidence_id = response.json()["id"]

    class FakeProvider:
        def _completion(self, messages, tools, tool_choice=None):
            import json

            phase = getattr(self, "_agent_phase", "")
            if phase == "PLANNER":
                name = "submit_evidence_plan"
                arguments = {
                    "answerMode": "SINGLE",
                    "requirements": [{
                        "requirementId": "R1",
                        "subQuestion": "What method does the project use?",
                        "retrievalQuery": "project retrieval verification",
                        "evidenceRole": "DIRECT",
                        "completionPolicy": "DIRECT",
                        "expectedSources": ["ASR"],
                        "required": True,
                    }],
                }
            elif phase == "VERIFIER":
                name = "submit_evidence_verification"
                arguments = {
                    "requirements": [{
                        "requirementId": "R1",
                        "supported": True,
                        "complete": True,
                        "supportLevel": "DIRECT",
                        "evidenceIds": ["E001"],
                        "missingInformation": "",
                        "contradictionEvidenceIds": [],
                    }],
                    "overallSufficient": True,
                    "shouldRefuse": False,
                    "refusalReason": "",
                }
            elif phase == "WRITER":
                name = "submit_grounded_report"
                arguments = {
                    "answerable": True,
                    "finalAnswer": "The project uses retrieval and verification.",
                    "title": "Project report",
                    "conclusions": ["The project uses retrieval and verification."],
                    "evidenceIds": ["E001"],
                    "suggestions": [],
                }
            else:
                name = "submit_critic_review"
                arguments = {
                    "approved": True,
                    "answerabilityCorrect": True,
                    "allRequiredSlotsCovered": True,
                    "contradictionFree": True,
                    "externalKnowledgeFree": True,
                    "citationIdsValid": True,
                    "missingRequirementIds": [],
                    "unsupportedClaims": [],
                    "contradictions": [],
                    "revisionInstruction": "",
                }
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": name,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }],
            }

    toolbox = AgentToolbox(
        "structured-agent-test",
        "What method does the project use?",
    )
    result = run_structured_evidence_agent(
        FakeProvider(),
        toolbox,
        "What method does the project use?",
    )

    assert result["accepted"] is True
    assert result["final_answer"] == "The project uses retrieval and verification."
    assert result["citations"][0]["evidence_id"] == evidence_id
    assert result["agentGraph"]["nodes"] == [
        "structured_planner",
        "prefetch_time_evidence",
        "prefetch_coverage_evidence",
        "retrieve_evidence_slots",
        "verify_evidence_ledger",
        "write_grounded_report",
        "critic_review",
        "retrieve_followup_evidence",
        "verify_followup_evidence",
        "revise_report",
    ]


def test_initial_analysis_is_a_persistent_async_task(monkeypatch) -> None:
    client.post("/api/demo/seed", json={"video_id": "async-analysis-video"})
    monkeypatch.setattr(main, "run_deepseek_agent", lambda question, video_id, history=None: {
        "question": question,
        "answer": "异步报告答案",
        "grounded": True,
        "provider": "DeepSeek",
        "kind": "initial_report",
        "report": {"finalAnswer": "异步报告答案", "evidence": [{"evidence_id": 1, "text": "证据"}]},
    })
    response = client.post("/api/analysis", json={
        "video_id": "async-analysis-video", "question": "视频讲了什么？",
    })
    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] is True
    assert body["taskId"]
    for _ in range(100):
        status = client.get(f"/api/analysis/{body['taskId']}").json()
        if status["state"] in {"COMPLETED", "FAILED"}:
            break
        import time
        time.sleep(0.02)
    assert status["state"] == "COMPLETED"
    assert status["result"]["provider"] == "DeepSeek"
    assert client.get("/api/videos/async-analysis-video/reports").json()["items"]


def test_initial_analysis_failure_is_visible_and_saved(monkeypatch) -> None:
    monkeypatch.setattr(main, "run_deepseek_agent", lambda question, video_id, history=None: {
        "error": "provider timeout", "provider": "DeepSeek error",
    })
    response = client.post("/api/analysis", json={"video_id": "failed-analysis-video", "question": "生成报告"})
    assert response.status_code == 200
    task_id = response.json()["taskId"]
    for _ in range(100):
        status = client.get(f"/api/analysis/{task_id}").json()
        if status["state"] in {"COMPLETED", "FAILED"}:
            break
        import time
        time.sleep(0.02)
    assert status["state"] == "COMPLETED"
    assert "provider timeout" in status["result"]["error"]
    reports = client.get("/api/videos/failed-analysis-video/reports").json()["items"]
    assert len(reports) == 1
    assert reports[0]["answerable"] is False


def test_legacy_local_fallback_remains_visible_in_report_list() -> None:
    session_id = main.get_or_create_session(None, "legacy-fallback-video")
    with __import__("sqlite3").connect(DB_PATH) as connection:
        connection.execute(
            "INSERT INTO agent_reports(session_id, question, answer, answerable, support_level, report_json, report_type) "
            "VALUES (?, ?, ?, 1, 'SUMMARY', ?, 'INITIAL')",
            (
                session_id,
                "旧问题",
                "DeepSeek 暂时不可用",
                '{"provider":"local evidence fallback","error":"provider timeout",'
                '"report":{"answerable":true,"evidence":[{"evidence_id":1}]}}',
            ),
        )
    reports = client.get("/api/videos/legacy-fallback-video/reports").json()["items"]
    assert len(reports) == 1
    assert reports[0]["answerable"] is True


def test_insufficient_evidence_report_is_kept_in_history() -> None:
    session_id = main.get_or_create_session(None, "insufficient-report-video")
    main.save_agent_turn(session_id, "视频中没有出现的主题", {
        "question": "视频中没有出现的主题",
        "answer": "视频未提供足够证据，无法从视频确定答案。",
        "grounded": False,
        "kind": "initial_report",
        "provider": "DeepSeek structured Agent",
        "report": {
            "answerable": False,
            "finalAnswer": "视频未提供足够证据，无法从视频确定答案。",
            "evidence": [],
        },
    })
    reports = client.get("/api/videos/insufficient-report-video/reports").json()["items"]
    assert len(reports) == 1
    assert reports[0]["answerable"] is False
    assert reports[0]["report"]["finalAnswer"].startswith("视频未提供足够证据")


def test_delete_report_removes_report_and_followups_but_keeps_video_evidence() -> None:
    video_id = "delete-report-video"
    evidence = client.post(
        "/api/evidence",
        json={
            "video_id": video_id,
            "start_seconds": 0,
            "end_seconds": 10,
            "text": "视频证据仍然保留。",
        },
    )
    assert evidence.status_code == 200
    session_id = main.get_or_create_session(None, video_id)
    main.save_agent_turn(session_id, "初始问题", {
        "answer": "证据不足，暂时无法确定。",
        "grounded": False,
        "kind": "initial_report",
        "provider": "DeepSeek structured Agent",
    })
    main.save_agent_turn(session_id, "继续追问", {
        "answer": "仍然无法确定。",
        "grounded": False,
        "kind": "follow_up",
        "provider": "DeepSeek follow-up",
    })
    reports = client.get(f"/api/videos/{video_id}/reports").json()["items"]
    assert len(reports) == 1
    report_id = reports[0]["id"]
    assert len(reports[0]["follow_ups"]) == 1

    deleted = client.delete(f"/api/reports/{report_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted_children"] == 1
    assert client.get(f"/api/videos/{video_id}/reports").json()["items"] == []
    assert client.get("/api/evidence", params={"video_id": video_id}).json()["items"]


def test_metrics_endpoint() -> None:
    response = client.get("/api/metrics")
    assert response.status_code == 200
    body = response.json()
    assert "capabilities" in body
    assert "statistics" in body
    assert "components" in body
    assert "available_tools" in body
    assert "Tool" in body["available_tools"] or isinstance(body["available_tools"], list)
