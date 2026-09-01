from types import SimpleNamespace

import seeit.api as main
import pytest
from fastapi.testclient import TestClient

from seeit.api import DB_PATH, app


client = TestClient(app)


@pytest.fixture(autouse=True)
def disable_external_kimi(monkeypatch) -> None:
    monkeypatch.setenv("KIMI_ENABLED", "false")


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
        connection.execute("DELETE FROM evidence")
        connection.execute("DELETE FROM videos")


def test_health() -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_evidence_answer_contains_citation() -> None:
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


def test_unknown_question_refuses() -> None:
    response = client.post("/api/ask", json={"question": "视频讲了什么天气预报？"})
    assert response.status_code == 200
    assert response.json()["grounded"] is False

def test_upload_video_creates_transcript_evidence(monkeypatch) -> None:
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

    response = client.post('/api/ask', json={
        'video_id': 'uploaded-demo',
        'question': '这个视频说明了什么处理流程？',
    })
    assert response.status_code == 200
    assert response.json()['grounded'] is True


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

    monkeypatch.setenv("KIMI_API_KEY", "test-key")
    monkeypatch.setenv("KIMI_ENABLED", "true")
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
    assert body["provider"] == "Kimi"
    assert body["citations"][0]["evidence_id"] == cited_evidence_id


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

    monkeypatch.setenv("KIMI_API_KEY", "test-key")
    monkeypatch.setenv("KIMI_ENABLED", "true")
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
    assert body["provider"] == "Kimi"
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

    monkeypatch.setenv("KIMI_API_KEY", "test-key")
    monkeypatch.setenv("KIMI_ENABLED", "true")
    monkeypatch.setenv("KIMI_SEND_SESSION_HISTORY", "true")
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
    assert messages[1:3] == [
        {"role": "user", "content": "What did the speaker introduce?"},
        {"role": "assistant", "content": "The speaker introduced project learning."},
    ]
    assert "What is project learning?" in messages[-1]["content"]


def test_metrics_endpoint() -> None:
    response = client.get("/api/metrics")
    assert response.status_code == 200
    body = response.json()
    assert "capabilities" in body
    assert "statistics" in body
    assert "components" in body
    assert "available_tools" in body
    assert "Tool" in body["available_tools"] or isinstance(body["available_tools"], list)
