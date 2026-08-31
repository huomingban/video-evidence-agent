from fastapi.testclient import TestClient

from app.main import DB_PATH, app


client = TestClient(app)


def setup_function() -> None:
    with __import__("sqlite3").connect(DB_PATH) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS evidence ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, video_id TEXT NOT NULL, "
            "start_seconds REAL NOT NULL, end_seconds REAL NOT NULL, text TEXT NOT NULL)"
        )
        connection.execute("DELETE FROM evidence")


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

def test_upload_video_creates_transcript_evidence() -> None:
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


def test_metrics_endpoint() -> None:
    response = client.get("/api/metrics")
    assert response.status_code == 200
    body = response.json()
    assert "capabilities" in body
    assert "statistics" in body
    assert "components" in body
    assert "available_tools" in body
    assert "Tool" in body["available_tools"] or isinstance(body["available_tools"], list)
