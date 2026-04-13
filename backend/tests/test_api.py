from fastapi.testclient import TestClient

import api.routes as routes
from main import app


class _DummyCombiner:
    async def analyze(
        self,
        audio_path: str,
        mode: str,
        include_segments: bool,
        segment_duration: float,
        diarize: bool = False,
        transcribe: bool = False,
        detect_environment: bool = True,
    ):
        return {
            "orpheus_version": "0.1.0",
            "audio_duration_seconds": 4.2,
            "humanness": {"score": 66.6, "classification": "human", "confidence": 0.8},
            "environment": {"environment": "office_open", "confidence": 0.81},
        }


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_sense_with_mocked_combiner(monkeypatch):
    monkeypatch.setattr(routes, "get_combiner", lambda: _DummyCombiner())
    client = TestClient(app)

    response = client.post(
        "/v1/sense?mode=full&segments=true&segment_duration=5",
        files={"file": ("test.wav", b"fake-audio-content", "audio/wav")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["orpheus_version"] == "0.1.0"
    assert "processing_time_ms" in body
    assert body["humanness"]["classification"] == "human"
