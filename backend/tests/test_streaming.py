import io
import json
import wave

import numpy as np
from fastapi.testclient import TestClient

import api.streaming as streaming
from engine.baselines import SpeakerBaseline
from engine.diarization import StreamDiarizer
from engine.humanness_adapter import HumannessAdapter
from engine.strategy import ConversationStrategy
from engine.tts_adapter import TTSAdapter
from main import app


def _wav_chunk_bytes(seconds: float = 0.5, sr: int = 16000, freq: float = 220.0, amp: float = 0.2) -> bytes:
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float32) / sr
    signal = (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    int16 = np.clip(signal * 32767.0, -32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16.tobytes())
    return buf.getvalue()


def test_websocket_open_and_close():
    client = TestClient(app)
    with client.websocket_connect("/v1/stream"):
        pass


def test_audio_chunk_analysis_returns_response(tmp_path, monkeypatch):
    monkeypatch.setattr(streaming, "DEFAULT_BASELINE_DIR", tmp_path)
    client = TestClient(app)

    with client.websocket_connect("/v1/stream?speaker_id=test_user") as ws:
        ws.send_bytes(_wav_chunk_bytes())
        response = ws.receive_json()

    assert "timestamp" in response
    assert response["speaker"] in {"agent", "customer", "both", "silence"}


def test_agent_chunk_gets_humanness_block(tmp_path, monkeypatch):
    monkeypatch.setattr(streaming, "DEFAULT_BASELINE_DIR", tmp_path)
    client = TestClient(app)
    with client.websocket_connect("/v1/stream?speaker_id=agent_first") as ws:
        ws.send_bytes(_wav_chunk_bytes(freq=200.0))
        response = ws.receive_json()

    assert response["speaker"] in {"agent", "both"}
    assert "agent_state" in response
    assert "humanness_score" in response["agent_state"]
    assert "tts_fix" in response


def test_customer_chunk_gets_emotion_block(tmp_path, monkeypatch):
    monkeypatch.setattr(streaming, "DEFAULT_BASELINE_DIR", tmp_path)
    client = TestClient(app)
    response = {}
    with client.websocket_connect("/v1/stream?speaker_id=customer_second") as ws:
        ws.send_bytes(_wav_chunk_bytes(freq=180.0))
        _ = ws.receive_json()
        ws.send_bytes(_wav_chunk_bytes(freq=300.0))
        response = ws.receive_json()
        if response["speaker"] == "agent":
            ws.send_bytes(_wav_chunk_bytes(freq=330.0))
            response = ws.receive_json()

    assert response["speaker"] in {"customer", "both"}
    assert "customer_state" in response
    assert "recommendation" in response
    assert "tts_params" in response
    assert "z_score_stress" in response["customer_state"]


def test_tts_adapter_rules():
    adapter = TTSAdapter()

    high = adapter.recommend(stress=0.9, engagement=0.4, arousal=0.8, z_score_stress=2.5)
    medium = adapter.recommend(stress=0.5, engagement=0.5, arousal=0.5, z_score_stress=0.5)
    low_high_eng = adapter.recommend(stress=0.2, engagement=0.8, arousal=0.3, z_score_stress=0.0)
    low_eng = adapter.recommend(stress=0.2, engagement=0.2, arousal=0.4, z_score_stress=0.0)

    assert high["speed"] == 0.85
    assert medium["pause_before_next"] == 0.8
    assert low_high_eng["pause_before_next"] == 0.5
    assert low_eng["speed"] == 1.05


def test_conversation_strategy_rules():
    strategy = ConversationStrategy()

    empathy = strategy.recommend(stress=0.8, engagement=0.4, trend="escalating", z_score=1.0)
    reengage = strategy.recommend(stress=0.3, engagement=0.2, trend="stable", z_score=0.0)
    advance = strategy.recommend(stress=0.2, engagement=0.8, trend="stable", z_score=0.0)
    fallback = strategy.recommend(stress=0.5, engagement=0.5, trend="stable", z_score=0.0)

    assert empathy["strategy"] == "empathy"
    assert empathy["urgency"] == "high"
    assert reengage["strategy"] == "re-engage"
    assert advance["strategy"] == "advance"
    assert fallback["strategy"] == "continue"


def test_baseline_updates_during_stream(tmp_path, monkeypatch):
    monkeypatch.setattr(streaming, "DEFAULT_BASELINE_DIR", tmp_path)
    client = TestClient(app)
    speaker = "baseline_stream_user"

    with client.websocket_connect(f"/v1/stream?speaker_id={speaker}") as ws:
        ws.send_bytes(_wav_chunk_bytes(freq=180.0))
        _ = ws.receive_json()
        ws.send_bytes(_wav_chunk_bytes(freq=310.0))
        _ = ws.receive_json()
        ws.send_bytes(_wav_chunk_bytes(freq=330.0))
        _ = ws.receive_json()

    baseline = SpeakerBaseline.load_from_path(speaker, tmp_path)
    payload = baseline.to_dict()
    stress_state = payload.get("features", {}).get("stress", {})
    assert int(stress_state.get("count", 0)) >= 2

    raw = json.loads((tmp_path / f"{speaker}.json").read_text(encoding="utf-8"))
    assert raw["speaker_id"] == speaker


def test_humanness_adapter_ranges():
    adapter = HumannessAdapter()
    robotic = adapter.analyze(None, 25.0)
    stiff = adapter.analyze(None, 50.0)
    ok = adapter.analyze(None, 70.0)
    human = adapter.analyze(None, 90.0)

    assert robotic["agent_state"]["diagnosis"] == "robotic"
    assert robotic["tts_fix"]["micro_hesitations"] is True
    assert stiff["agent_state"]["diagnosis"] == "too_stiff"
    assert ok["agent_state"]["diagnosis"] == "acceptable"
    assert human["agent_state"]["diagnosis"] == "human"
    assert human["tts_fix"]["pitch_variation"] == 0.0


def test_stream_diarizer_detects_switch():
    diarizer = StreamDiarizer()
    sr = 16000
    t = np.arange(int(0.5 * sr), dtype=np.float32) / sr
    agent = 0.2 * np.sin(2.0 * np.pi * 180.0 * t)
    customer = 0.2 * np.sin(2.0 * np.pi * 290.0 * t)

    first = diarizer.identify(agent.astype(np.float32))
    second = diarizer.identify(customer.astype(np.float32))

    assert first == "agent"
    assert second == "customer"


def test_humanness_trend_tracking():
    assert HumannessAdapter.trend([52, 54, 57, 60, 66]) == "improving"
    assert HumannessAdapter.trend([77, 74, 70, 68, 62]) == "degrading"
    assert HumannessAdapter.trend([70, 69, 70, 71, 70]) == "stable"
