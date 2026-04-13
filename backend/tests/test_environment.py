import io
import wave

import numpy as np

from engine.environment import EnvironmentDetector
from engine.environment_strategy import EnvironmentStrategy
from engine.strategy import ConversationStrategy


def _wav_bytes(signal: np.ndarray, sr: int = 16000) -> bytes:
    pcm = np.clip(signal * 32767.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def test_silence_detected_as_home_quiet():
    detector = EnvironmentDetector()
    silence = np.zeros(int(2.0 * 16000), dtype=np.float32)
    out = detector.detect_chunk(silence, sr=16000)
    assert out["environment"] == "home_quiet"
    assert out["noise_level"] == "low"


def test_environment_strategy_mapping():
    strategy = EnvironmentStrategy()
    office = strategy.recommend("office_open", "moderate")
    car = strategy.recommend("car", "high")

    assert office["avoid_personal_questions"] is True
    assert office["prefer_yes_no_questions"] is True
    assert car["offer_callback"] is True
    assert car["use_short_sentences"] is True


def test_environment_combined_with_emotion_strategy():
    conv = ConversationStrategy()
    car = conv.recommend(stress=0.9, engagement=0.4, trend="escalating", z_score=2.0, environment="car")
    office = conv.recommend(stress=0.9, engagement=0.3, trend="stable", z_score=1.0, environment="office_open")
    public = conv.recommend(stress=0.4, engagement=0.2, trend="stable", z_score=0.2, environment="public_space")

    assert car["strategy"] == "safety_callback"
    assert office["strategy"] == "privacy_timing_check"
    assert public["strategy"] == "contextual_reengage"


def test_detect_chunk_returns_valid_schema():
    detector = EnvironmentDetector()
    sr = 16000
    t = np.arange(int(2.0 * sr), dtype=np.float32) / sr
    noisy_tone = 0.03 * np.sin(2 * np.pi * 220 * t) + 0.02 * np.random.randn(t.shape[0]).astype(np.float32)
    out = detector.detect_chunk(noisy_tone.astype(np.float32), sr=sr)

    assert "environment" in out
    assert out["environment"] in EnvironmentDetector.ENVIRONMENTS
    assert 0.0 <= out["confidence"] <= 1.0
    assert out["noise_level"] in {"low", "moderate", "high"}
    assert isinstance(out["strategy_impact"], dict)
