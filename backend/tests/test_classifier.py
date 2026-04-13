from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf

from benchmarks.train_emotion_classifier import extract_egemaps_features
import engine.paralinguistic as pl
from engine.paralinguistic import ParalinguisticInterpreter


def _tmp_wav(seconds: float = 1.0, sr: int = 16000) -> Path:
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float32) / sr
    x = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    fd, name = tempfile.mkstemp(suffix=".wav")
    Path(name).unlink(missing_ok=True)
    path = Path(name)
    sf.write(str(path), x, sr, subtype="PCM_16")
    return path


def test_feature_extraction_returns_88_features():
    path = _tmp_wav()
    try:
        feats = extract_egemaps_features(path)
        assert len(feats) == 88
    finally:
        path.unlink(missing_ok=True)


def test_model_loads_and_prediction_available_if_model_exists():
    if not (pl.MODEL_PATH.exists() and pl.SCALER_PATH.exists() and pl.META_PATH.exists()):
        return
    pl._load_ml_assets.cache_clear()
    interp = ParalinguisticInterpreter()
    path = _tmp_wav()
    try:
        feats = extract_egemaps_features(path)
        out = interp.predict_emotion_ml(feats)
        assert out["available"] is True
        assert out["prediction"] in {"high_arousal", "low_arousal"}
    finally:
        path.unlink(missing_ok=True)


def test_prediction_probability_between_zero_and_one():
    interp = ParalinguisticInterpreter()
    path = _tmp_wav()
    try:
        feats = extract_egemaps_features(path)
        out = interp.predict_emotion_ml(feats)
        assert 0.0 <= float(out["probability"]) <= 1.0
    finally:
        path.unlink(missing_ok=True)


def test_fallback_when_model_missing(monkeypatch):
    monkeypatch.setattr(pl, "MODEL_PATH", Path("/tmp/does_not_exist_model.joblib"))
    monkeypatch.setattr(pl, "SCALER_PATH", Path("/tmp/does_not_exist_scaler.joblib"))
    monkeypatch.setattr(pl, "META_PATH", Path("/tmp/does_not_exist_meta.json"))
    pl._load_ml_assets.cache_clear()
    interp = ParalinguisticInterpreter()
    out = interp.predict_emotion_ml({"f0_mean": 150.0, "speech_rate": 3.0, "loudness_rms": 0.05})
    assert out["available"] is False
    assert out["source"] == "rule_based_fallback"
