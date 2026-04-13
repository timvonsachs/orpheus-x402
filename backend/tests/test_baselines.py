from pathlib import Path

from engine.baselines import SpeakerBaseline


def test_ewma_convergence():
    baseline = SpeakerBaseline("speaker-a", alpha=0.3, warmup_segments=5)
    for _ in range(20):
        baseline.update("stress", 0.8)
    payload = baseline.to_dict()
    ewma = payload["features"]["stress"]["ewma"]
    assert abs(ewma - 0.8) < 0.02


def test_z_score_calculation():
    baseline = SpeakerBaseline("speaker-b", alpha=0.3, warmup_segments=5)
    for v in [0.50, 0.52, 0.48, 0.51, 0.49, 0.50, 0.52]:
        baseline.update("engagement", v)
    z = baseline.z_score("engagement", 0.70)
    assert z > 2.0


def test_warmup_phase():
    baseline = SpeakerBaseline("speaker-c", alpha=0.3, warmup_segments=5)
    for v in [100.0, 102.0, 101.0]:
        baseline.update("f0_mean", v)
    assert baseline.is_warm("f0_mean") is False
    assert baseline.z_score("f0_mean", 130.0) == 0.0

    for v in [99.0, 101.0]:
        baseline.update("f0_mean", v)
    assert baseline.is_warm("f0_mean") is True


def test_serialization_roundtrip(tmp_path: Path):
    baseline = SpeakerBaseline("speaker-d", alpha=0.4, warmup_segments=4)
    for v in [3.2, 3.1, 3.3, 3.0, 3.4]:
        baseline.update("speech_rate", v)

    baseline.save_to_path(tmp_path)
    loaded = SpeakerBaseline.load_from_path("speaker-d", tmp_path, alpha=0.4, warmup_segments=4)

    assert loaded.speaker_id == "speaker-d"
    assert loaded.is_warm("speech_rate") is True
    orig = baseline.to_dict()["features"]["speech_rate"]
    restored = loaded.to_dict()["features"]["speech_rate"]
    assert abs(orig["ewma"] - restored["ewma"]) < 1e-9
    assert abs(orig["variance"] - restored["variance"]) < 1e-9
    assert orig["count"] == restored["count"]
