from engine.biomarkers import BiomarkerEngine


def test_guess_mime_type():
    assert BiomarkerEngine._guess_mime_type(".wav") == "audio/wav"
    assert BiomarkerEngine._guess_mime_type(".mp3") == "audio/mpeg"
    assert BiomarkerEngine._guess_mime_type(".flac") == "audio/flac"
    assert BiomarkerEngine._guess_mime_type(".unknown") == "application/octet-stream"


def test_engine_init_values():
    engine = BiomarkerEngine("http://localhost:8000/api/v1/analyze", timeout_seconds=42.0)
    assert engine.api_url.endswith("/analyze")
    assert engine.timeout_seconds == 42.0
