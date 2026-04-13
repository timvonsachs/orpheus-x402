import numpy as np

from engine.keyword_emotion import KeywordEmotionAnalyzer
from engine.transcription import Transcriber


class _DummyWord:
    def __init__(self, word: str, start: float, end: float):
        self.word = word
        self.start = start
        self.end = end


class _DummySegment:
    def __init__(self, start: float, end: float, text: str):
        self.start = start
        self.end = end
        self.text = text
        self.words = [_DummyWord(w, start, min(end, start + 0.2)) for w in text.split()]


class _DummyWhisper:
    def transcribe(self, path: str, word_timestamps: bool = True, vad_filter: bool = True):
        return iter([_DummySegment(0.0, 1.4, "hello world"), _DummySegment(1.5, 2.2, "stress spike")]), {}


def test_whisper_load_and_transcribe(monkeypatch, tmp_path):
    monkeypatch.setattr(Transcriber, "_load_model", lambda self: setattr(self, "_model", _DummyWhisper()))
    t = Transcriber(model_size="base")
    wav = tmp_path / "a.wav"
    import soundfile as sf

    sf.write(str(wav), np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16")
    out = t.transcribe(str(wav))
    assert len(out) >= 1
    assert "text" in out[0]


def test_timestamps_are_chronological(monkeypatch, tmp_path):
    monkeypatch.setattr(Transcriber, "_load_model", lambda self: setattr(self, "_model", _DummyWhisper()))
    t = Transcriber(model_size="base")
    wav = tmp_path / "b.wav"
    import soundfile as sf

    sf.write(str(wav), np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16")
    out = t.transcribe(str(wav))
    starts = [seg["start"] for seg in out]
    assert starts == sorted(starts)


def test_keyword_emotion_finds_trigger_phrases():
    analyzer = KeywordEmotionAnalyzer()
    segs = [
        {"start": 10.0, "text": "ok", "paralinguistic_summary": {"stress_level": 0.40}},
        {"start": 14.0, "text": "three weeks are you serious", "paralinguistic_summary": {"stress_level": 0.92}},
        {"start": 18.0, "text": "i understand let me help", "paralinguistic_summary": {"stress_level": 0.62}},
    ]
    out = analyzer.analyze(segs)
    assert len(out["trigger_phrases"]) >= 1
    assert len(out["calming_phrases"]) >= 1


def test_streaming_buffer_uses_two_second_window(monkeypatch):
    monkeypatch.setattr(Transcriber, "_load_model", lambda self: setattr(self, "_model", _DummyWhisper()))
    t = Transcriber(model_size="base")
    sr = 16000
    chunk = np.zeros(int(0.5 * sr), dtype=np.float32)
    out1 = t.transcribe_chunk(chunk, sr=sr)
    out2 = t.transcribe_chunk(chunk, sr=sr)
    out3 = t.transcribe_chunk(chunk, sr=sr)
    out4 = t.transcribe_chunk(chunk, sr=sr)
    assert out1["ready"] is False
    assert out2["ready"] is False
    assert out3["ready"] is False
    assert out4["ready"] is True
