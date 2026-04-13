from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

from engine.diarization import SpeakerDiarizer


def _tone(freq: float, sr: int, duration: float, amp: float = 0.18) -> np.ndarray:
    n = int(sr * duration)
    t = np.arange(n, dtype=np.float32) / sr
    return amp * np.sin(2 * math.pi * freq * t)


def _write_wav(path: Path, signal: np.ndarray, sr: int = 16000) -> None:
    pcm = np.clip(signal * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _build_two_speaker_fixture(path: Path) -> None:
    sr = 16000
    sig = np.concatenate(
        [
            _tone(140, sr, 1.0),           # speaker turn 1
            np.zeros(int(sr * 0.6)),       # silence
            _tone(230, sr, 1.1),           # speaker turn 2
            np.zeros(int(sr * 0.7)),       # silence
            _tone(150, sr, 0.9),           # speaker turn 3
        ]
    ).astype(np.float32)
    _write_wav(path, sig, sr=sr)


def test_silence_is_detected(tmp_path: Path):
    wav = tmp_path / "fixture.wav"
    _build_two_speaker_fixture(wav)
    diarizer = SpeakerDiarizer(energy_threshold=0.01, min_segment_duration=0.2)
    segments = diarizer.diarize(str(wav))
    assert any(s.get("speaker") == "silence" for s in segments)


def test_two_speakers_are_separated(tmp_path: Path):
    wav = tmp_path / "fixture.wav"
    _build_two_speaker_fixture(wav)
    diarizer = SpeakerDiarizer(energy_threshold=0.01, min_segment_duration=0.2)
    segments = diarizer.diarize(str(wav))
    speakers = {s.get("speaker") for s in segments if s.get("speaker") not in {"silence", "speech"}}
    assert "speaker_A" in speakers
    assert "speaker_B" in speakers


def test_label_speakers_first_is_agent(tmp_path: Path):
    wav = tmp_path / "fixture.wav"
    _build_two_speaker_fixture(wav)
    diarizer = SpeakerDiarizer(energy_threshold=0.01, min_segment_duration=0.2)
    segments = diarizer.diarize(str(wav))
    labeled = diarizer.label_speakers(segments, method="first_is_agent")
    first_speech = next(s for s in labeled if s.get("speaker") != "silence")
    assert first_speech.get("role") == "agent"
    roles = {s.get("role") for s in labeled if s.get("speaker") != "silence"}
    assert "customer" in roles


def test_segments_sorted_and_gapless(tmp_path: Path):
    wav = tmp_path / "fixture.wav"
    _build_two_speaker_fixture(wav)
    diarizer = SpeakerDiarizer(energy_threshold=0.01, min_segment_duration=0.2)
    segments = diarizer.diarize(str(wav))
    assert segments == sorted(segments, key=lambda s: float(s["start"]))
    for i in range(len(segments) - 1):
        assert abs(float(segments[i]["end"]) - float(segments[i + 1]["start"])) <= 0.06
