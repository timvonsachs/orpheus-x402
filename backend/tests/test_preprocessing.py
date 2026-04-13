from __future__ import annotations

import shutil
import subprocess
import wave
from pathlib import Path

from engine.preprocessing import AudioPreprocessor


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _make_wav(path: Path, sample_rate: int, channels: int, duration_seconds: float = 0.2) -> None:
    n_frames = int(sample_rate * duration_seconds)
    sampwidth = 2
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        frame = (0).to_bytes(2, byteorder="little", signed=True)
        wf.writeframes(frame * n_frames * channels)


def test_wav_already_16k_mono_no_reprocess(tmp_path: Path):
    src = tmp_path / "mono16.wav"
    _make_wav(src, sample_rate=16000, channels=1)
    pre = AudioPreprocessor()
    out = pre.process(str(src))
    assert out == str(src)


def test_stereo_becomes_mono(tmp_path: Path):
    if not _ffmpeg_available():
        return
    src = tmp_path / "stereo16.wav"
    _make_wav(src, sample_rate=16000, channels=2)

    pre = AudioPreprocessor()
    out = Path(pre.process(str(src)))
    meta = pre.validate(str(out))
    assert meta["channels"] == 1


def test_44100_becomes_16000(tmp_path: Path):
    if not _ffmpeg_available():
        return
    src = tmp_path / "mono44k.wav"
    _make_wav(src, sample_rate=44100, channels=1)

    pre = AudioPreprocessor()
    out = Path(pre.process(str(src)))
    meta = pre.validate(str(out))
    assert meta["sample_rate"] == 16000


def test_validate_returns_expected_fields(tmp_path: Path):
    src = tmp_path / "validate.wav"
    _make_wav(src, sample_rate=16000, channels=1)
    pre = AudioPreprocessor()
    meta = pre.validate(str(src))
    for key in ["sample_rate", "channels", "duration_seconds", "format", "needs_conversion"]:
        assert key in meta
    assert meta["sample_rate"] == 16000
    assert meta["channels"] == 1
