"""Audio preprocessing for robust downstream analysis."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional


class AudioPreprocessor:
    """Konvertiert Audio in 16kHz Mono WAV."""

    TARGET_SAMPLE_RATE = 16000
    TARGET_CHANNELS = 1
    TARGET_FORMAT = "wav"

    def process(self, input_path: str, output_path: Optional[str] = None) -> str:
        in_path = Path(input_path)
        if not in_path.exists():
            raise FileNotFoundError(f"Input file not found: {in_path}")

        meta = self.validate(str(in_path))
        if not meta.get("needs_conversion", True):
            return str(in_path)

        self._ensure_ffmpeg()

        if output_path is None:
            out_path = in_path.with_name(f"{in_path.stem}_16k.wav")
        else:
            out_path = Path(output_path)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(in_path),
            "-ar",
            str(self.TARGET_SAMPLE_RATE),
            "-ac",
            str(self.TARGET_CHANNELS),
            "-f",
            self.TARGET_FORMAT,
            str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not out_path.exists():
            msg = proc.stderr.strip() or "unknown ffmpeg error"
            raise RuntimeError(f"ffmpeg conversion failed for {in_path.name}: {msg}")
        return str(out_path)

    def process_temp(self, input_path: str) -> str:
        """Creates converted temp file if needed, else returns original path."""
        meta = self.validate(input_path)
        if not meta.get("needs_conversion", True):
            return input_path

        suffix = ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
        return self.process(input_path=input_path, output_path=tmp_path)

    def validate(self, audio_path: str) -> dict[str, Any]:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        info = self._probe_with_ffprobe(path) or self._probe_with_soundfile(path)
        if info is None:
            raise RuntimeError(f"Unable to inspect audio metadata for: {path}")

        sample_rate = int(info.get("sample_rate", 0) or 0)
        channels = int(info.get("channels", 0) or 0)
        fmt = str(info.get("format", path.suffix.lstrip(".").lower() or "unknown")).lower()
        duration = float(info.get("duration_seconds", 0.0) or 0.0)

        needs_conversion = (
            sample_rate != self.TARGET_SAMPLE_RATE
            or channels != self.TARGET_CHANNELS
            or "wav" not in fmt
        )

        return {
            "sample_rate": sample_rate,
            "channels": channels,
            "duration_seconds": round(duration, 3),
            "format": fmt,
            "needs_conversion": bool(needs_conversion),
        }

    @staticmethod
    def _ensure_ffmpeg() -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg ist nicht installiert oder nicht im PATH. "
                "Bitte installiere ffmpeg (z.B. `brew install ffmpeg`)."
            )

    @staticmethod
    def _probe_with_ffprobe(path: Path) -> Optional[dict[str, Any]]:
        if shutil.which("ffprobe") is None:
            return None

        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=sample_rate,channels",
            "-show_entries",
            "format=duration,format_name",
            "-of",
            "json",
            str(path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            return None
        try:
            payload = json.loads(proc.stdout)
            streams = payload.get("streams", []) or []
            fmt = payload.get("format", {}) or {}
            stream0 = streams[0] if streams else {}
            return {
                "sample_rate": int(stream0.get("sample_rate", 0) or 0),
                "channels": int(stream0.get("channels", 0) or 0),
                "duration_seconds": float(fmt.get("duration", 0.0) or 0.0),
                "format": str(fmt.get("format_name", "")),
            }
        except Exception:
            return None

    @staticmethod
    def _probe_with_soundfile(path: Path) -> Optional[dict[str, Any]]:
        try:
            import soundfile as sf

            info = sf.info(str(path))
            return {
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "duration_seconds": float(info.duration),
                "format": str(info.format or path.suffix.lstrip(".")),
            }
        except Exception:
            return None
