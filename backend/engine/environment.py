"""Feature-based environmental context detector (Ohr 3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from engine.environment_strategy import EnvironmentStrategy


class EnvironmentDetector:
    """Erkennt die Umgebung des Sprechers aus Hintergrund-Audio."""

    ENVIRONMENTS = [
        "home_quiet",
        "home_noisy",
        "office_quiet",
        "office_open",
        "car",
        "outdoor_street",
        "outdoor_nature",
        "public_space",
        "transit",
    ]

    def __init__(self):
        self.strategy = EnvironmentStrategy()

    def detect(self, audio_path: str) -> dict:
        import soundfile as sf

        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        mono = wav.mean(axis=1).astype(np.float32)
        return self.detect_chunk(mono, sr=int(sr))

    def detect_chunk(self, audio_chunk: np.ndarray, sr: int = 16000) -> dict:
        x = np.asarray(audio_chunk, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if x.size == 0:
            x = np.zeros(1600, dtype=np.float32)

        feats = self._features(x, sr)
        env, confidence = self._classify(feats)
        noise_level = self._noise_bucket(feats["rms"])

        result = {
            "environment": env,
            "confidence": round(confidence, 3),
            "noise_level": noise_level,
            "background_voices": bool(feats["voice_band_ratio"] > 0.35 and feats["zcr"] > 0.07),
            "keyboard_detected": bool(feats["high_band_ratio"] > 0.20 and feats["spectral_flatness"] > 0.45),
            "traffic_detected": bool(feats["low_band_ratio"] > 0.50 and feats["rolloff"] < 2500),
            "music_detected": bool(feats["spectral_peakedness"] > 0.20 and feats["voice_band_ratio"] > 0.25),
            "strategy_impact": self.strategy.recommend(env, noise_level),
        }
        return result

    @staticmethod
    def _features(x: np.ndarray, sr: int) -> dict[str, float]:
        rms = float(np.sqrt(np.mean(x**2) + 1e-9))
        zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0)) if len(x) > 1 else 0.0

        win = np.hanning(len(x)).astype(np.float32)
        spectrum = np.abs(np.fft.rfft(x * win)) + 1e-9
        freqs = np.fft.rfftfreq(len(x), 1.0 / max(sr, 1))
        power = spectrum**2
        power_sum = float(np.sum(power)) + 1e-9

        centroid = float(np.sum(freqs * power) / power_sum)
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * power) / power_sum))

        cdf = np.cumsum(power) / power_sum
        roll_idx = int(np.searchsorted(cdf, 0.85))
        rolloff = float(freqs[min(roll_idx, len(freqs) - 1)])

        low = float(np.sum(power[(freqs >= 20) & (freqs < 300)]) / power_sum)
        voice = float(np.sum(power[(freqs >= 300) & (freqs < 3400)]) / power_sum)
        high = float(np.sum(power[(freqs >= 3400) & (freqs < min(8000, sr / 2))]) / power_sum)

        geometric = float(np.exp(np.mean(np.log(spectrum))))
        arithmetic = float(np.mean(spectrum))
        flatness = geometric / max(arithmetic, 1e-9)
        peakedness = float((np.max(spectrum) - np.mean(spectrum)) / (np.max(spectrum) + 1e-9))

        return {
            "rms": rms,
            "zcr": zcr,
            "centroid": centroid,
            "bandwidth": bandwidth,
            "rolloff": rolloff,
            "low_band_ratio": low,
            "voice_band_ratio": voice,
            "high_band_ratio": high,
            "spectral_flatness": flatness,
            "spectral_peakedness": peakedness,
        }

    @staticmethod
    def _noise_bucket(rms: float) -> str:
        if rms < 0.02:
            return "low"
        if rms < 0.06:
            return "moderate"
        return "high"

    def _classify(self, f: dict[str, float]) -> tuple[str, float]:
        if f["rms"] < 0.015 and f["high_band_ratio"] < 0.15:
            return "home_quiet", 0.84
        if f["low_band_ratio"] > 0.55 and f["centroid"] < 1200:
            if f["rms"] > 0.05:
                return "car", 0.78
            return "transit", 0.72
        if f["voice_band_ratio"] > 0.45 and f["rms"] > 0.05:
            if f["high_band_ratio"] > 0.22:
                return "public_space", 0.76
            return "office_open", 0.73
        if f["rms"] > 0.06 and f["zcr"] > 0.10:
            return "outdoor_street", 0.71
        if f["rms"] < 0.03 and f["centroid"] < 1700 and f["voice_band_ratio"] < 0.3:
            return "outdoor_nature", 0.66
        if f["voice_band_ratio"] > 0.32 and f["rms"] < 0.05:
            return "office_quiet", 0.68
        if f["rms"] > 0.04:
            return "home_noisy", 0.63
        return "home_quiet", 0.6
