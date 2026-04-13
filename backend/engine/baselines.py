"""EWMA-basierte Speaker-Baselines fuer akustische Features."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


SUPPORTED_FEATURES = {
    "stress",
    "engagement",
    "f0_mean",
    "jitter",
    "speech_rate",
    "pause_rate",
}


@dataclass
class _FeatureState:
    ewma: float
    variance: float
    count: int

    def to_dict(self) -> dict[str, float | int]:
        return {"ewma": self.ewma, "variance": self.variance, "count": self.count}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "_FeatureState":
        return cls(
            ewma=float(payload.get("ewma", 0.0)),
            variance=float(payload.get("variance", 0.0)),
            count=int(payload.get("count", 0)),
        )


class SpeakerBaseline:
    def __init__(self, speaker_id: str, alpha: float = 0.3, warmup_segments: int = 5):
        self.speaker_id = speaker_id
        self.alpha = alpha
        self.warmup_segments = warmup_segments
        self._states: dict[str, _FeatureState] = {}

    def update(self, feature_name: str, value: float) -> None:
        if feature_name not in SUPPORTED_FEATURES:
            return
        try:
            value = float(value)
        except (TypeError, ValueError):
            return

        state = self._states.get(feature_name)
        if state is None:
            self._states[feature_name] = _FeatureState(ewma=value, variance=0.0, count=1)
            return

        old_ewma = state.ewma
        new_ewma = self.alpha * value + (1 - self.alpha) * old_ewma
        deviation = value - old_ewma
        new_variance = (1 - self.alpha) * (state.variance + self.alpha * (deviation**2))

        state.ewma = new_ewma
        state.variance = max(new_variance, 0.0)
        state.count += 1

    def z_score(self, feature_name: str, value: float) -> float:
        state = self._states.get(feature_name)
        if state is None:
            return 0.0
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not self.is_warm(feature_name):
            return 0.0

        std = math.sqrt(max(state.variance, 1e-8))
        return (value - state.ewma) / std

    def is_warm(self, feature_name: str) -> bool:
        state = self._states.get(feature_name)
        if state is None:
            return False
        return state.count >= self.warmup_segments

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "alpha": self.alpha,
            "warmup_segments": self.warmup_segments,
            "features": {name: st.to_dict() for name, st in self._states.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SpeakerBaseline":
        baseline = cls(
            speaker_id=str(payload.get("speaker_id", "default")),
            alpha=float(payload.get("alpha", 0.3)),
            warmup_segments=int(payload.get("warmup_segments", 5)),
        )
        features = payload.get("features", {}) or {}
        for name, state_payload in features.items():
            if name in SUPPORTED_FEATURES and isinstance(state_payload, dict):
                baseline._states[name] = _FeatureState.from_dict(state_payload)
        return baseline

    @classmethod
    def load_from_path(
        cls,
        speaker_id: str,
        base_dir: Path,
        alpha: float = 0.3,
        warmup_segments: int = 5,
    ) -> "SpeakerBaseline":
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / f"{speaker_id}.json"
        if not path.exists():
            return cls(speaker_id=speaker_id, alpha=alpha, warmup_segments=warmup_segments)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            baseline = cls.from_dict(payload)
            # Runtime-Parameter bevorzugen, falls abweichend geladen.
            baseline.alpha = alpha
            baseline.warmup_segments = warmup_segments
            baseline.speaker_id = speaker_id
            return baseline
        except Exception:
            return cls(speaker_id=speaker_id, alpha=alpha, warmup_segments=warmup_segments)

    def save_to_path(self, base_dir: Path) -> None:
        base_dir.mkdir(parents=True, exist_ok=True)
        path = base_dir / f"{self.speaker_id}.json"
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
