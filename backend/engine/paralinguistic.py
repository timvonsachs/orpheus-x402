"""Interpretation von Biomarkern zu paralinguistischen Signalen."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "emotion_classifier.joblib"
SCALER_PATH = ROOT / "models" / "feature_scaler.joblib"
META_PATH = ROOT / "models" / "emotion_classifier_meta.json"


@lru_cache(maxsize=1)
def _load_ml_assets() -> tuple[Optional[object], Optional[object], dict]:
    try:
        if not (MODEL_PATH.exists() and SCALER_PATH.exists() and META_PATH.exists()):
            return None, None, {}
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        return model, scaler, meta
    except Exception:
        return None, None, {}


class ParalinguisticInterpreter:
    def interpret(self, biomarkers: dict) -> dict:
        ml = self.predict_emotion_ml(biomarkers)
        out = {
            "arousal": self._compute_arousal(biomarkers),
            "valence_estimate": self._estimate_valence(biomarkers),
            "engagement": self._compute_engagement(biomarkers),
            "stress_level": self._compute_stress(biomarkers),
            "confidence_level": self._compute_confidence(biomarkers),
        }
        if ml.get("available"):
            # Prefer trained classifier when available.
            out["arousal"] = round(float(ml.get("probability", out["arousal"])), 2)
        out["ml_emotion"] = ml
        return out

    def predict_emotion_ml(self, features: dict) -> dict:
        model, scaler, meta = _load_ml_assets()
        if model is None or scaler is None or not meta:
            return {
                "available": False,
                "source": "rule_based_fallback",
                "prediction": "unknown",
                "probability": round(float(self._compute_arousal(features)), 3),
            }
        try:
            names = list(meta.get("feature_names", []))
            if not names:
                return {
                    "available": False,
                    "source": "rule_based_fallback",
                    "prediction": "unknown",
                    "probability": round(float(self._compute_arousal(features)), 3),
                }
            vec = np.asarray([[float(features.get(name, 0.0) or 0.0) for name in names]], dtype=np.float32)
            vec_s = scaler.transform(vec)
            proba = model.predict_proba(vec_s)[0]
            if len(proba) >= 2:
                p_high = float(proba[1])
                pred = "high_arousal" if p_high >= 0.5 else "low_arousal"
                return {
                    "available": True,
                    "source": "ml_classifier",
                    "prediction": pred,
                    "probability": round(p_high, 4),
                }
            return {
                "available": False,
                "source": "rule_based_fallback",
                "prediction": "unknown",
                "probability": round(float(self._compute_arousal(features)), 3),
            }
        except Exception:
            return {
                "available": False,
                "source": "rule_based_fallback",
                "prediction": "unknown",
                "probability": round(float(self._compute_arousal(features)), 3),
            }

    def _compute_arousal(self, b: dict) -> float:
        f0_norm = self._normalize(b.get("f0_mean", 0), low=80, high=300)
        rate_norm = self._normalize(b.get("speech_rate", 0), low=2.0, high=7.0)
        loud_norm = self._normalize(b.get("loudness_rms", 0), low=0.01, high=0.15)
        value = 0.4 * f0_norm + 0.35 * rate_norm + 0.25 * loud_norm
        return round(self._clamp01(value), 2)

    def _compute_stress(self, b: dict) -> float:
        jitter_norm = self._normalize(b.get("jitter", 0), low=0.01, high=0.06)
        shimmer_norm = self._normalize(b.get("shimmer", 0), low=0.03, high=0.15)
        hnr_inv = 1 - self._normalize(b.get("hnr", 0), low=2, high=15)
        f0_var_norm = self._normalize(b.get("f0_var", 0), low=5, high=50)
        value = 0.3 * jitter_norm + 0.25 * shimmer_norm + 0.25 * hnr_inv + 0.2 * f0_var_norm
        return round(self._clamp01(value), 2)

    def _compute_engagement(self, b: dict) -> float:
        rate_norm = self._normalize(b.get("speech_rate", 0), low=2.0, high=6.0)
        pause_inv = 1 - self._normalize(b.get("pause_rate", 0), low=10, high=50)
        f0_range_norm = self._normalize(b.get("f0_range_st", 0), low=3, high=20)
        value = 0.35 * rate_norm + 0.35 * pause_inv + 0.3 * f0_range_norm
        return round(self._clamp01(value), 2)

    def _compute_confidence(self, b: dict) -> float:
        jitter_inv = 1 - self._normalize(b.get("jitter", 0), low=0.01, high=0.06)
        hnr_norm = self._normalize(b.get("hnr", 0), low=2, high=15)
        pause_inv = 1 - self._normalize(b.get("pause_rate", 0), low=10, high=40)
        value = 0.35 * jitter_inv + 0.35 * hnr_norm + 0.3 * pause_inv
        return round(self._clamp01(value), 2)

    def _estimate_valence(self, b: dict) -> str:
        hnr = b.get("hnr", 5)
        jitter = b.get("jitter", 0.03)
        if hnr > 7 and jitter < 0.03:
            return "positive"
        if hnr < 4 or jitter > 0.05:
            return "negative"
        return "neutral"

    @staticmethod
    def _normalize(value: float, low: float, high: float) -> float:
        if high == low:
            return 0.5
        return (value - low) / (high - low)

    @staticmethod
    def _clamp01(value: float) -> float:
        return min(max(value, 0.0), 1.0)
