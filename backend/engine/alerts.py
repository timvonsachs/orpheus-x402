"""Detektion signifikanter akustischer Ereignisse."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from engine.baselines import SpeakerBaseline


class AlertDetector:
    def __init__(
        self,
        alpha: float = 0.3,
        warmup_segments: int = 5,
        z_threshold: float = 2.0,
        baseline_dir: Optional[str] = None,
    ):
        self.alpha = alpha
        self.warmup_segments = warmup_segments
        self.z_threshold = z_threshold
        if baseline_dir is None:
            self.baseline_dir = Path(__file__).resolve().parent.parent / "data" / "baselines"
        else:
            self.baseline_dir = Path(baseline_dir)

    def detect(self, segments: list[dict], speaker_id: str = "default") -> list[dict]:
        alerts: list[dict] = []
        if len(segments) < 2:
            return alerts

        baseline = SpeakerBaseline.load_from_path(
            speaker_id=speaker_id,
            base_dir=self.baseline_dir,
            alpha=self.alpha,
            warmup_segments=self.warmup_segments,
        )

        avg_pause = (
            sum(s.get("biomarkers", {}).get("pause_dur", 0.5) for s in segments) / max(len(segments), 1)
        )

        for idx, curr in enumerate(segments):
            prev = segments[idx - 1] if idx > 0 else None
            curr = segments[idx]
            prev_b = prev.get("biomarkers", {}) if prev else {}
            curr_b = curr.get("biomarkers", {})
            curr_para = curr.get("paralinguistic_summary", {})

            prev_jitter = float(prev_b.get("jitter", 0) or 0)
            curr_jitter = float(curr_b.get("jitter", 0) or 0)

            stress = float(curr_para.get("stress_level", 0) or 0)
            engagement = float(curr_para.get("engagement", 0) or 0)
            f0_mean = float(curr_b.get("f0_mean", 0) or 0)
            speech_rate = float(curr_b.get("speech_rate", 0) or 0)
            pause_rate = float(curr_b.get("pause_rate", 0) or 0)

            stress_z = baseline.z_score("stress", stress)
            engagement_z = baseline.z_score("engagement", engagement)
            f0_z = baseline.z_score("f0_mean", f0_mean)
            jitter_z = baseline.z_score("jitter", curr_jitter)
            rate_z = baseline.z_score("speech_rate", speech_rate)
            pause_rate_z = baseline.z_score("pause_rate", pause_rate)

            if idx > 0:
                stress_or_jitter_warm = baseline.is_warm("stress") or baseline.is_warm("jitter")
                if stress_or_jitter_warm:
                    if stress_z > self.z_threshold or jitter_z > self.z_threshold:
                        alerts.append(
                            {
                                "type": "stress_spike",
                                "timestamp": curr.get("start", 0),
                                "description": (
                                    f"Stress/Jitter weichen stark von Speaker-Baseline ab "
                                    f"(z={max(stress_z, jitter_z):.2f})."
                                ),
                                "severity": "medium",
                            }
                        )
                elif prev_jitter > 0 and (curr_jitter - prev_jitter) / prev_jitter > 0.3:
                    delta_pct = (curr_jitter - prev_jitter) / prev_jitter * 100
                    alerts.append(
                        {
                            "type": "stress_spike",
                            "timestamp": curr.get("start", 0),
                            "description": f"Jitter stieg um {delta_pct:.0f}% gegenueber dem Vorsegment.",
                            "severity": "medium",
                        }
                    )

            prev_rate = float(prev_b.get("speech_rate", 5) or 5)
            curr_rate = float(curr_b.get("speech_rate", 5) or 5)
            disengagement_warm = (
                baseline.is_warm("engagement")
                or baseline.is_warm("speech_rate")
                or baseline.is_warm("pause_rate")
            )
            if idx > 0:
                if disengagement_warm:
                    if (
                        engagement_z < -self.z_threshold
                        or rate_z < -self.z_threshold
                        or pause_rate_z > self.z_threshold
                    ):
                        alerts.append(
                            {
                                "type": "disengagement",
                                "timestamp": curr.get("start", 0),
                                "description": (
                                    "Engagement/Sprechtempo weichen negativ von der "
                                    f"Speaker-Baseline ab (z={min(engagement_z, rate_z):.2f})."
                                ),
                                "severity": "high",
                            }
                        )
                elif curr_rate < 3.0 and prev_rate > 4.0:
                    alerts.append(
                        {
                            "type": "disengagement",
                            "timestamp": curr.get("start", 0),
                            "description": f"Sprechrate fiel von {prev_rate:.1f} auf {curr_rate:.1f} Silben/s.",
                            "severity": "high",
                        }
                    )

            curr_pause = float(curr_b.get("pause_dur", 0.5) or 0.5)
            if idx > 0:
                if baseline.is_warm("pause_rate"):
                    if pause_rate_z > self.z_threshold:
                        alerts.append(
                            {
                                "type": "hesitation",
                                "timestamp": curr.get("start", 0),
                                "description": (
                                    f"Pause-Rate liegt deutlich ueber Speaker-Baseline "
                                    f"(z={pause_rate_z:.2f})."
                                ),
                                "severity": "low",
                            }
                        )
                elif curr_pause > max(1.0, avg_pause * 2):
                    alerts.append(
                        {
                            "type": "hesitation",
                            "timestamp": curr.get("start", 0),
                            "description": f"Pausendauer {curr_pause:.2f}s liegt deutlich ueber Baseline {avg_pause:.2f}s.",
                            "severity": "low",
                        }
                    )

            prev_f0 = float(prev_b.get("f0_mean", 0) or 0)
            curr_f0 = float(curr_b.get("f0_mean", 0) or 0)
            if idx > 0:
                if baseline.is_warm("f0_mean"):
                    if f0_z < -self.z_threshold:
                        alerts.append(
                            {
                                "type": "f0_drop",
                                "timestamp": curr.get("start", 0),
                                "description": f"Mittlere F0 liegt klar unter Speaker-Baseline (z={f0_z:.2f}).",
                                "severity": "medium",
                            }
                        )
                elif prev_f0 > 0 and (prev_f0 - curr_f0) / prev_f0 > 0.2:
                    drop_pct = (prev_f0 - curr_f0) / prev_f0 * 100
                    alerts.append(
                        {
                            "type": "f0_drop",
                            "timestamp": curr.get("start", 0),
                            "description": f"Mittlere F0 fiel um {drop_pct:.0f}% gegenueber dem Vorsegment.",
                            "severity": "medium",
                        }
                    )

            baseline.update("stress", stress)
            baseline.update("engagement", engagement)
            baseline.update("f0_mean", f0_mean)
            baseline.update("jitter", curr_jitter)
            baseline.update("speech_rate", speech_rate)
            baseline.update("pause_rate", pause_rate)

        baseline.save_to_path(self.baseline_dir)
        return alerts
