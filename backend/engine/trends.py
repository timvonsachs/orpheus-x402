"""Zeitliche Trendanalyse ueber Segmentdaten."""

from __future__ import annotations


class TrendAnalyzer:
    def analyze(self, segments: list[dict]) -> dict:
        if len(segments) < 2:
            return {"interpretation": "Zu kurze Aufnahme fuer Trendanalyse."}

        arousal_series = [s.get("paralinguistic_summary", {}).get("arousal", 0.0) for s in segments]
        stress_series = [s.get("paralinguistic_summary", {}).get("stress_level", 0.0) for s in segments]
        engagement_series = [s.get("paralinguistic_summary", {}).get("engagement", 0.0) for s in segments]
        f0_series = [s.get("biomarkers", {}).get("f0_mean", 0.0) for s in segments]
        rate_series = [s.get("biomarkers", {}).get("speech_rate", 0.0) for s in segments]
        pause_dur_series = [s.get("biomarkers", {}).get("pause_dur", 0.0) for s in segments]

        a_delta, a_trend = self._trend(arousal_series, threshold=0.05)
        s_delta, s_trend = self._trend(stress_series, threshold=0.05)
        e_delta, e_trend = self._trend(engagement_series, threshold=0.05)
        _, f_trend = self._trend(f0_series, threshold=3.0)
        _, r_trend = self._trend(rate_series, threshold=0.25)
        _, p_trend = self._trend(pause_dur_series, threshold=0.15)

        return {
            "arousal_trend": a_trend,
            "arousal_delta": round(a_delta, 3),
            "engagement_trend": e_trend,
            "engagement_delta": round(e_delta, 3),
            "stress_trend": s_trend,
            "stress_delta": round(s_delta, 3),
            "f0_trend": f_trend,
            "speech_rate_trend": "slowing" if r_trend == "falling" else ("speeding_up" if r_trend == "rising" else "stable"),
            "pause_duration_trend": p_trend if p_trend != "stable" else "stable",
            "interpretation": self._interpret(a_trend, s_trend, e_trend, f_trend, r_trend),
        }

    @staticmethod
    def _trend(series: list[float], threshold: float) -> tuple[float, str]:
        if len(series) < 2:
            return 0.0, "stable"
        delta = series[-1] - series[0]
        if abs(delta) < threshold:
            return delta, "stable"
        return delta, "rising" if delta > 0 else "falling"

    @staticmethod
    def _interpret(arousal: str, stress: str, engagement: str, f0: str, rate: str) -> str:
        parts: list[str] = []

        if stress == "rising" and engagement == "falling":
            parts.append("Steigender Stress bei sinkendem Engagement.")
            parts.append("Empfehlung: Tempo reduzieren, Bedenken spiegeln, offene Frage stellen.")
        if arousal == "rising" and f0 == "rising" and stress == "stable":
            parts.append("Positiver Aktivierungsanstieg deutet auf Interesse hin.")
        if engagement == "falling" and rate == "falling":
            parts.append("Sprechverhalten verlangsamt sich, moegliche Abkopplung im Gespraech.")
        if stress == "rising" and arousal == "rising" and f0 == "falling":
            parts.append("Arousal steigt bei fallender Grundfrequenz, Muster passt zu Frustration.")
        if not parts:
            parts.append("Keine markanten zeitlichen Verschiebungen erkennbar.")

        return " ".join(parts)
