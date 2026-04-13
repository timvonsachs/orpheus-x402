"""Analyze phrase-level correlation with stress changes."""

from __future__ import annotations

import re


class KeywordEmotionAnalyzer:
    """Findet welche Worte/Phrasen mit Stress-Spikes korrelieren."""

    WORD_RE = re.compile(r"[A-Za-z0-9']+")

    def analyze(self, segments_with_text: list) -> dict:
        trigger_phrases: list[dict] = []
        calming_phrases: list[dict] = []

        for idx in range(1, len(segments_with_text)):
            prev = segments_with_text[idx - 1]
            cur = segments_with_text[idx]
            prev_stress = self._stress(prev)
            cur_stress = self._stress(cur)
            text = str(cur.get("text", "") or "").strip()
            if not text:
                continue
            delta = cur_stress - prev_stress
            if abs(delta) < 0.12:
                continue

            phrase = self._extract_phrase(text)
            item = {
                "text": phrase,
                "context": text,
                "stress_before": round(prev_stress, 3),
                "stress_after": round(cur_stress, 3),
                "stress_delta": round(delta, 3),
                "timestamp": round(float(cur.get("start", 0.0) or 0.0), 3),
            }
            if delta > 0:
                trigger_phrases.append(item)
            else:
                calming_phrases.append(item)

        return {"trigger_phrases": trigger_phrases[:20], "calming_phrases": calming_phrases[:20]}

    def _extract_phrase(self, text: str) -> str:
        words = [w.lower() for w in self.WORD_RE.findall(text)]
        if not words:
            return text[:40]
        if len(words) >= 3:
            return " ".join(words[:3])
        return " ".join(words)

    @staticmethod
    def _stress(seg: dict) -> float:
        para = seg.get("paralinguistic_summary", {}) or {}
        return float(para.get("stress_level", 0.0) or 0.0)
