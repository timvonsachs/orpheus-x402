"""Rule-based mapping from emotional state to TTS parameters."""

from __future__ import annotations


class TTSAdapter:
    """Maps customer emotional state to recommended TTS parameters."""

    def recommend(self, stress: float, engagement: float, arousal: float, z_score_stress: float) -> dict[str, float]:
        stress = float(stress or 0.0)
        engagement = float(engagement or 0.0)
        arousal = float(arousal or 0.0)
        z_score_stress = float(z_score_stress or 0.0)

        if stress > 0.7 or z_score_stress > 2.0:
            return {
                "speed": 0.85,
                "pitch_shift": -0.1,
                "warmth": 0.9,
                "pause_before_next": 1.5,
            }

        if engagement < 0.3:
            return {
                "speed": 1.05,
                "pitch_shift": 0.05,
                "warmth": 0.6,
                "pause_before_next": 0.3,
            }

        if stress < 0.4 and engagement > 0.7:
            return {
                "speed": 1.0,
                "pitch_shift": 0.0,
                "warmth": 0.5 + 0.2 * min(max(arousal, 0.0), 1.0),
                "pause_before_next": 0.5,
            }

        if 0.4 <= stress <= 0.7:
            return {
                "speed": 0.95,
                "pitch_shift": 0.0,
                "warmth": 0.7,
                "pause_before_next": 0.8,
            }

        return {
            "speed": 0.98,
            "pitch_shift": 0.0,
            "warmth": 0.65,
            "pause_before_next": 0.6,
        }
