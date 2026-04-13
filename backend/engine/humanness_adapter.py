"""Rule-based adapter for agent humanness diagnostics and TTS fixes."""

from __future__ import annotations


class HumannessAdapter:
    """Analysiert Agent-Audio auf Humanness und gibt TTS-Korrekturen."""

    def analyze(self, audio_chunk: object, humanness_score: float) -> dict:
        del audio_chunk  # Aktuell rein regelbasiert auf Score.
        score = float(humanness_score or 0.0)

        if score < 40:
            diagnosis = "robotic"
            tts_fix = {
                "pitch_variation": 0.2,
                "pause_randomness": 0.4,
                "breathing_sounds": True,
                "micro_hesitations": True,
                "speed_variation": 0.15,
            }
        elif score < 60:
            diagnosis = "too_stiff"
            tts_fix = {
                "pitch_variation": 0.15,
                "pause_randomness": 0.3,
                "breathing_sounds": True,
                "micro_hesitations": False,
                "speed_variation": 0.1,
            }
        elif score < 80:
            diagnosis = "acceptable"
            tts_fix = {
                "pitch_variation": 0.05,
                "pause_randomness": 0.15,
                "breathing_sounds": False,
                "micro_hesitations": False,
                "speed_variation": 0.05,
            }
        else:
            diagnosis = "human"
            tts_fix = {
                "pitch_variation": 0.0,
                "pause_randomness": 0.0,
                "breathing_sounds": False,
                "micro_hesitations": False,
                "speed_variation": 0.0,
            }

        details = {
            "pitch_variation_low": score < 75,
            "pauses_too_regular": score < 70,
            "breathing_absent": score < 65,
            "speed_too_constant": score < 70,
        }
        if diagnosis == "human":
            details = {
                "pitch_variation_low": False,
                "pauses_too_regular": False,
                "breathing_absent": False,
                "speed_too_constant": False,
            }

        return {
            "agent_state": {
                "humanness_score": round(score, 1),
                "diagnosis": diagnosis,
                "details": details,
            },
            "tts_fix": tts_fix,
        }

    @staticmethod
    def trend(scores: list[float]) -> str:
        if len(scores) < 2:
            return "stable"
        window = [float(s) for s in scores[-5:]]
        delta = window[-1] - window[0]
        if delta > 4.0:
            return "improving"
        if delta < -4.0:
            return "degrading"
        return "stable"
