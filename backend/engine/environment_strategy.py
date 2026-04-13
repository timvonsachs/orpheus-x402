"""Environment-to-strategy impact mapping for realtime calls."""

from __future__ import annotations


class EnvironmentStrategy:
    """Mappt erkannte Umgebung auf konkrete Agent-Strategieanpassungen."""

    def recommend(self, environment: str, noise_level: str) -> dict:
        env = str(environment or "home_quiet")
        noise = str(noise_level or "low")

        base = {
            "use_short_sentences": noise in {"moderate", "high"},
            "avoid_personal_questions": False,
            "speak_louder": noise == "high",
            "expect_interruptions": noise in {"moderate", "high"},
            "offer_callback": False,
            "privacy_mode": False,
            "prioritize_essentials": noise in {"moderate", "high"},
            "prefer_yes_no_questions": False,
        }

        if env == "home_quiet":
            return base
        if env == "home_noisy":
            base["use_short_sentences"] = True
            base["speak_louder"] = True
            base["expect_interruptions"] = True
            base["prioritize_essentials"] = True
            return base
        if env == "office_quiet":
            base["prefer_yes_no_questions"] = True
            return base
        if env == "office_open":
            base["use_short_sentences"] = True
            base["avoid_personal_questions"] = True
            base["privacy_mode"] = True
            base["prefer_yes_no_questions"] = True
            base["expect_interruptions"] = True
            return base
        if env == "car":
            base["use_short_sentences"] = True
            base["speak_louder"] = True
            base["offer_callback"] = True
            base["prioritize_essentials"] = True
            base["expect_interruptions"] = True
            return base
        if env == "outdoor_street":
            base["use_short_sentences"] = True
            base["speak_louder"] = True
            base["prioritize_essentials"] = True
            base["expect_interruptions"] = True
            return base
        if env == "outdoor_nature":
            return base
        if env == "public_space":
            base["use_short_sentences"] = True
            base["avoid_personal_questions"] = True
            base["privacy_mode"] = True
            base["prefer_yes_no_questions"] = True
            base["expect_interruptions"] = True
            return base
        if env == "transit":
            base["use_short_sentences"] = True
            base["speak_louder"] = True
            base["offer_callback"] = True
            base["prioritize_essentials"] = True
            base["expect_interruptions"] = True
            return base
        return base
