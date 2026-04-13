"""Rule-based conversation strategy recommendations."""

from __future__ import annotations
from typing import Optional


class ConversationStrategy:
    """Maps emotional state to conversation strategy recommendation."""

    def recommend(
        self,
        stress: float,
        engagement: float,
        trend: str,
        z_score: float,
        environment: Optional[str] = None,
        noise_level: Optional[str] = None,
    ) -> dict[str, str]:
        stress = float(stress or 0.0)
        engagement = float(engagement or 0.0)
        z_score = float(z_score or 0.0)
        trend = str(trend or "stable")
        environment = str(environment or "")
        noise_level = str(noise_level or "")

        if stress > 0.7 and environment == "car":
            return {
                "strategy": "safety_callback",
                "action": "Kunde klingt belastet im Auto. Biete an, spaeter zurueckzurufen.",
                "urgency": "high",
            }

        if stress > 0.7 and environment == "office_open":
            return {
                "strategy": "privacy_timing_check",
                "action": "Frage, ob ein besserer Zeitpunkt fuer ein sensibles Gespraech passt.",
                "urgency": "high",
            }

        if engagement < 0.3 and environment == "public_space":
            return {
                "strategy": "contextual_reengage",
                "action": "Ablenkung scheint umgebungsbedingt. Kurz halten und Kernnutzen hervorheben.",
                "urgency": "medium",
            }

        if stress > 0.7 and trend == "escalating":
            return {
                "strategy": "empathy",
                "action": "Stop pushing. Ask an open question. Acknowledge the concern.",
                "urgency": "high",
            }

        if stress > 0.7 and trend == "stable":
            return {
                "strategy": "de-escalate",
                "action": "Slow down. Summarize what customer said. Confirm understanding.",
                "urgency": "medium",
            }

        if engagement < 0.3:
            return {
                "strategy": "re-engage",
                "action": "Change topic. Ask a surprising question. Share an unexpected fact.",
                "urgency": "medium",
            }

        if stress < 0.4 and engagement > 0.7:
            return {
                "strategy": "advance",
                "action": "Customer is receptive. Move to next step or close.",
                "urgency": "low",
            }

        if z_score > 2.0:
            return {
                "strategy": "de-escalate",
                "action": "State calm intent and ask for the most important blocker.",
                "urgency": "high",
            }

        if noise_level == "high":
            return {
                "strategy": "clarify_briefly",
                "action": "Kurz und klar sprechen, Schluesselpunkte wiederholen.",
                "urgency": "medium",
            }

        return {
            "strategy": "continue",
            "action": "Maintain current approach.",
            "urgency": "low",
        }
