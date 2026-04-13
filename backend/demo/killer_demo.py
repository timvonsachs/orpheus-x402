"""
Killer Demo: Agent mit vs. ohne Orpheus Acoustic Sense.

Zeigt an konkreten Sales-Szenarien, wie ein Agent mit paralinguistischem
Kontext bessere Entscheidungen trifft als ein Agent, der nur den Text sieht.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ORPHEUS_URL = os.getenv("ORPHEUS_URL", "http://localhost:8001/v1/sense")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

USE_LIVE_LLM = OPENAI_API_KEY is not None


async def analyze_audio(audio_path: str) -> dict:
    """Sendet Audio an Orpheus und gibt die Analyse zurueck."""
    path = Path(audio_path)
    suffix = path.suffix.lower()
    mime = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
        ".ogg": "audio/ogg",
    }.get(suffix, "application/octet-stream")

    async with httpx.AsyncClient(timeout=120) as client:
        with path.open("rb") as f:
            response = await client.post(
                ORPHEUS_URL,
                files={"file": (path.name, f, mime)},
            )
        response.raise_for_status()
        return response.json()


async def call_llm(messages: List[Dict[str, str]]) -> str:
    """Call OpenAI Chat Completions API."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY fehlt.")

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_MODEL,
                "messages": messages,
                "max_tokens": 200,
                "temperature": 0.7,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def agent_without_orpheus(transcript: str) -> str:
    """Agent sieht nur den Transkript-Text."""
    if not USE_LIVE_LLM:
        return FALLBACK_RESPONSES["without_orpheus"]

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI sales agent for a SaaS company selling a CRM platform. "
                "You are on a live call with a potential customer. Respond naturally and conversationally. "
                "Your goal is to close the deal or book a follow-up meeting. "
                "Keep your response to 2-3 sentences."
            ),
        },
        {
            "role": "user",
            "content": f'The customer just said: "{transcript}"\n\nWhat do you say next?',
        },
    ]
    return await call_llm(messages)


def build_orpheus_context(orpheus_data: dict) -> str:
    """Bereitet den relevanten paralinguistischen Kontext fuer den Prompt auf."""
    para = orpheus_data.get("paralinguistic", {}).get("summary", {})
    trends = orpheus_data.get("trends", {})
    alerts = orpheus_data.get("alerts", [])
    biomarkers = orpheus_data.get("paralinguistic", {}).get("biomarkers", {})

    parts = [
        f"Arousal: {para.get('arousal', 'N/A')}",
        f"Stress level: {para.get('stress_level', 'N/A')}",
        f"Engagement: {para.get('engagement', 'N/A')}",
        f"Confidence: {para.get('confidence_level', 'N/A')}",
        f"Valence: {para.get('valence_estimate', 'N/A')}",
    ]

    if trends:
        parts.append(f"Stress trend: {trends.get('stress_trend', 'N/A')} ({trends.get('stress_delta', 0):+.2f})")
        parts.append(
            f"Engagement trend: {trends.get('engagement_trend', 'N/A')} ({trends.get('engagement_delta', 0):+.2f})"
        )
        if trends.get("interpretation"):
            parts.append(f"Interpretation: {trends['interpretation']}")

    if alerts:
        alert_strs = [f"[{a.get('severity', 'n/a')}] {a.get('description', '')}" for a in alerts]
        parts.append(f"Alerts: {'; '.join(alert_strs)}")

    if biomarkers:
        parts.append(f"F0 (pitch): {biomarkers.get('f0_mean', 'N/A')} Hz")
        parts.append(f"Jitter (vocal tension): {biomarkers.get('jitter', 'N/A')}")
        parts.append(f"Speech rate: {biomarkers.get('speech_rate', 'N/A')} syl/sec")
        parts.append(f"Pause rate: {biomarkers.get('pause_rate', 'N/A')}%")
        parts.append(f"HNR (voice clarity): {biomarkers.get('hnr', 'N/A')}")

    return "\n".join(parts)


async def agent_with_orpheus(transcript: str, orpheus_data: dict) -> str:
    """Agent sieht Text + akustischen Kontext von Orpheus."""
    if not USE_LIVE_LLM:
        return FALLBACK_RESPONSES["with_orpheus"]

    orpheus_context = build_orpheus_context(orpheus_data)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI sales agent for a SaaS company selling a CRM platform. "
                "You are on a live call with a potential customer. Respond naturally and conversationally. "
                "Your goal is to close the deal or book a follow-up meeting.\n\n"
                "IMPORTANT: You have access to real-time acoustic analysis of the customer's voice. "
                "This tells you HOW they sound, not just WHAT they say. Use this information to adapt your approach. "
                "If the customer sounds stressed or disengaged, do not push harder - acknowledge their state and adjust.\n\n"
                "Keep your response to 2-3 sentences."
            ),
        },
        {
            "role": "user",
            "content": (
                f'The customer just said: "{transcript}"\n\n'
                "ACOUSTIC ANALYSIS (Orpheus Acoustic Sense):\n"
                f"{orpheus_context}\n\n"
                "Based on both the words AND the acoustic signals, what do you say next?"
            ),
        },
    ]
    return await call_llm(messages)


def print_analysis_box(orpheus_data: dict) -> None:
    para = orpheus_data.get("paralinguistic", {}).get("summary", {})
    trends = orpheus_data.get("trends", {})
    alerts = orpheus_data.get("alerts", [])

    print("\n  +-- ORPHEUS ACOUSTIC ANALYSIS --------------------------------------+")
    print(f"  | Arousal:    {para.get('arousal', 'N/A')}")
    print(f"  | Stress:     {para.get('stress_level', 'N/A')}")
    print(f"  | Engagement: {para.get('engagement', 'N/A')}")
    print(f"  | Confidence: {para.get('confidence_level', 'N/A')}")
    print(f"  | Valence:    {para.get('valence_estimate', 'N/A')}")
    if trends.get("interpretation"):
        print("  |")
        print(f"  | Trend: {trends['interpretation']}")
    if alerts:
        print("  |")
        for alert in alerts:
            print(f"  | ! [{alert.get('severity', 'n/a')}] {alert.get('description', '')}")
    print("  +-------------------------------------------------------------------+")


def print_agent_box(title: str, subtitle: str, response: str) -> None:
    print(f"\n  +-- {title} ---------------------------------------------+")
    print(f"  | {subtitle}")
    for line in response.replace("\n", " ").split(". "):
        text = line.strip()
        if not text:
            continue
        if not text.endswith("."):
            text = f"{text}."
        print(f"  | {text}")
    print("  +-------------------------------------------------------------------+")


async def run_demo(audio_path: Optional[str] = None) -> None:
    print("\n" + "=" * 70)
    print("  ORPHEUS ACOUSTIC SENSE - KILLER DEMO")
    print("  Same words. Same customer. Different intelligence.")
    print("=" * 70)

    for i, scenario in enumerate(SCENARIOS):
        print(f"\n{'-' * 70}")
        print(f"  SCENARIO {i + 1}: {scenario['name']}")
        print(f"{'-' * 70}")
        print(f"\n  Context: {scenario['context']}")
        print(f'  Customer says: "{scenario["transcript"]}"')

        selected_audio = audio_path or scenario.get("audio_file")
        if selected_audio and Path(selected_audio).exists():
            print("\n  [Analyzing audio with Orpheus...]")
            try:
                orpheus_data = await analyze_audio(selected_audio)
            except Exception as exc:
                print(f"  [WARN] Audio analysis failed, fallback to simulated data: {exc}")
                orpheus_data = SIMULATED_ORPHEUS[scenario["name"]]
        else:
            print("\n  [Using simulated Orpheus data for demo]")
            orpheus_data = SIMULATED_ORPHEUS[scenario["name"]]

        print_analysis_box(orpheus_data)

        response_without = await agent_without_orpheus(scenario["transcript"])
        print_agent_box(
            "AGENT WITHOUT ORPHEUS",
            f'Only sees: "{scenario["transcript"]}"',
            response_without,
        )

        response_with = await agent_with_orpheus(scenario["transcript"], orpheus_data)
        print_agent_box(
            "AGENT WITH ORPHEUS",
            "Sees words + acoustic analysis",
            response_with,
        )

    print(f"\n{'=' * 70}")
    print("  The same words. Completely different responses.")
    print("  Orpheus hears what the customer does not say.")
    print("=" * 70 + "\n")


FALLBACK_RESPONSES = {
    "without_orpheus": (
        "Great to hear you find it interesting! Our CRM platform has helped companies like yours "
        "increase sales by 30%. I'd love to set up a quick demo - does Thursday at 2pm work for you?"
    ),
    "with_orpheus": (
        "I can sense this is a big decision, and I want to make sure you have all the information you need. "
        "A lot of our customers had similar hesitations about the pricing initially. "
        "Would it help if I walked you through exactly how the ROI breaks down for a company your size?"
    ),
}


SCENARIOS = [
    {
        "name": "Price Hesitation",
        "context": "Agent just told the customer the price is $2,400/year. Customer responds:",
        "transcript": "Hmm, that's interesting. Let me think about it.",
        "audio_file": None,
    },
    {
        "name": "Fake Enthusiasm",
        "context": "Agent is presenting features. Customer responds:",
        "transcript": "Oh wow, that sounds really great, yeah.",
        "audio_file": None,
    },
    {
        "name": "Growing Frustration",
        "context": "Agent is explaining the onboarding process (3rd minute of call). Customer responds:",
        "transcript": "Sure, I understand. Go on.",
        "audio_file": None,
    },
]


SIMULATED_ORPHEUS: Dict[str, Dict[str, Any]] = {
    "Price Hesitation": {
        "paralinguistic": {
            "summary": {
                "arousal": 0.45,
                "stress_level": 0.68,
                "engagement": 0.35,
                "confidence_level": 0.28,
                "valence_estimate": "negative",
            },
            "biomarkers": {
                "f0_mean": 142.3,
                "jitter": 0.044,
                "speech_rate": 3.4,
                "pause_rate": 32.1,
                "hnr": 4.8,
            },
        },
        "trends": {
            "stress_trend": "rising",
            "stress_delta": 0.22,
            "engagement_trend": "falling",
            "engagement_delta": -0.31,
            "interpretation": (
                "Speaker shows increasing stress with decreasing engagement since pricing was mentioned. "
                "Long pause before response indicates deliberation. Low confidence suggests price objection."
            ),
        },
        "alerts": [
            {
                "type": "hesitation",
                "timestamp": 2.1,
                "description": "Pause duration exceeds baseline - significant hesitation.",
                "severity": "high",
            },
            {
                "type": "f0_drop",
                "timestamp": 3.4,
                "description": "F0 dropped on 'think about it' - emotional withdrawal.",
                "severity": "medium",
            },
        ],
    },
    "Fake Enthusiasm": {
        "paralinguistic": {
            "summary": {
                "arousal": 0.52,
                "stress_level": 0.41,
                "engagement": 0.29,
                "confidence_level": 0.55,
                "valence_estimate": "neutral",
            },
            "biomarkers": {
                "f0_mean": 168.7,
                "jitter": 0.031,
                "speech_rate": 5.1,
                "pause_rate": 15.2,
                "hnr": 6.2,
            },
        },
        "trends": {
            "stress_trend": "stable",
            "stress_delta": 0.03,
            "engagement_trend": "falling",
            "engagement_delta": -0.28,
            "interpretation": (
                "Words are positive but acoustic profile contradicts. "
                "Engagement is dropping - likely polite disengagement."
            ),
        },
        "alerts": [
            {
                "type": "disengagement",
                "timestamp": 1.5,
                "description": "Narrow F0 range despite enthusiastic words.",
                "severity": "high",
            }
        ],
    },
    "Growing Frustration": {
        "paralinguistic": {
            "summary": {
                "arousal": 0.71,
                "stress_level": 0.74,
                "engagement": 0.22,
                "confidence_level": 0.61,
                "valence_estimate": "negative",
            },
            "biomarkers": {
                "f0_mean": 195.2,
                "jitter": 0.052,
                "speech_rate": 5.8,
                "pause_rate": 8.3,
                "hnr": 4.1,
            },
        },
        "trends": {
            "stress_trend": "rising",
            "stress_delta": 0.34,
            "engagement_trend": "falling",
            "engagement_delta": -0.45,
            "interpretation": (
                "Speaker sounds frustrated while staying polite. "
                "Rising stress with falling engagement suggests presentational overload."
            ),
        },
        "alerts": [
            {
                "type": "stress_spike",
                "timestamp": 0.8,
                "description": "Jitter elevated well above comfortable range.",
                "severity": "high",
            },
            {
                "type": "disengagement",
                "timestamp": 1.2,
                "description": "Response length dropped significantly.",
                "severity": "high",
            },
        ],
    },
}


if __name__ == "__main__":
    audio = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run_demo(audio))
