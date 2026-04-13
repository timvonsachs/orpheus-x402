"""Kleiner Demo-Client fuer den Endpoint /v1/sense."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orpheus Acoustic Sense Demo Client")
    parser.add_argument("audio", type=str, help="Pfad zur Audio-Datei")
    parser.add_argument("--mode", choices=["full", "human", "agent"], default="full")
    parser.add_argument("--url", default="http://localhost:8001/v1/sense")
    parser.add_argument("--segments", action="store_true", default=True)
    parser.add_argument("--segment-duration", type=float, default=5.0)
    parser.add_argument("--raw", action="store_true", help="Rohes JSON ausgeben")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise SystemExit(f"Datei nicht gefunden: {audio_path}")

    params = {
        "mode": args.mode,
        "segments": str(args.segments).lower(),
        "segment_duration": args.segment_duration,
    }

    with audio_path.open("rb") as f:
        response = httpx.post(
            args.url,
            params=params,
            files={"file": (audio_path.name, f, "application/octet-stream")},
            timeout=180.0,
        )
    response.raise_for_status()
    payload = response.json()

    if args.raw:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    print(f"\n{'=' * 60}")
    print(f"ORPHEUS ACOUSTIC SENSE v{payload.get('orpheus_version', '?')}")
    print(f"{'=' * 60}")
    print(f"Processing Time: {payload.get('processing_time_ms', '?')} ms")
    if payload.get("audio_duration_seconds") is not None:
        print(f"Audio Duration: {payload['audio_duration_seconds']} s")

    if "humanness" in payload and payload["humanness"]:
        h = payload["humanness"]
        print("\n--- OHR 2: AGENT HUMANNNESS ---")
        print(f"Score:          {h.get('score')}/100")
        print(f"Classification: {h.get('classification')}")
        print(f"Confidence:     {h.get('confidence')}")
        if h.get("interpretation"):
            print(f"Interpretation: {h['interpretation']}")

    if "paralinguistic" in payload and payload["paralinguistic"]:
        p = payload["paralinguistic"]
        s = p.get("summary", {})
        print("\n--- OHR 1: HUMAN PARALINGUISTIK ---")
        print(f"Arousal:    {s.get('arousal')}")
        print(f"Stress:     {s.get('stress_level')}")
        print(f"Engagement: {s.get('engagement')}")
        print(f"Confidence: {s.get('confidence_level')}")
        print(f"Valence:    {s.get('valence_estimate')}")

    if payload.get("trends"):
        t = payload["trends"]
        print("\n--- TRENDS ---")
        print(f"Arousal:    {t.get('arousal_trend')} ({t.get('arousal_delta', 0):+.3f})")
        print(f"Stress:     {t.get('stress_trend')} ({t.get('stress_delta', 0):+.3f})")
        print(f"Engagement: {t.get('engagement_trend')} ({t.get('engagement_delta', 0):+.3f})")
        print(f"Interpretation: {t.get('interpretation')}")

    if payload.get("alerts"):
        print("\n--- ALERTS ---")
        for alert in payload["alerts"]:
            print(f"[{alert.get('severity', '').upper()}] @{alert.get('timestamp')}s: {alert.get('description')}")

    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
