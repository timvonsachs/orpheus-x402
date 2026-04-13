"""Formatiert relevante Highlights aus einer Sense-Response."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python demo/demo_analysis.py path/to/result.json")

    p = Path(sys.argv[1])
    payload = json.loads(p.read_text(encoding="utf-8"))

    print("== Orpheus Acoustic Sense Analyse ==")
    print(f"Version: {payload.get('orpheus_version')}")
    print(f"Processing: {payload.get('processing_time_ms')} ms")
    print(f"Dauer: {payload.get('audio_duration_seconds')} s")

    humanness = payload.get("humanness") or {}
    if humanness:
        print(f"Humanness: {humanness.get('score')} ({humanness.get('classification')})")

    para = payload.get("paralinguistic", {}).get("summary", {})
    if para:
        print(
            f"Paralinguistik -> Arousal={para.get('arousal')} | Stress={para.get('stress_level')} "
            f"| Engagement={para.get('engagement')}"
        )

    trends = payload.get("trends") or {}
    if trends:
        print(f"Trend-Interpretation: {trends.get('interpretation')}")

    alerts = payload.get("alerts") or []
    if alerts:
        print("Alerts:")
        for alert in alerts:
            print(f"- {alert.get('type')} [{alert.get('severity')}]: {alert.get('description')}")


if __name__ == "__main__":
    main()
