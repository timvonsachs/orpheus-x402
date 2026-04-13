"""Reprocess only failed real-call analyses using audio preprocessing."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any, Optional

import httpx

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.preprocessing import AudioPreprocessor
MANIFEST = ROOT / "data" / "real_calls" / "manifest.csv"
RESULTS_DIR = ROOT / "data" / "real_calls" / "results"
API_URL = "http://localhost:8001/v1/sense"


def load_manifest() -> list[dict[str, str]]:
    if not MANIFEST.exists():
        return []
    with MANIFEST.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def missing_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for row in rows:
        file_path = Path(row.get("file_path", ""))
        if not file_path.exists():
            continue
        result_path = RESULTS_DIR / f"{file_path.stem}.json"
        if result_path.exists():
            continue
        missing.append(row)
    return missing


def analyze_with_api(processed_path: Path) -> Optional[dict[str, Any]]:
    mime = "audio/wav"
    with httpx.Client(timeout=300) as client:
        with processed_path.open("rb") as f:
            resp = client.post(
                API_URL,
                files={"file": (processed_path.name, f, mime)},
                params={"mode": "full", "segments": "true"},
            )
    if resp.status_code != 200:
        print(f"[WARN] API error {resp.status_code} for {processed_path.name}: {resp.text[:180]}")
        return None
    return resp.json()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pre = AudioPreprocessor()

    rows = load_manifest()
    if not rows:
        raise SystemExit(f"Manifest leer oder fehlt: {MANIFEST}")

    missing = missing_rows(rows)
    total = len(missing)
    if total == 0:
        print("Keine fehlgeschlagenen Calls gefunden. Alle Calls haben bereits Ergebnisse.")
        return

    success = 0
    for row in missing:
        orig = Path(row["file_path"])
        print(f"[INFO] Reprocessing: {orig.name}")
        try:
            processed = Path(pre.process(orig.as_posix()))
        except Exception as exc:
            print(f"[WARN] Preprocessing failed for {orig.name}: {exc}")
            continue

        try:
            payload = analyze_with_api(processed)
            if payload is None:
                continue
            out = RESULTS_DIR / f"{orig.stem}.json"
            if out.exists():
                # Safety: do not overwrite successful pre-existing results.
                print(f"[SKIP] Result already exists: {out.name}")
                continue
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            success += 1
            print(f"[OK] Saved result: {out.name}")
        finally:
            if processed != orig and processed.exists():
                processed.unlink(missing_ok=True)

    print(f"{success} von {total} zuvor fehlgeschlagene Calls erfolgreich analysiert")


if __name__ == "__main__":
    main()
