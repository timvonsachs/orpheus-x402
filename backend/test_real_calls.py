"""Real-world validation pipeline for telephone call audio.

Commands:
  - collect: Download candidate calls from YouTube into data/real_calls/
  - analyze: Send files to Orpheus API and store JSON + summary CSV
  - report:  Build markdown + plotly report + PNG screenshots
  - all:     collect -> analyze -> report
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import subprocess
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
import plotly.graph_objects as go

ORPHEUS_URL = "http://localhost:8001/v1/sense"
ORPHEUS_HEALTH = "http://localhost:8001/health"
LEGACY_ORPHEUS_HEALTH = "http://localhost:8000/api/v1/health"

SEARCH_QUERIES = [
    "real sales call recording",
    "cold call recording real",
    "sales call objection handling recording",
    "customer service call recording real",
    "sales call gone wrong",
    "real phone sales conversation",
    "insurance sales call recording",
    "real estate cold call recording",
    "B2B sales call example real",
    "appointment setting call recording",
]

DATA_DIR = Path("data/real_calls")
RESULTS_DIR = DATA_DIR / "results"
REPORTS_DIR = Path("reports")
MANIFEST_PATH = DATA_DIR / "manifest.csv"
SUMMARY_PATH = DATA_DIR / "results_summary.csv"
REPORT_HTML = REPORTS_DIR / "real_call_analysis.html"
REPORT_MD = REPORTS_DIR / "real_call_summary.md"

MANIFEST_FIELDS = ["file_path", "title", "url", "duration", "search_query"]
SUMMARY_FIELDS = [
    "file_path",
    "title",
    "url",
    "duration",
    "status",
    "humanness_score",
    "humanness_classification",
    "paralinguistic_arousal",
    "paralinguistic_stress",
    "paralinguistic_engagement",
    "paralinguistic_confidence",
    "paralinguistic_valence",
    "segments_count",
    "alerts_count",
    "alert_types",
    "arousal_trend",
    "stress_trend",
    "engagement_trend",
    "trend_interpretation",
    "processing_time_ms",
    "biomarker_valid_count",
    "biomarker_total_checked",
    "stress_range",
    "engagement_range",
]

BIOMARKER_CHECKS = {
    "f0_mean": (50, 400),
    "jitter": (0.001, 0.1),
    "shimmer": (0.01, 0.3),
    "hnr": (0, 30),
    "speech_rate": (1, 10),
    "pause_rate": (0, 80),
}


@dataclass
class Candidate:
    title: str
    url: str
    duration: int
    search_query: str


def sanitize_filename(name: str, max_len: int = 80) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    if not cleaned:
        cleaned = "untitled"
    return cleaned[:max_len]


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_cmd(command: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, capture_output=True, text=True, check=False)


def yt_dlp_command(args: List[str]) -> List[str]:
    direct = shutil.which("yt-dlp")
    if direct:
        return [direct] + args
    return [sys.executable, "-m", "yt_dlp"] + args


def load_manifest() -> List[Dict[str, str]]:
    if not MANIFEST_PATH.exists():
        return []
    with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest(rows: List[Dict[str, str]]) -> None:
    with MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def fetch_candidates_for_query(query: str, per_query: int) -> List[Candidate]:
    proc = run_cmd(yt_dlp_command(["--flat-playlist", f"ytsearch{per_query}:{query}", "-j"]))
    if proc.returncode != 0:
        print(f"[WARN] yt-dlp failed for query '{query}': {proc.stderr.strip()[:200]}")
        return []

    candidates: List[Candidate] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        duration = int(obj.get("duration") or 0)
        if not (60 <= duration <= 15 * 60):
            continue
        video_id = obj.get("id")
        url = obj.get("url")
        if video_id:
            url = f"https://www.youtube.com/watch?v={video_id}"
        if not url:
            continue
        candidates.append(
            Candidate(
                title=str(obj.get("title") or "Untitled"),
                url=str(url),
                duration=duration,
                search_query=query,
            )
        )
    return candidates


def discover_calls(per_query: int = 5) -> List[Candidate]:
    all_candidates: List[Candidate] = []
    seen_urls: set[str] = set()

    for query in SEARCH_QUERIES:
        candidates = fetch_candidates_for_query(query, per_query=per_query)
        for c in candidates:
            if c.url in seen_urls:
                continue
            seen_urls.add(c.url)
            all_candidates.append(c)
    return all_candidates


def resolve_downloaded_audio(base_path: Path) -> Optional[Path]:
    matches = sorted(base_path.parent.glob(base_path.name + ".*"))
    if not matches:
        return None
    for path in matches:
        if path.suffix.lower() == ".wav":
            return path
    return matches[0]


def convert_to_wav(src: Path, dst: Path) -> bool:
    proc = run_cmd(["ffmpeg", "-y", "-i", str(src), str(dst)])
    return proc.returncode == 0 and dst.exists()


def download_audio(candidate: Candidate, idx: int) -> Optional[Path]:
    base_name = f"{idx:03d}_{sanitize_filename(candidate.title)}"
    base_path = DATA_DIR / base_name
    wav_target = DATA_DIR / f"{base_name}.wav"

    if wav_target.exists():
        return wav_target

    cmd = yt_dlp_command([
        "-x",
        "--audio-format",
        "wav",
        "-o",
        str(base_path) + ".%(ext)s",
        candidate.url,
    ])
    proc = run_cmd(cmd)
    if proc.returncode != 0:
        print(f"[WARN] Download failed (wav) for {candidate.url}. Trying mp3 fallback.")
        fallback = run_cmd(
            yt_dlp_command([
                "-x",
                "--audio-format",
                "mp3",
                "-o",
                str(base_path) + ".%(ext)s",
                candidate.url,
            ])
        )
        if fallback.returncode != 0:
            print(f"[WARN] Download failed for {candidate.url}: {fallback.stderr.strip()[:200]}")
            return None

    out = resolve_downloaded_audio(base_path)
    if out is None:
        print(f"[WARN] No downloaded file found for {candidate.url}")
        return None

    if out.suffix.lower() != ".wav":
        ok = convert_to_wav(out, wav_target)
        if not ok:
            print(f"[WARN] ffmpeg conversion to wav failed for {out}")
            return None
        return wav_target

    if out != wav_target:
        out.rename(wav_target)
    return wav_target


def collect_calls(target_count: int, per_query: int) -> None:
    ensure_dirs()
    existing = load_manifest()
    existing_urls = {row["url"] for row in existing}
    downloaded = len(existing)
    print(f"[INFO] Existing manifest entries: {downloaded}")

    candidates = discover_calls(per_query=per_query)
    if not candidates:
        print("[WARN] No candidates discovered.")
        return

    rows = existing[:]
    idx = downloaded + 1
    for cand in candidates:
        if len(rows) >= target_count:
            break
        if cand.url in existing_urls:
            continue
        path = download_audio(cand, idx=idx)
        if not path:
            continue
        row = {
            "file_path": str(path),
            "title": cand.title,
            "url": cand.url,
            "duration": str(cand.duration),
            "search_query": cand.search_query,
        }
        rows.append(row)
        existing_urls.add(cand.url)
        idx += 1
        print(f"[OK] Downloaded: {path.name}")

    write_manifest(rows)
    print(f"[DONE] Manifest updated: {MANIFEST_PATH} ({len(rows)} rows)")


async def check_services() -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        for name, url in [("Acoustic Sense", ORPHEUS_HEALTH), ("Legacy Orpheus", LEGACY_ORPHEUS_HEALTH)]:
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    print(f"[OK] {name} reachable: {url}")
                else:
                    print(f"[WARN] {name} health returned {r.status_code}: {url}")
            except Exception as exc:
                print(f"[WARN] {name} health check failed: {exc}")


def evaluate_biomarker_ranges(biomarkers: Dict[str, Any]) -> Tuple[int, int]:
    valid = 0
    total = 0
    for feat, (low, high) in BIOMARKER_CHECKS.items():
        val = biomarkers.get(feat)
        if val is None:
            continue
        total += 1
        try:
            fval = float(val)
        except (TypeError, ValueError):
            continue
        if low <= fval <= high:
            valid += 1
    return valid, total


def segment_ranges(segments: List[Dict[str, Any]]) -> Tuple[float, float]:
    if not segments:
        return 0.0, 0.0
    stress_values: List[float] = []
    engagement_values: List[float] = []
    for seg in segments:
        p = seg.get("paralinguistic_summary", {})
        try:
            if p.get("stress_level") is not None:
                stress_values.append(float(p["stress_level"]))
            if p.get("engagement") is not None:
                engagement_values.append(float(p["engagement"]))
        except (TypeError, ValueError):
            continue
    stress_range = max(stress_values) - min(stress_values) if len(stress_values) >= 2 else 0.0
    engagement_range = max(engagement_values) - min(engagement_values) if len(engagement_values) >= 2 else 0.0
    return round(stress_range, 3), round(engagement_range, 3)


async def analyze_call(audio_path: Path) -> Optional[Dict[str, Any]]:
    async with httpx.AsyncClient(timeout=300) as client:
        mime = "audio/wav" if audio_path.suffix.lower() == ".wav" else "application/octet-stream"
        with audio_path.open("rb") as f:
            response = await client.post(
                ORPHEUS_URL,
                files={"file": (audio_path.name, f, mime)},
                params={"mode": "full", "segments": "true"},
            )
        if response.status_code == 200:
            return response.json()
        print(f"[WARN] API error {response.status_code} for {audio_path.name}: {response.text[:180]}")
        return None


async def analyze_calls() -> None:
    ensure_dirs()
    await check_services()
    manifest_rows = load_manifest()
    if not manifest_rows:
        print("[WARN] manifest.csv leer. Bitte zuerst 'collect' ausfuehren.")
        return

    summary_rows: List[Dict[str, str]] = []
    for row in manifest_rows:
        audio_path = Path(row["file_path"])
        if not audio_path.exists():
            print(f"[WARN] Missing file: {audio_path}")
            continue

        print(f"[INFO] Analyzing {audio_path.name}")
        payload = await analyze_call(audio_path)
        if payload is None:
            summary_rows.append(
                {
                    "file_path": row["file_path"],
                    "title": row["title"],
                    "url": row["url"],
                    "duration": row["duration"],
                    "status": "error",
                    **{k: "" for k in SUMMARY_FIELDS if k not in {"file_path", "title", "url", "duration", "status"}},
                }
            )
            continue

        json_path = RESULTS_DIR / f"{audio_path.stem}.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        hum = payload.get("humanness", {})
        para = payload.get("paralinguistic", {}).get("summary", {})
        biomarkers = payload.get("paralinguistic", {}).get("biomarkers", {})
        trends = payload.get("trends", {})
        alerts = payload.get("alerts", []) or []
        segments = payload.get("segments", []) or []
        stress_range, engagement_range = segment_ranges(segments)
        valid_count, total_count = evaluate_biomarker_ranges(biomarkers)

        summary_rows.append(
            {
                "file_path": row["file_path"],
                "title": row["title"],
                "url": row["url"],
                "duration": row["duration"],
                "status": "ok",
                "humanness_score": str(hum.get("score", "")),
                "humanness_classification": str(hum.get("classification", "")),
                "paralinguistic_arousal": str(para.get("arousal", "")),
                "paralinguistic_stress": str(para.get("stress_level", "")),
                "paralinguistic_engagement": str(para.get("engagement", "")),
                "paralinguistic_confidence": str(para.get("confidence_level", "")),
                "paralinguistic_valence": str(para.get("valence_estimate", "")),
                "segments_count": str(len(segments)),
                "alerts_count": str(len(alerts)),
                "alert_types": ",".join(sorted({str(a.get("type", "")) for a in alerts if a.get("type")})),
                "arousal_trend": str(trends.get("arousal_trend", "")),
                "stress_trend": str(trends.get("stress_trend", "")),
                "engagement_trend": str(trends.get("engagement_trend", "")),
                "trend_interpretation": str(trends.get("interpretation", "")),
                "processing_time_ms": str(payload.get("processing_time_ms", "")),
                "biomarker_valid_count": str(valid_count),
                "biomarker_total_checked": str(total_count),
                "stress_range": str(stress_range),
                "engagement_range": str(engagement_range),
            }
        )

    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"[DONE] Wrote summary: {SUMMARY_PATH}")


def read_summary() -> List[Dict[str, str]]:
    if not SUMMARY_PATH.exists():
        return []
    with SUMMARY_PATH.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_call_payload(file_path: str) -> Optional[Dict[str, Any]]:
    stem = Path(file_path).stem
    json_path = RESULTS_DIR / f"{stem}.json"
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def select_best_calls(summary_rows: List[Dict[str, str]], top_n: int = 3) -> List[Dict[str, str]]:
    scored: List[Tuple[float, Dict[str, str]]] = []
    for row in summary_rows:
        if row.get("status") != "ok":
            continue
        score = (
            safe_float(row.get("stress_range"))
            + safe_float(row.get("engagement_range"))
            + min(safe_float(row.get("alerts_count")), 4) * 0.05
            + (0.2 if safe_float(row.get("segments_count")) >= 3 else 0.0)
        )
        scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scored[:top_n]]


def build_timeline_figure(title: str, payload: Dict[str, Any]) -> go.Figure:
    segments = payload.get("segments", []) or []
    times = [safe_float(s.get("start")) for s in segments]
    stress = [safe_float(s.get("paralinguistic_summary", {}).get("stress_level")) for s in segments]
    engagement = [safe_float(s.get("paralinguistic_summary", {}).get("engagement")) for s in segments]
    confidence = [safe_float(s.get("paralinguistic_summary", {}).get("confidence_level")) for s in segments]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=stress, mode="lines+markers", name="Stress", line=dict(color="red")))
    fig.add_trace(
        go.Scatter(x=times, y=engagement, mode="lines+markers", name="Engagement", line=dict(color="green"))
    )
    fig.add_trace(
        go.Scatter(x=times, y=confidence, mode="lines+markers", name="Confidence", line=dict(color="blue"))
    )

    for alert in payload.get("alerts", []) or []:
        sev = str(alert.get("severity", "low"))
        color = "red" if sev == "high" else ("goldenrod" if sev == "medium" else "gray")
        ts = safe_float(alert.get("timestamp"))
        fig.add_vline(x=ts, line_color=color, line_dash="dash", opacity=0.8)

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Value (0-1)",
        template="plotly_white",
        height=400,
    )
    return fig


def build_biomarker_heatmap(title: str, payload: Dict[str, Any]) -> go.Figure:
    segments = payload.get("segments", []) or []
    features = ["f0_mean", "jitter", "shimmer", "hnr", "speech_rate", "pause_rate", "pause_dur"]
    matrix: List[List[float]] = []
    x_labels: List[str] = []
    for seg in segments:
        bio = seg.get("biomarkers", {}) or {}
        matrix.append([safe_float(bio.get(feat)) for feat in features])
        x_labels.append(f"{safe_float(seg.get('start')):.1f}-{safe_float(seg.get('end')):.1f}s")

    if not matrix:
        matrix = [[0.0 for _ in features]]
        x_labels = ["n/a"]

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=list(map(list, zip(*matrix))),
                x=x_labels,
                y=features,
                colorscale="Viridis",
                colorbar=dict(title="Raw value"),
            )
        ]
    )
    fig.update_layout(title=title, template="plotly_white", height=380)
    return fig


def render_report_html(best_rows: List[Dict[str, str]], payloads: Dict[str, Dict[str, Any]]) -> None:
    sections: List[str] = []
    include_js = "cdn"
    for idx, row in enumerate(best_rows, start=1):
        payload = payloads[row["file_path"]]
        title = f"Call {idx}: {row['title']}"
        timeline = build_timeline_figure(title + " - Timeline", payload)
        heatmap = build_biomarker_heatmap(title + " - Biomarker Heatmap", payload)

        timeline_html = timeline.to_html(full_html=False, include_plotlyjs=include_js)
        include_js = False
        heatmap_html = heatmap.to_html(full_html=False, include_plotlyjs=False)

        summary_box = (
            f"<p><b>Humanness:</b> {row.get('humanness_score', 'n/a')} "
            f"({row.get('humanness_classification', 'n/a')}) | "
            f"<b>Alerts:</b> {row.get('alerts_count', '0')} | "
            f"<b>Stress range:</b> {row.get('stress_range', '0')} | "
            f"<b>Engagement range:</b> {row.get('engagement_range', '0')}</p>"
            f"<p><b>Trend insight:</b> {row.get('trend_interpretation', 'n/a')}</p>"
        )
        sections.append(f"<h2>{title}</h2>{summary_box}{timeline_html}{heatmap_html}<hr/>")

    html = (
        "<html><head><meta charset='utf-8'><title>Orpheus Real Call Analysis</title></head><body>"
        "<h1>Orpheus Real-World Validation: Telephone Audio</h1>"
        + "".join(sections)
        + "</body></html>"
    )
    REPORT_HTML.write_text(html, encoding="utf-8")
    print(f"[DONE] Wrote dashboard: {REPORT_HTML}")


def write_timeline_pngs(best_rows: List[Dict[str, str]], payloads: Dict[str, Dict[str, Any]]) -> None:
    for idx, row in enumerate(best_rows, start=1):
        payload = payloads[row["file_path"]]
        fig = build_timeline_figure(f"Demo Call {idx} Timeline", payload)
        out = REPORTS_DIR / f"demo_call_{idx}_timeline.png"
        try:
            fig.write_image(str(out), width=1200, height=400)
            print(f"[DONE] Wrote screenshot: {out}")
        except Exception as exc:
            print(f"[WARN] PNG export failed for {out}: {exc}")


def build_markdown_summary(summary_rows: List[Dict[str, str]], best_rows: List[Dict[str, str]]) -> str:
    ok_rows = [r for r in summary_rows if r.get("status") == "ok"]
    total = len(summary_rows)
    ok_total = len(ok_rows)

    biomarker_plausible = 0
    hum_scores: List[float] = []
    hum_human = 0
    stress_ranges: List[float] = []
    engagement_ranges: List[float] = []
    dynamic_calls = 0
    total_alerts = 0

    for row in ok_rows:
        valid = safe_float(row.get("biomarker_valid_count"))
        checked = safe_float(row.get("biomarker_total_checked"))
        if checked > 0 and valid / checked >= 0.8:
            biomarker_plausible += 1

        hs = safe_float(row.get("humanness_score"), default=-1.0)
        if hs >= 0:
            hum_scores.append(hs)
        if row.get("humanness_classification") == "human":
            hum_human += 1

        sr = safe_float(row.get("stress_range"))
        er = safe_float(row.get("engagement_range"))
        stress_ranges.append(sr)
        engagement_ranges.append(er)
        if sr > 0.1 or er > 0.1:
            dynamic_calls += 1

        total_alerts += int(safe_float(row.get("alerts_count")))

    avg_hum = mean(hum_scores) if hum_scores else 0.0
    min_hum = min(hum_scores) if hum_scores else 0.0
    max_hum = max(hum_scores) if hum_scores else 0.0
    avg_stress_range = mean(stress_ranges) if stress_ranges else 0.0
    avg_engage_range = mean(engagement_ranges) if engagement_ranges else 0.0

    if ok_total == 0:
        conclusion = "FAILS"
    else:
        plausible_ratio = biomarker_plausible / ok_total
        dynamic_ratio = dynamic_calls / ok_total
        all_human_like = hum_human == ok_total and (min_hum >= 60 if hum_scores else False)
        if plausible_ratio >= 0.8 and dynamic_ratio >= 0.6 and all_human_like:
            conclusion = "WORKS"
        elif plausible_ratio >= 0.5:
            conclusion = "PARTIALLY"
        else:
            conclusion = "FAILS"

    best_lines = []
    for i, row in enumerate(best_rows, start=1):
        why = (
            f"Stress range {row.get('stress_range', '0')}, "
            f"engagement range {row.get('engagement_range', '0')}, "
            f"alerts {row.get('alerts_count', '0')}"
        )
        best_lines.append(f"{i}. {row.get('title', 'n/a')}: {why}")

    md = f"""# Orpheus Real-World Validation: Telephone Audio

## Dataset
- {ok_total} successful analyses out of {total} collected calls
- Sources: sales calls, cold calls, customer service (YouTube)
- Audio quality: telephone compression, noise, spontaneous speech

## Key Question
Do the Orpheus biomarkers work on real telephone audio?

## Results

### Biomarker Validity
- {biomarker_plausible} of {ok_total} calls reached >=80% plausible biomarker checks
- Check ranges:
  - f0_mean: 50-400 Hz
  - jitter: 0.001-0.1
  - shimmer: 0.01-0.3
  - hnr: 0-30
  - speech_rate: 1-10
  - pause_rate: 0-80

### Emotional Variation
- Average stress range across segments: {avg_stress_range:.3f}
- Average engagement range across segments: {avg_engage_range:.3f}
- Calls with detectable emotional dynamics (>0.1 stress or engagement range): {dynamic_calls} of {ok_total}

### Alert Activity
- Total alerts fired: {total_alerts}
- Alert plausibility requires manual listening around alert timestamps

### Humanness Scores
- Average: {avg_hum:.2f}
- Range: {min_hum:.2f} - {max_hum:.2f}
- Classified as human: {hum_human} of {ok_total}

## Conclusion
{conclusion} on telephone audio.

## Best Demo Calls
{chr(10).join(best_lines) if best_lines else "- n/a"}
"""
    return md


def generate_report() -> None:
    ensure_dirs()
    summary_rows = read_summary()
    if not summary_rows:
        print("[WARN] No summary data. Run analyze first.")
        return

    best_rows = select_best_calls(summary_rows, top_n=3)
    payloads: Dict[str, Dict[str, Any]] = {}
    for row in best_rows:
        payload = load_call_payload(row["file_path"])
        if payload:
            payloads[row["file_path"]] = payload

    best_rows = [r for r in best_rows if r["file_path"] in payloads]
    render_report_html(best_rows, payloads)
    write_timeline_pngs(best_rows, payloads)
    REPORT_MD.write_text(build_markdown_summary(summary_rows, best_rows), encoding="utf-8")
    print(f"[DONE] Wrote markdown report: {REPORT_MD}")


async def run_all(args: argparse.Namespace) -> None:
    collect_calls(target_count=args.target, per_query=args.per_query)
    await analyze_calls()
    generate_report()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orpheus real call validation")
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect", help="Download YouTube call audio")
    p_collect.add_argument("--target", type=int, default=15, help="Target number of calls in manifest")
    p_collect.add_argument("--per-query", type=int, default=5, help="ytsearchN depth per query")

    sub.add_parser("analyze", help="Analyze manifest audio through API")
    sub.add_parser("report", help="Generate HTML/MD/PNG reports")

    p_all = sub.add_parser("all", help="collect + analyze + report")
    p_all.add_argument("--target", type=int, default=15, help="Target number of calls in manifest")
    p_all.add_argument("--per-query", type=int, default=5, help="ytsearchN depth per query")

    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    if args.command == "collect":
        collect_calls(target_count=args.target, per_query=args.per_query)
    elif args.command == "analyze":
        await analyze_calls()
    elif args.command == "report":
        generate_report()
    elif args.command == "all":
        await run_all(args)
    else:
        raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    asyncio.run(main())
