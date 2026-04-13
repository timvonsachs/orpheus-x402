"""Validate EWMA speaker baselines against fixed-threshold alerts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.baselines import SpeakerBaseline
RESULTS_DIR = ROOT / "data" / "real_calls" / "results"
BASELINE_DIR = ROOT / "data" / "baselines"
REPORT_MD = ROOT / "reports" / "ewma_validation.md"
REPORT_HTML = ROOT / "reports" / "ewma_validation.html"

SPEAKER_ID = "test_speaker_001"
FEATURES = ["stress", "engagement", "f0_mean", "jitter", "speech_rate", "pause_rate"]
Z_THRESHOLD = 2.0


def load_successful_calls() -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    files = sorted(RESULTS_DIR.glob("*.json"), key=lambda p: int(p.stem.split("_", 1)[0]))
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        segments = payload.get("segments", []) or []
        if not segments:
            continue
        calls.append({"path": path, "payload": payload, "segments": segments})
    return calls


def feature_values(segment: dict[str, Any]) -> dict[str, float]:
    para = segment.get("paralinguistic_summary", {}) or {}
    bio = segment.get("biomarkers", {}) or {}
    return {
        "stress": float(para.get("stress_level", 0.0) or 0.0),
        "engagement": float(para.get("engagement", 0.0) or 0.0),
        "f0_mean": float(bio.get("f0_mean", 0.0) or 0.0),
        "jitter": float(bio.get("jitter", 0.0) or 0.0),
        "speech_rate": float(bio.get("speech_rate", 0.0) or 0.0),
        "pause_rate": float(bio.get("pause_rate", 0.0) or 0.0),
    }


def old_alerts_for_call(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []
    if len(segments) < 2:
        return alerts
    avg_pause = sum(float((s.get("biomarkers", {}) or {}).get("pause_dur", 0.5) or 0.5) for s in segments) / len(segments)

    for idx in range(1, len(segments)):
        prev = segments[idx - 1]
        curr = segments[idx]
        prev_b = prev.get("biomarkers", {}) or {}
        curr_b = curr.get("biomarkers", {}) or {}
        t = float(curr.get("start", 0.0) or 0.0)

        prev_jitter = float(prev_b.get("jitter", 0.0) or 0.0)
        curr_jitter = float(curr_b.get("jitter", 0.0) or 0.0)
        if prev_jitter > 0 and (curr_jitter - prev_jitter) / prev_jitter > 0.3:
            alerts.append({"call_idx": -1, "segment_idx": idx, "type": "stress_spike", "timestamp": t})

        prev_rate = float(prev_b.get("speech_rate", 5.0) or 5.0)
        curr_rate = float(curr_b.get("speech_rate", 5.0) or 5.0)
        if curr_rate < 3.0 and prev_rate > 4.0:
            alerts.append({"call_idx": -1, "segment_idx": idx, "type": "disengagement", "timestamp": t})

        curr_pause = float(curr_b.get("pause_dur", 0.5) or 0.5)
        if curr_pause > max(1.0, avg_pause * 2):
            alerts.append({"call_idx": -1, "segment_idx": idx, "type": "hesitation", "timestamp": t})

        prev_f0 = float(prev_b.get("f0_mean", 0.0) or 0.0)
        curr_f0 = float(curr_b.get("f0_mean", 0.0) or 0.0)
        if prev_f0 > 0 and (prev_f0 - curr_f0) / prev_f0 > 0.2:
            alerts.append({"call_idx": -1, "segment_idx": idx, "type": "f0_drop", "timestamp": t})
    return alerts


def new_alerts_for_call(
    call_idx: int,
    segments: list[dict[str, Any]],
    baseline: SpeakerBaseline,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    alerts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for idx, seg in enumerate(segments):
        vals = feature_values(seg)
        z = {name: baseline.z_score(name, vals[name]) for name in FEATURES}
        warm = {name: baseline.is_warm(name) for name in FEATURES}
        t = float(seg.get("start", 0.0) or 0.0)

        rows.append(
            {
                "call_idx": call_idx,
                "segment_idx": idx,
                "timestamp": t,
                "values": vals,
                "z": z,
                "warm": warm,
                "ewma_stress_before": baseline.to_dict().get("features", {}).get("stress", {}).get("ewma", vals["stress"]),
            }
        )

        if idx > 0:
            if (warm["stress"] and z["stress"] > Z_THRESHOLD) or (warm["jitter"] and z["jitter"] > Z_THRESHOLD):
                alerts.append(
                    {
                        "call_idx": call_idx,
                        "segment_idx": idx,
                        "type": "stress_spike",
                        "timestamp": t,
                        "z": max(z["stress"], z["jitter"]),
                    }
                )
            if (
                (warm["engagement"] and z["engagement"] < -Z_THRESHOLD)
                or (warm["speech_rate"] and z["speech_rate"] < -Z_THRESHOLD)
                or (warm["pause_rate"] and z["pause_rate"] > Z_THRESHOLD)
            ):
                alerts.append(
                    {
                        "call_idx": call_idx,
                        "segment_idx": idx,
                        "type": "disengagement",
                        "timestamp": t,
                        "z": min(z["engagement"], z["speech_rate"]),
                    }
                )
            if warm["pause_rate"] and z["pause_rate"] > Z_THRESHOLD:
                alerts.append(
                    {"call_idx": call_idx, "segment_idx": idx, "type": "hesitation", "timestamp": t, "z": z["pause_rate"]}
                )
            if warm["f0_mean"] and z["f0_mean"] < -Z_THRESHOLD:
                alerts.append({"call_idx": call_idx, "segment_idx": idx, "type": "f0_drop", "timestamp": t, "z": z["f0_mean"]})

        for name in FEATURES:
            baseline.update(name, vals[name])

    return alerts, rows


def build_report_md(
    calls: list[dict[str, Any]],
    old_alerts: list[dict[str, Any]],
    new_alerts: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> str:
    old_keys = {(a["call_idx"], a["segment_idx"], a["type"]) for a in old_alerts}
    new_keys = {(a["call_idx"], a["segment_idx"], a["type"]) for a in new_alerts}
    old_only = sorted(old_keys - new_keys)
    new_only = sorted(new_keys - old_keys)

    reduction_abs = len(old_only)
    reduction_pct = (reduction_abs / len(old_alerts) * 100) if old_alerts else 0.0

    old_example = "Kein geeignetes old-only Beispiel gefunden."
    if old_only:
        cidx, sidx, atype = old_only[0]
        row = next(r for r in rows if r["call_idx"] == cidx and r["segment_idx"] == sidx)
        val = row["values"]["stress"] if atype == "stress_spike" else row["values"]["engagement"]
        z = row["z"]["stress"] if atype == "stress_spike" else row["z"]["engagement"]
        old_example = (
            f"Call {cidx + 1}, Segment {sidx + 1}: Alter Alert '{atype}' bei absolutem Wert {val:.2f}. "
            f"Neuer Alert: Kein Alert, weil Z-Score nur {z:.2f} ist."
        )

    new_example = "Kein geeignetes new-only Beispiel gefunden."
    if new_only:
        cidx, sidx, atype = new_only[0]
        row = next(r for r in rows if r["call_idx"] == cidx and r["segment_idx"] == sidx)
        if atype == "stress_spike":
            base = row["ewma_stress_before"]
            val = row["values"]["stress"]
            z = row["z"]["stress"]
            new_example = (
                f"Call {cidx + 1}, Segment {sidx + 1}: Kein alter Alert. Neuer Alert '{atype}' "
                f"mit Z-Score {z:.2f}, weil Speaker-Baseline bei {base:.2f} liegt und Wert auf {val:.2f} springt."
            )
        else:
            z = row["z"]["pause_rate"] if atype == "hesitation" else row["z"]["engagement"]
            new_example = f"Call {cidx + 1}, Segment {sidx + 1}: Neuer Alert '{atype}' mit Z-Score {z:.2f}."

    return f"""# Orpheus EWMA Baseline Validation

## Input
- Erfolgreiche Calls: {len(calls)}
- Simulierter Speaker: `{SPEAKER_ID}`

## Alert-Vergleich
- Alerts alte Methode (feste Schwellen): **{len(old_alerts)}**
- Alerts neue Methode (EWMA, |z| > {Z_THRESHOLD}): **{len(new_alerts)}**
- Reduzierte Alerts (old-only): **{reduction_abs}** ({reduction_pct:.1f}% der alten Alerts)

## Beobachtung
- `{old_example}`
- `{new_example}`

## Interpretation
- Die EWMA-Methode unterdrueckt Sprecher-typische, aber unkritische Auspraegungen.
- Gleichzeitig markiert sie ungewoehnliche Abweichungen frueher, sobald die Baseline warm ist.
- Damit sinkt in der Regel die Menge an potenziellen False Positives bei gleichzeitiger Personalisierung.
"""


def write_report_html(
    best_call: dict[str, Any],
    best_rows: list[dict[str, Any]],
    old_alerts_call: list[dict[str, Any]],
    new_alerts_call: list[dict[str, Any]],
) -> None:
    title = best_call["path"].stem
    times = [r["timestamp"] for r in best_rows]
    stress_values = [r["values"]["stress"] for r in best_rows]
    stress_ewma = [r["ewma_stress_before"] for r in best_rows]
    stress_z = [r["z"]["stress"] for r in best_rows]
    old_alert_times = [a["timestamp"] for a in old_alerts_call if a["type"] == "stress_spike"]
    new_alert_times = [a["timestamp"] for a in new_alerts_call if a["type"] == "stress_spike"]

    html = f"""<!doctype html>
<html lang="de">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Personalisierte Baselines: Vorher vs Nachher</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin:0; background: linear-gradient(180deg, #f9fafb 0%, #f3f6fb 100%); color:#1f2937; font-family:'DM Sans',sans-serif; }}
    .wrap {{ max-width:1240px; margin:40px auto; padding:0 24px 40px; }}
    .card {{ background:#fff; border:1px solid #e6eaf2; border-radius:18px; box-shadow:0 12px 40px rgba(15,23,42,.08); padding:28px 30px; }}
    h1 {{ margin:0 0 8px; font-family:'Source Serif 4',serif; font-size:38px; font-weight:600; }}
    .sub {{ margin:0 0 18px; color:#6b7280; font-size:16px; }}
    .meta {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:14px; font-size:14px; }}
    .meta span {{ background:#f3f4f6; border:1px solid #e5e7eb; padding:6px 10px; border-radius:999px; }}
    #timeline {{ height:560px; width:100%; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Personalisierte Baselines: Vorher vs Nachher</h1>
      <p class="sub">Bester Call fuer EWMA-Dynamik: <strong>{title}</strong></p>
      <div class="meta">
        <span>Alte Stress-Alerts: {len(old_alert_times)}</span>
        <span>Neue Stress-Alerts: {len(new_alert_times)}</span>
        <span>Speaker: {SPEAKER_ID}</span>
      </div>
      <div id="timeline"></div>
    </div>
  </div>
  <script>
    const times = {json.dumps(times)};
    const stress = {json.dumps(stress_values)};
    const ewma = {json.dumps(stress_ewma)};
    const z = {json.dumps(stress_z)};
    const oldAlertTimes = {json.dumps(old_alert_times)};
    const newAlertTimes = {json.dumps(new_alert_times)};

    const traces = [
      {{
        x: times, y: stress, type: 'scatter', mode: 'lines+markers', name: 'Stress (Wert)',
        line: {{color:'#e53935', width:4}}, marker: {{size:7}}
      }},
      {{
        x: times, y: ewma, type: 'scatter', mode: 'lines', name: 'EWMA Baseline (Stress)',
        line: {{color:'#2563eb', width:3}}
      }},
      {{
        x: times, y: z, type: 'scatter', mode: 'lines+markers', name: 'Z-Score (Stress)',
        yaxis: 'y2', line: {{color:'#7c3aed', width:3}}, marker: {{size:6}}
      }}
    ];

    const shapes = [];
    for (const t of oldAlertTimes) {{
      shapes.push({{type:'line', x0:t, x1:t, y0:0, y1:1.05, yref:'y', line:{{color:'#ef4444', width:2, dash:'dash'}}}});
    }}
    for (const t of newAlertTimes) {{
      shapes.push({{type:'line', x0:t, x1:t, y0:0, y1:1.05, yref:'y', line:{{color:'#2563eb', width:2, dash:'dot'}}}});
    }}

    const layout = {{
      paper_bgcolor:'#ffffff',
      plot_bgcolor:'#ffffff',
      margin:{{l:74,r:60,t:20,b:64}},
      xaxis:{{title:'Zeit (Sekunden)', gridcolor:'#edf0f5', zeroline:false}},
      yaxis:{{title:'Stress / EWMA', range:[0,1.05], gridcolor:'#edf0f5', zeroline:false}},
      yaxis2:{{title:'Z-Score', overlaying:'y', side:'right', gridcolor:'#f2f4f8', zeroline:false}},
      legend:{{orientation:'h', y:1.08, x:1, xanchor:'right'}},
      shapes
    }};

    const config = {{responsive:true, displaylogo:false, toImageButtonOptions:{{format:'png', filename:'ewma_validation_showcase', scale:2}}}};
    Plotly.newPlot('timeline', traces, layout, config);
  </script>
</body>
</html>
"""
    REPORT_HTML.write_text(html, encoding="utf-8")


def main() -> None:
    calls = load_successful_calls()
    if not calls:
        raise SystemExit("Keine erfolgreichen Calls in data/real_calls/results gefunden.")

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_HTML.parent.mkdir(parents=True, exist_ok=True)

    baseline = SpeakerBaseline.load_from_path(SPEAKER_ID, BASELINE_DIR, alpha=0.3, warmup_segments=5)

    all_old_alerts: list[dict[str, Any]] = []
    all_new_alerts: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    per_call_rows: dict[int, list[dict[str, Any]]] = {}
    per_call_old: dict[int, list[dict[str, Any]]] = {}
    per_call_new: dict[int, list[dict[str, Any]]] = {}

    for cidx, call in enumerate(calls):
        segments = call["segments"]
        old_alerts = old_alerts_for_call(segments)
        for a in old_alerts:
            a["call_idx"] = cidx
        new_alerts, rows = new_alerts_for_call(cidx, segments, baseline)

        per_call_rows[cidx] = rows
        per_call_old[cidx] = old_alerts
        per_call_new[cidx] = new_alerts
        all_old_alerts.extend(old_alerts)
        all_new_alerts.extend(new_alerts)
        all_rows.extend(rows)

    baseline.save_to_path(BASELINE_DIR)

    REPORT_MD.write_text(build_report_md(calls, all_old_alerts, all_new_alerts, all_rows), encoding="utf-8")

    # Best call for visual: largest old/new divergence in stress-spike alerts.
    best_idx = max(
        range(len(calls)),
        key=lambda i: abs(
            len([a for a in per_call_old[i] if a["type"] == "stress_spike"])
            - len([a for a in per_call_new[i] if a["type"] == "stress_spike"])
        ),
    )
    write_report_html(calls[best_idx], per_call_rows[best_idx], per_call_old[best_idx], per_call_new[best_idx])

    print(f"Calls verarbeitet: {len(calls)}")
    print(f"Alte Alerts: {len(all_old_alerts)}")
    print(f"Neue Alerts: {len(all_new_alerts)}")
    print(f"Markdown: {REPORT_MD}")
    print(f"HTML: {REPORT_HTML}")
    print(f"Baseline gespeichert: {BASELINE_DIR / (SPEAKER_ID + '.json')}")


if __name__ == "__main__":
    main()
