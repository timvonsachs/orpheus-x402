"""Run light diarization on the best demo call and render showcase HTML."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.diarization import SpeakerDiarizer


CALL_WAV = ROOT / "data" / "real_calls" / "002_sales_call_example_1.wav"
CALL_JSON = ROOT / "data" / "real_calls" / "results" / "002_sales_call_example_1.json"
OUT_HTML = ROOT / "reports" / "diarization_demo.html"


def attach_customer_metrics(
    diarized: list[dict[str, Any]],
    analysis_segments: list[dict[str, Any]],
) -> tuple[list[float], list[float], list[float]]:
    times: list[float] = []
    stress: list[float] = []
    engagement: list[float] = []

    customer_ranges = [
        (float(s["start"]), float(s["end"]))
        for s in diarized
        if s.get("role") == "customer" and s.get("speaker") != "silence"
    ]

    for seg in analysis_segments:
        start = float(seg.get("start", 0.0) or 0.0)
        end = float(seg.get("end", 0.0) or 0.0)
        midpoint = (start + end) / 2
        is_customer = any(cs <= midpoint <= ce for cs, ce in customer_ranges)
        if not is_customer:
            continue
        para = seg.get("paralinguistic_summary", {}) or {}
        times.append(midpoint)
        stress.append(float(para.get("stress_level", 0.0) or 0.0))
        engagement.append(float(para.get("engagement", 0.0) or 0.0))
    return times, stress, engagement


def turn_taking_stats(diarized: list[dict[str, Any]]) -> dict[str, float | int]:
    speech = [s for s in diarized if s.get("role") in {"agent", "customer"}]
    if not speech:
        return {"total_turns": 0, "customer_talk_ratio": 0.0, "avg_customer_turn_length": 0.0, "interruptions": 0}

    total = 0.0
    customer = 0.0
    c_turns: list[float] = []
    interruptions = 0
    prev_role = None
    prev_end = None
    for seg in speech:
        start = float(seg["start"])
        end = float(seg["end"])
        dur = max(0.0, end - start)
        role = str(seg["role"])
        total += dur
        if role == "customer":
            customer += dur
            c_turns.append(dur)
        if prev_role is not None and role != prev_role and prev_end is not None and (start - prev_end) < 0.2:
            interruptions += 1
        prev_role = role
        prev_end = end

    return {
        "total_turns": len(speech),
        "customer_talk_ratio": round(customer / total, 3) if total > 0 else 0.0,
        "avg_customer_turn_length": round(sum(c_turns) / len(c_turns), 3) if c_turns else 0.0,
        "interruptions": interruptions,
    }


def main() -> None:
    if not CALL_WAV.exists():
        raise SystemExit(f"Call WAV fehlt: {CALL_WAV}")
    if not CALL_JSON.exists():
        raise SystemExit(f"Analyse-JSON fehlt: {CALL_JSON}")

    diarizer = SpeakerDiarizer()
    diarized = diarizer.diarize(str(CALL_WAV))
    diarized = diarizer.label_speakers(diarized, method="first_is_agent")

    payload = json.loads(CALL_JSON.read_text(encoding="utf-8"))
    analysis_segments = payload.get("segments", []) or []
    times, stress, engagement = attach_customer_metrics(diarized, analysis_segments)
    stats = turn_taking_stats(diarized)

    agent_bars = []
    customer_bars = []
    for seg in diarized:
        role = seg.get("role")
        if role == "agent":
            agent_bars.append({"start": seg["start"], "end": seg["end"]})
        elif role == "customer":
            customer_bars.append({"start": seg["start"], "end": seg["end"]})

    html = f"""<!doctype html>
<html lang="de">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Diarization Demo</title>
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
    #roles {{ height:250px; width:100%; margin-bottom:16px; }}
    #customer {{ height:360px; width:100%; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Speaker Diarization Light Demo</h1>
      <p class="sub">Best Call: 002_sales_call_example_1 - Agent/Kunde Trennung + Kundenemotionen.</p>
      <div class="meta">
        <span>Total Turns: {stats["total_turns"]}</span>
        <span>Customer Talk Ratio: {stats["customer_talk_ratio"]}</span>
        <span>Avg Customer Turn Length: {stats["avg_customer_turn_length"]}s</span>
        <span>Interruptions: {stats["interruptions"]}</span>
      </div>
      <div id="roles"></div>
      <div id="customer"></div>
    </div>
  </div>
  <script>
    const agentBars = {json.dumps(agent_bars)};
    const customerBars = {json.dumps(customer_bars)};
    const times = {json.dumps(times)};
    const stress = {json.dumps(stress)};
    const engagement = {json.dumps(engagement)};

    const roleShapes = [];
    for (const s of agentBars) {{
      roleShapes.push({{type:'rect', x0:s.start, x1:s.end, y0:0.55, y1:0.95, line:{{width:0}}, fillcolor:'rgba(37,99,235,0.75)'}});
    }}
    for (const s of customerBars) {{
      roleShapes.push({{type:'rect', x0:s.start, x1:s.end, y0:0.05, y1:0.45, line:{{width:0}}, fillcolor:'rgba(251,146,60,0.8)'}});
    }}

    Plotly.newPlot('roles', [], {{
      paper_bgcolor:'#ffffff',
      plot_bgcolor:'#ffffff',
      margin:{{l:60,r:20,t:20,b:45}},
      xaxis:{{title:'Zeit (Sekunden)', gridcolor:'#edf0f5', zeroline:false}},
      yaxis:{{range:[0,1], showgrid:false, tickvals:[0.25,0.75], ticktext:['Kunde','Agent']}},
      shapes: roleShapes
    }}, {{displaylogo:false, responsive:true}});

    Plotly.newPlot('customer', [
      {{
        x: times, y: stress, type:'scatter', mode:'lines+markers',
        name:'Customer Stress', line:{{color:'#e53935', width:4}}
      }},
      {{
        x: times, y: engagement, type:'scatter', mode:'lines+markers',
        name:'Customer Engagement', line:{{color:'#16a34a', width:4}}
      }}
    ], {{
      paper_bgcolor:'#ffffff',
      plot_bgcolor:'#ffffff',
      margin:{{l:60,r:20,t:20,b:55}},
      xaxis:{{title:'Zeit (Sekunden)', gridcolor:'#edf0f5', zeroline:false}},
      yaxis:{{title:'Wert (0-1)', range:[0,1], gridcolor:'#edf0f5', zeroline:false}},
      legend:{{orientation:'h', y:1.1, x:1, xanchor:'right'}}
    }}, {{displaylogo:false, responsive:true, toImageButtonOptions:{{format:'png', filename:'diarization_demo', scale:2}}}});
  </script>
</body>
</html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Diarization segments: {len(diarized)}")
    print(f"Customer metric points: {len(times)}")
    print(f"Turn stats: {stats}")
    print(f"Report written: {OUT_HTML}")


if __name__ == "__main__":
    main()
