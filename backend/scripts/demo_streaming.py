"""Streaming demo: send 500ms chunks and render timeline report."""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import wave

import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.keyword_emotion import KeywordEmotionAnalyzer
from main import app

REPORT_HTML = ROOT / "reports" / "streaming_demo.html"
REPORT_REALTIME_V2 = ROOT / "reports" / "orpheus-realtime-demo-v2.html"
REAL_CALLS_DIR = ROOT / "data" / "real_calls"
TARGET_SR = 16000
CHUNK_SECONDS = 0.5
CHUNK_SAMPLES = int(TARGET_SR * CHUNK_SECONDS)
MAX_DEMO_SECONDS = 60.0


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1).astype(np.float32)
        return mono, int(sr)
    except Exception:
        import torchaudio

        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0).numpy().astype(np.float32), int(sr)


def _resample(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return samples
    duration = len(samples) / float(src_sr)
    dst_len = max(1, int(round(duration * dst_sr)))
    src_x = np.linspace(0.0, duration, num=len(samples), endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    return np.interp(dst_x, src_x, samples).astype(np.float32)


def _to_wav_bytes(samples: np.ndarray, sample_rate: int = TARGET_SR) -> bytes:
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _pick_call() -> Path:
    candidates = sorted(REAL_CALLS_DIR.glob("*.wav"))
    if not candidates:
        raise SystemExit(f"Keine WAV-Dateien gefunden in: {REAL_CALLS_DIR}")
    return candidates[0]


def _build_report(rows: list[dict], source_call: Path) -> None:
    REPORT_HTML.parent.mkdir(parents=True, exist_ok=True)
    times = [float(r["timestamp"]) for r in rows]
    speakers = [str(r.get("speaker", "silence")) for r in rows]
    stress = [float((r.get("customer_state") or {}).get("stress", np.nan)) for r in rows]
    engagement = [float((r.get("customer_state") or {}).get("engagement", np.nan)) for r in rows]
    speed = [float((r.get("tts_params") or {}).get("speed", np.nan)) for r in rows]
    humanness = [float((r.get("agent_state") or {}).get("humanness_score", np.nan)) for r in rows]
    humanness_trend = [str((r.get("agent_state") or {}).get("trend", "stable")) for r in rows]
    diagnosis = [str((r.get("agent_state") or {}).get("diagnosis", "")) for r in rows]
    tts_fix_pitch = [float((r.get("tts_fix") or {}).get("pitch_variation", np.nan)) for r in rows]
    strategy = [str((r.get("recommendation") or {}).get("strategy", "")) for r in rows]
    alerts = [str(r["alert"]) for r in rows]
    env_detected = [str((r.get("environment") or {}).get("detected", "")) for r in rows]
    env_confidence = [float((r.get("environment") or {}).get("confidence", np.nan)) for r in rows]
    env_noise = [str((r.get("environment") or {}).get("noise_level", "")) for r in rows]
    env_impact = [dict((r.get("environment") or {}).get("strategy_impact", {}) or {}) for r in rows]

    marker_times = [times[i] for i in range(1, len(strategy)) if strategy[i] and strategy[i] != strategy[i - 1]]
    marker_labels = [strategy[i] for i in range(1, len(strategy)) if strategy[i] and strategy[i] != strategy[i - 1]]
    humanness_fix_annotations = [
        (times[i], humanness[i], tts_fix_pitch[i], diagnosis[i])
        for i in range(len(rows))
        if speakers[i] in {"agent", "both"} and not np.isnan(humanness[i]) and not np.isnan(tts_fix_pitch[i])
    ]

    env_changes = []
    for i in range(1, len(env_detected)):
        if env_detected[i] and env_detected[i] != env_detected[i - 1]:
            env_changes.append((times[i], env_detected[i], env_noise[i]))

    html = f"""<!doctype html>
<html lang="de">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Streaming Demo</title>
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
    .grid {{ display:grid; grid-template-columns:2fr 1fr; gap:14px; }}
    #chart {{ height:620px; width:100%; }}
    #envChart {{ height:620px; width:100%; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Orpheus Realtime Streaming Demo</h1>
      <p class="sub">Chunk-basiertes Feedback (500ms) fuer: <strong>{source_call.name}</strong></p>
      <div class="meta">
        <span>Chunks: {len(rows)}</span>
        <span>Strategie-Wechsel: {len(marker_times)}</span>
        <span>Alerts != none: {sum(1 for a in alerts if a != "none")}</span>
        <span>Agent-Chunks: {sum(1 for s in speakers if s in ("agent", "both"))}</span>
        <span>Kunden-Chunks: {sum(1 for s in speakers if s in ("customer", "both"))}</span>
        <span>Environment-Wechsel: {len(env_changes)}</span>
      </div>
      <div class="grid">
        <div id="chart"></div>
        <div id="envChart"></div>
      </div>
    </div>
  </div>
  <script>
    const times = {json.dumps(times)};
    const speakers = {json.dumps(speakers)};
    const stress = {json.dumps(stress)};
    const engagement = {json.dumps(engagement)};
    const speed = {json.dumps(speed)};
    const humanness = {json.dumps(humanness)};
    const humannessTrend = {json.dumps(humanness_trend)};
    const diagnosis = {json.dumps(diagnosis)};
    const ttsFixPitch = {json.dumps(tts_fix_pitch)};
    const strategy = {json.dumps(strategy)};
    const alerts = {json.dumps(alerts)};
    const envDetected = {json.dumps(env_detected)};
    const envConfidence = {json.dumps(env_confidence)};
    const envNoise = {json.dumps(env_noise)};
    const envImpact = {json.dumps(env_impact)};
    const markerTimes = {json.dumps(marker_times)};
    const markerLabels = {json.dumps(marker_labels)};
    const humannessFix = {json.dumps(humanness_fix_annotations)};

    const customerTimes = [];
    const customerStress = [];
    const customerEngagement = [];
    const customerSpeed = [];
    const agentTimes = [];
    const agentHumanness = [];
    const agentTrendValue = [];

    for (let i = 0; i < times.length; i++) {{
      if (speakers[i] === 'customer' || speakers[i] === 'both') {{
        customerTimes.push(times[i]);
        customerStress.push(stress[i]);
        customerEngagement.push(engagement[i]);
        customerSpeed.push(speed[i]);
      }}
      if (speakers[i] === 'agent' || speakers[i] === 'both') {{
        agentTimes.push(times[i]);
        agentHumanness.push(humanness[i]);
        const trend = humannessTrend[i];
        agentTrendValue.push(trend === 'improving' ? 1 : (trend === 'degrading' ? -1 : 0));
      }}
    }}

    const traces = [
      {{
        x: customerTimes, y: customerStress, type:'scatter', mode:'lines+markers',
        name:'Kunde Stress', line:{{color:'#f59e0b', width:4}}
      }},
      {{
        x: customerTimes, y: customerEngagement, type:'scatter', mode:'lines+markers',
        name:'Kunde Engagement', line:{{color:'#fb923c', width:3, dash:'dot'}}
      }},
      {{
        x: agentTimes, y: agentHumanness, type:'scatter', mode:'lines+markers',
        name:'Agent Humanness', yaxis:'y2', line:{{color:'#2563eb', width:4}}
      }},
      {{
        x: customerTimes, y: customerSpeed, type:'scatter', mode:'lines',
        name:'Kunden-TTS Speed', line:{{color:'#b45309', width:2, dash:'dash'}}
      }},
      {{
        x: agentTimes, y: agentTrendValue, type:'scatter', mode:'lines',
        name:'Humanness Trend', yaxis:'y3', line:{{color:'#1d4ed8', width:2, dash:'dot'}}
      }}
    ];

    const shapes = [];
    const annotations = [];
    for (let i = 0; i < markerTimes.length; i++) {{
      shapes.push({{
        type:'line', x0:markerTimes[i], x1:markerTimes[i], y0:0, y1:1.05, yref:'y',
        line:{{color:'#111827', width:2, dash:'dash'}}
      }});
      annotations.push({{
        x: markerTimes[i], y: 1.02, yref:'paper', showarrow:false,
        text: 'strategy: ' + markerLabels[i], font: {{size:11, color:'#111827'}}
      }});
    }}

    for (const row of humannessFix) {{
      annotations.push({{
        x: row[0], y: row[1], yref:'y2', showarrow:true, arrowhead:2, ax:0, ay:-35,
        text: 'Humanness ' + row[1].toFixed(0) + ' -> TTS Fix Pitch +' + row[2].toFixed(2) + ' (' + row[3] + ')',
        font: {{size:10, color:'#1e3a8a'}},
        bgcolor:'rgba(219,234,254,0.75)',
      }});
    }}

    for (let i = 0; i < times.length; i++) {{
      if (!envDetected[i]) continue;
      const prev = i > 0 ? envDetected[i - 1] : '';
      if (i > 0 && envDetected[i] === prev) continue;
      const impact = envImpact[i] || {{}};
      const tags = [];
      if (impact.use_short_sentences) tags.push('kurze Saetze');
      if (impact.avoid_personal_questions) tags.push('keine sensiblen Fragen');
      if (impact.offer_callback) tags.push('Rueckruf anbieten');
      annotations.push({{
        x: times[i], y: 0.02, yref:'paper', showarrow:false,
        text: 'Env: ' + envDetected[i] + ' (' + envNoise[i] + ') - ' + tags.join(', '),
        font: {{size:10, color:'#065f46'}},
        bgcolor:'rgba(209,250,229,0.85)',
      }});
    }}

    const layout = {{
      paper_bgcolor:'#ffffff',
      plot_bgcolor:'#ffffff',
      margin:{{l:72,r:72,t:20,b:60}},
      xaxis:{{title:'Zeit (Sekunden)', gridcolor:'#edf0f5', zeroline:false}},
      yaxis:{{title:'Kunde (Stress/Engagement/TTS Speed)', range:[0,1.2], gridcolor:'#edf0f5', zeroline:false}},
      yaxis2:{{title:'Humanness Score', overlaying:'y', side:'right', range:[0,100], zeroline:false}},
      yaxis3:{{title:'Trend', overlaying:'y', side:'right', anchor:'free', position:1.0, range:[-1.2,1.2], showgrid:false, tickvals:[-1,0,1], ticktext:['degrading','stable','improving']}},
      legend:{{orientation:'h', y:1.08, x:1, xanchor:'right'}},
      shapes,
      annotations
    }};

    Plotly.newPlot('chart', traces, layout, {{
      displaylogo:false,
      responsive:true,
      toImageButtonOptions:{{format:'png', filename:'streaming_demo', scale:2}}
    }});

    const envColor = {{
      home_quiet:'#10b981',
      home_noisy:'#f59e0b',
      office_quiet:'#3b82f6',
      office_open:'#1d4ed8',
      car:'#ef4444',
      outdoor_street:'#f97316',
      outdoor_nature:'#22c55e',
      public_space:'#a855f7',
      transit:'#e11d48'
    }};
    const envBars = [];
    for (let i = 0; i < times.length; i++) {{
      if (!envDetected[i]) continue;
      envBars.push({{
        x:[times[i]],
        y:[envConfidence[i]],
        type:'bar',
        marker:{{color: envColor[envDetected[i]] || '#6b7280'}},
        name: envDetected[i],
        hovertemplate:'%{{x:.1f}}s<br>' + envDetected[i] + '<br>confidence=%{{y:.2f}}<extra></extra>',
        showlegend:false
      }});
    }}
    Plotly.newPlot('envChart', envBars, {{
      title:'Ohr 3: Environment Detection',
      xaxis:{{title:'Zeit (Sekunden)'}},
      yaxis:{{title:'Confidence', range:[0,1]}},
      paper_bgcolor:'#fff',
      plot_bgcolor:'#fff',
      margin:{{l:60,r:20,t:45,b:55}}
    }}, {{displaylogo:false, responsive:true}});
  </script>
</body>
</html>
"""
    REPORT_HTML.write_text(html, encoding="utf-8")


def _build_realtime_v2(rows: list[dict], source_call: Path) -> None:
    timeline = []
    transcript_entries = []
    prev_text = ""
    for r in rows:
        ts = float(r.get("timestamp", 0.0) or 0.0)
        speaker = str(r.get("speaker", "silence"))
        stress = float((r.get("customer_state") or {}).get("stress", np.nan))
        hum = float((r.get("agent_state") or {}).get("humanness_score", np.nan))
        env = str((r.get("environment") or {}).get("detected", ""))
        text = str(r.get("text", "") or "").strip()
        timeline.append(
            {
                "t": ts,
                "speaker": speaker,
                "stress": None if np.isnan(stress) else stress,
                "humanness": None if np.isnan(hum) else hum,
                "environment": env,
                "strategy": str((r.get("recommendation") or {}).get("strategy", "")),
                "alert": str(r.get("alert", "none")),
            }
        )
        if text and text != prev_text:
            prev_text = text
            transcript_entries.append(
                {
                    "t": ts,
                    "speaker": "customer" if speaker in {"customer", "both"} else "agent",
                    "text": text,
                    "stress": None if np.isnan(stress) else stress,
                    "humanness": None if np.isnan(hum) else hum,
                }
            )

    keyword_input = []
    for row in transcript_entries:
        keyword_input.append(
            {
                "start": row["t"],
                "text": row["text"],
                "paralinguistic_summary": {"stress_level": float(row["stress"] or 0.0)},
            }
        )
    ke = KeywordEmotionAnalyzer().analyze(keyword_input)
    trigger_phrases = [x["text"] for x in ke.get("trigger_phrases", [])]

    payload = {"timeline": timeline, "transcript": transcript_entries, "triggers": trigger_phrases, "source": source_call.name}
    html = f"""<!doctype html>
<html lang="de">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Orpheus Realtime Demo v2</title>
  <style>
    body {{ margin:0; font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:#f3f6fb; color:#111827; }}
    .wrap {{ max-width:1200px; margin:20px auto; padding:0 16px; display:grid; gap:12px; }}
    .top {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; }}
    .card {{ background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:12px; }}
    .big {{ font-size:38px; font-weight:700; }}
    .muted {{ color:#6b7280; font-size:12px; }}
    .timeline {{ height:140px; border:1px solid #e5e7eb; border-radius:10px; position:relative; background:#fff; overflow:hidden; }}
    .line {{ position:absolute; top:0; bottom:0; width:2px; background:#111827; left:0; }}
    .bar {{ position:absolute; top:0; bottom:0; opacity:.65; }}
    .agent {{ background:#2563eb; }}
    .customer {{ background:#fb923c; }}
    .both {{ background:linear-gradient(90deg,#2563eb,#fb923c); }}
    .silence {{ background:#9ca3af; }}
    .controls {{ display:flex; gap:8px; }}
    button {{ padding:8px 10px; border:1px solid #d1d5db; background:#fff; border-radius:8px; cursor:pointer; font-weight:600; }}
    button.active {{ background:#111827; color:#fff; }}
    .transcript {{ max-height:340px; overflow:auto; border:1px solid #e5e7eb; border-radius:10px; background:#fff; padding:10px; }}
    .row {{ padding:8px 10px; border-radius:8px; margin-bottom:6px; }}
    .sp-agent {{ border-left:4px solid #2563eb; }}
    .sp-customer {{ border-left:4px solid #fb923c; }}
    .stress-hot {{ background:#fee2e2; }}
    .hum-low {{ background:#fef9c3; }}
    .trigger {{ font-weight:700; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card"><h2>Orpheus Realtime v2</h2><div class="muted">Quelle: {source_call.name}</div></div>
    <div class="top">
      <div class="card"><div class="muted">Ohr 1 Stress</div><div id="stressNow" class="big">0.00</div></div>
      <div class="card"><div class="muted">Ohr 2 Humanness</div><div id="humNow" class="big">0.0</div></div>
      <div class="card"><div class="muted">Ohr 3 Environment</div><div id="envNow" class="big" style="font-size:24px">-</div></div>
    </div>
    <div class="card">
      <div class="timeline" id="timeline"><div class="line" id="cursor"></div></div>
      <div class="controls">
        <button id="playBtn">Play</button>
        <button class="sp active" data-sp="1">1x</button>
        <button class="sp" data-sp="2">2x</button>
        <button class="sp" data-sp="4">4x</button>
      </div>
    </div>
    <div class="card">
      <h3>Synchrones Transkript</h3>
      <div class="transcript" id="transcript"></div>
    </div>
  </div>
  <script>
    const DATA = {json.dumps(payload)};
    const N = DATA.timeline.length;
    const cursor = document.getElementById('cursor');
    const tl = document.getElementById('timeline');
    const transcriptEl = document.getElementById('transcript');
    const stressEl = document.getElementById('stressNow');
    const humEl = document.getElementById('humNow');
    const envEl = document.getElementById('envNow');
    const playBtn = document.getElementById('playBtn');
    let idx = -1, speed = 1, playing = false, timer = null;

    function bars() {{
      for (let i=0;i<N;i++) {{
        const b = document.createElement('div');
        b.className = 'bar ' + (DATA.timeline[i].speaker || 'silence');
        b.style.left = (i * 100 / N) + '%';
        b.style.width = (100 / N) + '%';
        tl.appendChild(b);
      }}
    }}

    function markTriggers(text) {{
      let out = text;
      for (const t of DATA.triggers) {{
        if (!t) continue;
        const re = new RegExp('\\\\b' + t.replace(/[.*+?^${{}}()|[\\]\\\\]/g, '\\\\$&') + '\\\\b', 'ig');
        out = out.replace(re, '<span class="trigger">$&</span>');
      }}
      return out;
    }}

    function addTranscript(t) {{
      const row = document.createElement('div');
      row.className = 'row ' + (t.speaker === 'agent' ? 'sp-agent' : 'sp-customer');
      if ((t.stress || 0) > 0.75) row.classList.add('stress-hot');
      if (t.speaker === 'agent' && (t.humanness || 100) < 50) row.classList.add('hum-low');
      row.innerHTML = '[' + t.t.toFixed(1) + 's] [' + t.speaker.toUpperCase() + '] "' + markTriggers(t.text) + '"';
      transcriptEl.appendChild(row);
      transcriptEl.scrollTop = transcriptEl.scrollHeight;
    }}

    function step() {{
      if (!playing) return;
      idx += 1;
      if (idx >= N) {{ playing = false; playBtn.textContent = 'Play'; return; }}
      const d = DATA.timeline[idx];
      stressEl.textContent = ((d.stress ?? 0)).toFixed(2);
      humEl.textContent = ((d.humanness ?? 0)).toFixed(1);
      envEl.textContent = d.environment || '-';
      cursor.style.left = ((idx/(N-1))*100).toFixed(2)+'%';
      for (const t of DATA.transcript) {{
        if (!t._shown && t.t <= d.t) {{ t._shown = true; addTranscript(t); }}
      }}
      timer = setTimeout(step, Math.max(40, 200 / speed));
    }}

    playBtn.addEventListener('click', () => {{
      if (playing) {{ playing = false; clearTimeout(timer); playBtn.textContent = 'Play'; return; }}
      if (idx >= N-1) {{
        idx = -1;
        transcriptEl.innerHTML = '';
        for (const t of DATA.transcript) t._shown = false;
      }}
      playing = true; playBtn.textContent = 'Pause'; step();
    }});
    document.querySelectorAll('.sp').forEach(btn => btn.addEventListener('click', () => {{
      document.querySelectorAll('.sp').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      speed = Number(btn.dataset.sp || '1');
    }}));

    bars();
  </script>
</body>
</html>
"""
    REPORT_REALTIME_V2.write_text(html, encoding="utf-8")


def main() -> None:
    call_path = _pick_call()
    audio, sr = _load_audio(call_path)
    if sr != TARGET_SR:
        audio = _resample(audio, sr, TARGET_SR)
    max_samples = int(MAX_DEMO_SECONDS * TARGET_SR)
    audio = audio[:max_samples]

    client = TestClient(app)
    rows: list[dict] = []

    with client.websocket_connect("/v1/stream?speaker_id=demo_stream_user&transcribe=true") as ws:
        for start in range(0, len(audio), CHUNK_SAMPLES):
            chunk = audio[start : start + CHUNK_SAMPLES]
            if len(chunk) == 0:
                continue
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
            ws.send_bytes(_to_wav_bytes(chunk))
            response = ws.receive_json()
            if "error" in response:
                print(f"[WARN] {response['error']}")
                continue
            rows.append(response)

    if not rows:
        raise SystemExit("Keine Streaming-Responses erhalten.")

    _build_report(rows, call_path)
    _build_realtime_v2(rows, call_path)
    print(f"Quelle: {call_path}")
    print(f"Chunks verarbeitet: {len(rows)}")
    print(f"Report: {REPORT_HTML}")
    print(f"Realtime v2: {REPORT_REALTIME_V2}")


if __name__ == "__main__":
    main()
