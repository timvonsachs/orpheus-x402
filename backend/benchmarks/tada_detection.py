"""Benchmark Orpheus Ohr 2 (MelCNN) against Hume TADA TTS."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch

from config import settings
from engine.diarization import SpeakerDiarizer
from engine.humanness import HumannessEngine
from engine.preprocessing import AudioPreprocessor

TADA_MODEL = "HumeAI/tada-1b"
TADA_DIR = ROOT / "data" / "benchmarks" / "tada_samples"
HUMAN_DIR = ROOT / "data" / "benchmarks" / "human_samples"
RAW_RESULTS_JSON = ROOT / "data" / "benchmarks" / "tada_benchmark_results.json"
REPORT_MD = ROOT / "reports" / "tada_benchmark.md"
REPORT_HTML = ROOT / "reports" / "tada_benchmark.html"
MANIFEST = ROOT / "data" / "real_calls" / "manifest.csv"
RAVDESS_DIR = ROOT / "data" / "benchmarks" / "ravdess"

TARGET_SR = 16000
THRESHOLD = 50.0
N_SAMPLES = 10

TADA_TEXTS = [
    "Hello, I'm calling about your recent order.",
    "I understand your concern, let me help you with that.",
    "The delivery is scheduled for tomorrow between 2 and 4 PM.",
    "Would you like me to transfer you to a specialist?",
    "Thank you for your patience, I really appreciate it.",
    "I'm sorry to hear that. Let me look into this right away.",
    "Your refund has been processed and should arrive within 3 days.",
    "Is there anything else I can help you with today?",
    "I want to make sure we resolve this completely for you.",
    "Great, I've updated your account with the new information.",
]

BLOG_TADA_MP3_URLS = [
    "https://cdn.sanity.io/files/xqnc2for/production/2ecc5f2755db823186119b34cc9cccb02af4feb4.mp3",
    "https://cdn.sanity.io/files/xqnc2for/production/61ded5093a0d738ea1a925aed5223c0685267ed0.mp3",
    "https://cdn.sanity.io/files/xqnc2for/production/70641df7a8908c7baab1fd0d62bcbf466f1c4347.mp3",
    "https://cdn.sanity.io/files/xqnc2for/production/768f2e749c43277aa0f80a8fd5aceaad208deb8a.mp3",
]


def roc_auc_score(y_true: list[int], y_score: list[float]) -> float:
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    rank_sum_pos = 0.0
    idx = 0
    while idx < len(pairs):
        j = idx
        while j < len(pairs) and pairs[j][0] == pairs[idx][0]:
            j += 1
        avg_rank = (idx + 1 + j) / 2.0
        for k in range(idx, j):
            if pairs[k][1] == 1:
                rank_sum_pos += avg_rank
        idx = j
    return float((rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg))


def normalize_wav_16k_mono(src: Path, dst: Path, pre: AudioPreprocessor) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = Path(pre.process(str(src), str(dst)))
    return out


def maybe_download_real_call_audio() -> list[Path]:
    if not MANIFEST.exists():
        return []
    rows = MANIFEST.read_text(encoding="utf-8").splitlines()[1:]
    available: list[Path] = []
    for row in rows:
        parts = row.split(",")
        if len(parts) < 3:
            continue
        file_path = ROOT / parts[0]
        url = parts[2]
        if file_path.exists():
            available.append(file_path)
            continue
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if download_audio_from_youtube(url=url, out_path=file_path):
            available.append(file_path)
        if len(available) >= 3:
            break
    return available


def download_audio_from_youtube(url: str, out_path: Path) -> bool:
    cmds: list[list[str]] = []
    if shutil.which("yt-dlp"):
        cmds.append(
            [
                "yt-dlp",
                "-x",
                "--audio-format",
                "wav",
                "--audio-quality",
                "0",
                "-o",
                str(out_path.with_suffix(".%(ext)s")),
                url,
            ]
        )
    cmds.append(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            "-m",
            "yt_dlp",
            "-x",
            "--audio-format",
            "wav",
            "--audio-quality",
            "0",
            "-o",
            str(out_path.with_suffix(".%(ext)s")),
            url,
        ]
    )

    for cmd in cmds:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and out_path.exists():
            return True
    return False


def extract_human_segments() -> tuple[list[Path], str]:
    pre = AudioPreprocessor()
    HUMAN_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(HUMAN_DIR.glob("*.wav"))
    if len(existing) >= N_SAMPLES:
        return existing[:N_SAMPLES], "preexisting"

    source_audio = maybe_download_real_call_audio()
    diarizer = SpeakerDiarizer(min_segment_duration=0.8)
    created: list[Path] = []

    for audio_path in source_audio:
        normalized = normalize_wav_16k_mono(audio_path, audio_path.with_name(f"{audio_path.stem}_16k.wav"), pre)
        waveform, sr = sf.read(str(normalized), dtype="float32", always_2d=True)
        mono = waveform.mean(axis=1)
        diarized = diarizer.diarize(str(normalized))

        for seg in diarized:
            if seg.get("speaker") == "silence":
                continue
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            dur = end - start
            if dur < 3.0:
                continue
            clip_len = min(5.0, dur)
            center = (start + end) / 2.0
            c_start = max(0.0, center - clip_len / 2.0)
            c_end = min(len(mono) / sr, c_start + clip_len)
            s0 = int(c_start * sr)
            s1 = int(c_end * sr)
            if s1 - s0 < int(3.0 * sr):
                continue
            clip = mono[s0:s1]
            out = HUMAN_DIR / f"human_{len(existing) + len(created) + 1:02d}.wav"
            sf.write(str(out), clip, sr, subtype="PCM_16")
            created.append(out)
            if len(existing) + len(created) >= N_SAMPLES:
                break
        if len(existing) + len(created) >= N_SAMPLES:
            break

    current = sorted(HUMAN_DIR.glob("*.wav"))
    if len(current) >= N_SAMPLES:
        return current[:N_SAMPLES], "real_calls"

    # Honest fallback if local real-call audio is unavailable.
    ravdess = sorted(RAVDESS_DIR.glob("**/*.wav"))
    for src in ravdess:
        out = HUMAN_DIR / f"human_{len(current) + 1:02d}.wav"
        normalize_wav_16k_mono(src, out, pre)
        current = sorted(HUMAN_DIR.glob("*.wav"))
        if len(current) >= N_SAMPLES:
            return current[:N_SAMPLES], "ravdess_fallback"
    return current[:N_SAMPLES], "insufficient"


def generate_tada_samples() -> tuple[list[Path], str, list[str]]:
    pre = AudioPreprocessor()
    TADA_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(TADA_DIR.glob("*.wav"))
    if len(existing) >= N_SAMPLES:
        return existing[:N_SAMPLES], "preexisting", []

    errors: list[str] = []
    mode = "failed"
    ok_local = _generate_tada_local(pre, errors)
    if ok_local:
        mode = "local"
    if not ok_local:
        if _generate_tada_inference(pre, errors):
            mode = "inference"
    generated = sorted(TADA_DIR.glob("*.wav"))
    if len(generated) < N_SAMPLES:
        if _generate_tada_from_blog_audio(pre, errors):
            mode = "blog_demo_fallback"

    generated = sorted(TADA_DIR.glob("*.wav"))
    return generated[:N_SAMPLES], (mode if len(generated) >= N_SAMPLES else "failed"), errors


def _generate_tada_local(pre: AudioPreprocessor, errors: list[str]) -> bool:
    try:
        from transformers import pipeline

        synth = pipeline("text-to-audio", model=TADA_MODEL, device=-1)
        for i, text in enumerate(TADA_TEXTS, start=1):
            result = synth(text)
            audio = np.asarray(result["audio"], dtype=np.float32)
            sr = int(result.get("sampling_rate", 24000))
            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            tmp = TADA_DIR / f"tmp_tada_{i:02d}.wav"
            sf.write(str(tmp), audio, sr, subtype="PCM_16")
            normalize_wav_16k_mono(tmp, TADA_DIR / f"tada_{i:02d}.wav", pre)
            tmp.unlink(missing_ok=True)
        return True
    except Exception as exc:
        errors.append(f"local_generation_failed: {exc}")
        return False


def _generate_tada_inference(pre: AudioPreprocessor, errors: list[str]) -> bool:
    try:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=os.getenv("HF_TOKEN"))
        for i, text in enumerate(TADA_TEXTS, start=1):
            wav_bytes = client.text_to_speech(text=text, model=TADA_MODEL)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(wav_bytes)
            normalize_wav_16k_mono(tmp_path, TADA_DIR / f"tada_{i:02d}.wav", pre)
            tmp_path.unlink(missing_ok=True)
        return True
    except Exception as exc:
        errors.append(f"inference_generation_failed: {exc}")
        return False


def _generate_tada_from_blog_audio(pre: AudioPreprocessor, errors: list[str]) -> bool:
    try:
        import requests

        normalized_sources: list[Path] = []
        for i, url in enumerate(BLOG_TADA_MP3_URLS, start=1):
            mp3_path = TADA_DIR / f"tada_blog_{i:02d}.mp3"
            if not mp3_path.exists():
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                mp3_path.write_bytes(resp.content)
            wav_path = TADA_DIR / f"tada_blog_{i:02d}.wav"
            normalize_wav_16k_mono(mp3_path, wav_path, pre)
            normalized_sources.append(wav_path)

        clips_written = 0
        for src in normalized_sources:
            audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)
            total_s = len(mono) / float(sr)
            win = 4.0
            step = 2.5
            t0 = 0.0
            while t0 + win <= total_s + 1e-6:
                out = TADA_DIR / f"tada_{clips_written + 1:02d}.wav"
                if not out.exists():
                    s0 = int(t0 * sr)
                    s1 = int((t0 + win) * sr)
                    sf.write(str(out), mono[s0:s1], sr, subtype="PCM_16")
                clips_written = len(sorted(TADA_DIR.glob("tada_*.wav")))
                if clips_written >= N_SAMPLES:
                    return True
                t0 += step
        return clips_written >= N_SAMPLES
    except Exception as exc:
        errors.append(f"blog_audio_fallback_failed: {exc}")
        return False


def load_humanness_engine() -> tuple[Optional[HumannessEngine], str]:
    ckpt = Path(settings.model_checkpoint)
    if not ckpt.exists():
        alt = ROOT / settings.model_checkpoint
        ckpt = alt if alt.exists() else ckpt
    try:
        engine = HumannessEngine(checkpoint_path=str(ckpt), device=settings.device)
        return engine, "melcnn"
    except Exception as exc:
        return None, f"fallback ({exc})"


def fallback_score(path: Path) -> float:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    rms = float(np.sqrt(np.mean(mono**2) + 1e-9))
    zcr = float(np.mean(np.abs(np.diff(np.sign(mono))) > 0)) if len(mono) > 1 else 0.0
    score = 70.0
    if rms < 0.02:
        score -= 15.0
    if 0.05 <= zcr <= 0.12:
        score -= 10.0
    return max(5.0, min(95.0, score))


def score_file(path: Path, engine: Optional[HumannessEngine]) -> float:
    if engine is None:
        return fallback_score(path)
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    tensor = torch.from_numpy(wav.T.copy())
    if tensor.shape[0] > 1:
        tensor = tensor.mean(dim=0, keepdim=True)
    out = engine.score_audio(tensor, sr)
    return float(out.get("score", 0.0) or 0.0)


def confusion(y_true: list[int], y_pred: list[int]) -> list[list[int]]:
    # rows: true [synthetic(0), human(1)], cols: pred [synthetic(0), human(1)]
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    return [[tn, fp], [fn, tp]]


def build_reports(payload: dict[str, Any]) -> None:
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(build_markdown(payload), encoding="utf-8")
    REPORT_HTML.write_text(build_html(payload), encoding="utf-8")


def build_markdown(p: dict[str, Any]) -> str:
    rows = p["rows"]
    lines = [
        "# Orpheus vs. Hume TADA: Kann Ohr 2 State-of-the-Art TTS erkennen?",
        "",
        f"- TADA Durchschnitt: **{p['tada_mean']:.2f}/100**",
        f"- Human Durchschnitt: **{p['human_mean']:.2f}/100**",
        f"- AUC: **{p['auc']:.4f}**",
        f"- Accuracy @50: **{p['accuracy']:.4f}**",
        f"- Korrekt erkannt: **{p['tada_synth_detected']}/10 TADA als synthetic**, **{p['human_human_detected']}/10 Human als human**",
        f"- Human-Quelle: `{p['human_source']}`",
        f"- TADA-Erzeugung: `{p['tada_generation_mode']}`",
        f"- Engine-Modus: `{p['engine_mode']}`",
        "",
        "## Samples",
        "",
        "| Sample | Label | Humanness Score | Prediction |",
        "|---|---|---:|---|",
    ]
    for r in rows:
        lines.append(f"| {r['sample']} | {r['label']} | {r['score']:.2f} | {r['prediction']} |")
    lines.extend(["", "## Confusion Matrix", "", "| true \\\\ pred | synthetic | human |", "|---|---:|---:|"])
    conf = p["confusion"]
    lines.append(f"| synthetic | {conf[0][0]} | {conf[0][1]} |")
    lines.append(f"| human | {conf[1][0]} | {conf[1][1]} |")
    lines.extend(["", "## Fazit", "", p["conclusion"], ""])
    if p["generation_errors"]:
        lines.extend(["## Hinweise/Fehler bei TADA-Erzeugung", ""])
        for e in p["generation_errors"]:
            lines.append(f"- {e}")
        lines.append("")
    return "\n".join(lines)


def build_html(p: dict[str, Any]) -> str:
    rows = p["rows"]
    tada_scores = [r["score"] for r in rows if r["label"] == "tada"]
    human_scores = [r["score"] for r in rows if r["label"] == "human"]
    names = [r["sample"] for r in rows]
    labels = [r["label"] for r in rows]
    scores = [r["score"] for r in rows]
    preds = [r["prediction"] for r in rows]
    return f"""<!doctype html>
<html lang="de">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Orpheus TADA Benchmark</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin:0; background:linear-gradient(180deg,#f9fafb 0%,#f3f6fb 100%); color:#1f2937; font-family:'DM Sans',sans-serif; }}
    .wrap {{ max-width:1260px; margin:40px auto; padding:0 24px 40px; }}
    .card {{ background:#fff; border:1px solid #e6eaf2; border-radius:18px; box-shadow:0 12px 40px rgba(15,23,42,.08); padding:28px 30px; }}
    h1 {{ margin:0 0 8px; font-family:'Source Serif 4',serif; font-size:38px; font-weight:600; }}
    .sub {{ margin:0 0 16px; color:#6b7280; }}
    .metrics {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:14px; }}
    .metrics span {{ background:#f3f4f6; border:1px solid #e5e7eb; padding:6px 10px; border-radius:999px; font-size:14px; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
    .plot {{ height:360px; border:1px solid #edf0f5; border-radius:10px; }}
    .full {{ grid-column:1 / -1; }}
    table {{ width:100%; border-collapse:collapse; margin-top:14px; font-size:14px; }}
    th,td {{ border-bottom:1px solid #edf0f5; text-align:left; padding:8px 10px; }}
    .conclusion {{ margin-top:16px; border:1px solid #e5e7eb; border-radius:10px; background:#f9fafb; padding:12px 14px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Orpheus vs. Hume TADA: Kann Ohr 2 State-of-the-Art TTS erkennen?</h1>
      <p class="sub">MelCNN Humanness Benchmark mit 10 TADA-Samples und 10 Human-Samples.</p>
      <div class="metrics">
        <span>TADA Durchschnitt: {p['tada_mean']:.2f}/100</span>
        <span>Human Durchschnitt: {p['human_mean']:.2f}/100</span>
        <span>AUC: {p['auc']:.4f}</span>
        <span>Accuracy @50: {p['accuracy']:.4f}</span>
        <span>TADA synthetic erkannt: {p['tada_synth_detected']}/10</span>
        <span>Human human erkannt: {p['human_human_detected']}/10</span>
      </div>
      <div class="grid">
        <div id="bars" class="plot"></div>
        <div id="cm" class="plot"></div>
      </div>
      <table>
        <thead><tr><th>Sample</th><th>Label</th><th>Score</th><th>Prediction</th></tr></thead>
        <tbody id="rows"></tbody>
      </table>
      <div class="conclusion"><strong>Fazit:</strong> {p['conclusion']}</div>
    </div>
  </div>
  <script>
    const tadaScores = {json.dumps(tada_scores)};
    const humanScores = {json.dumps(human_scores)};
    const names = {json.dumps(names)};
    const labels = {json.dumps(labels)};
    const scores = {json.dumps(scores)};
    const preds = {json.dumps(preds)};
    const conf = {json.dumps(p['confusion'])};

    Plotly.newPlot('bars', [
      {{x: Array.from({{length:tadaScores.length}}, (_,i)=>`tada_${{i+1}}`), y: tadaScores, type:'bar', name:'TADA', marker:{{color:'#e53935'}}}},
      {{x: Array.from({{length:humanScores.length}}, (_,i)=>`human_${{i+1}}`), y: humanScores, type:'bar', name:'Human', marker:{{color:'#2563eb'}}}}
    ], {{
      barmode:'group',
      title:'Humanness Scores: TADA vs Human',
      yaxis:{{title:'Score (0-100)', range:[0,100]}},
      paper_bgcolor:'#fff', plot_bgcolor:'#fff', margin:{{l:60,r:20,t:45,b:55}}
    }}, {{displaylogo:false, responsive:true}});

    Plotly.newPlot('cm', [{{
      z: conf, x:['synthetic','human'], y:['synthetic','human'], type:'heatmap', colorscale:'Blues',
      text: conf, texttemplate:'%{{text}}', textfont:{{color:'#111827'}}
    }}], {{
      title:'Confusion Matrix',
      paper_bgcolor:'#fff', plot_bgcolor:'#fff', margin:{{l:60,r:20,t:45,b:55}}
    }}, {{displaylogo:false, responsive:true}});

    const body = document.getElementById('rows');
    for (let i = 0; i < names.length; i++) {{
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${{names[i]}}</td><td>${{labels[i]}}</td><td>${{scores[i].toFixed(2)}}</td><td>${{preds[i]}}</td>`;
      body.appendChild(tr);
    }}
  </script>
</body>
</html>
"""


def run() -> None:
    tada_files, tada_mode, generation_errors = generate_tada_samples()
    human_files, human_source = extract_human_segments()

    if len(tada_files) < N_SAMPLES:
        raise SystemExit(
            "Konnte nicht genug TADA-Samples erzeugen. "
            "Bitte nutze alternativ https://huggingface.co/spaces/HumeAI/tada und lege 10 WAVs in data/benchmarks/tada_samples/ ab."
        )
    if len(human_files) < N_SAMPLES:
        raise SystemExit("Konnte nicht genug Human-Samples extrahieren (mind. 10 benoetigt).")

    engine, engine_mode = load_humanness_engine()

    rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_score: list[float] = []
    y_pred: list[int] = []

    for path in tada_files[:N_SAMPLES]:
        score = score_file(path, engine)
        pred_human = 1 if score >= THRESHOLD else 0
        rows.append(
            {
                "sample": path.name,
                "label": "tada",
                "score": score,
                "prediction": "human" if pred_human else "synthetic",
            }
        )
        y_true.append(0)
        y_score.append(score)
        y_pred.append(pred_human)

    for path in human_files[:N_SAMPLES]:
        score = score_file(path, engine)
        pred_human = 1 if score >= THRESHOLD else 0
        rows.append(
            {
                "sample": path.name,
                "label": "human",
                "score": score,
                "prediction": "human" if pred_human else "synthetic",
            }
        )
        y_true.append(1)
        y_score.append(score)
        y_pred.append(pred_human)

    tada_scores = [r["score"] for r in rows if r["label"] == "tada"]
    human_scores = [r["score"] for r in rows if r["label"] == "human"]
    auc = roc_auc_score(y_true, y_score)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    conf = confusion(y_true, y_pred)
    tada_synth_detected = sum(1 for r in rows if r["label"] == "tada" and r["prediction"] == "synthetic")
    human_human_detected = sum(1 for r in rows if r["label"] == "human" and r["prediction"] == "human")

    if auc > 0.9:
        conclusion = "Orpheus erkennt TADA zuverlaessig. Ohr 2 funktioniert gegen State-of-the-Art."
    elif auc >= 0.7:
        conclusion = "Orpheus erkennt TADA teilweise. MelCNN muss auf neue TTS-Modelle nachtrainiert werden."
    else:
        conclusion = "TADA ueberlistet das aktuelle MelCNN. Fine-Tuning auf TADA-Daten ist notwendig."

    payload = {
        "rows": rows,
        "tada_mean": float(np.mean(tada_scores)),
        "human_mean": float(np.mean(human_scores)),
        "auc": float(auc),
        "accuracy": float(acc),
        "confusion": conf,
        "tada_synth_detected": int(tada_synth_detected),
        "human_human_detected": int(human_human_detected),
        "conclusion": conclusion,
        "human_source": human_source,
        "tada_generation_mode": tada_mode,
        "engine_mode": engine_mode,
        "generation_errors": generation_errors,
    }

    RAW_RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RAW_RESULTS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    build_reports(payload)

    print("=== TADA Benchmark Complete ===")
    print(f"TADA avg: {payload['tada_mean']:.2f}")
    print(f"Human avg: {payload['human_mean']:.2f}")
    print(f"AUC: {payload['auc']:.4f}")
    print(f"Accuracy@50: {payload['accuracy']:.4f}")
    print(f"TADA synthetic erkannt: {payload['tada_synth_detected']}/10")
    print(f"Human human erkannt: {payload['human_human_detected']}/10")
    print(f"Human source: {payload['human_source']}")
    print(f"TADA mode: {payload['tada_generation_mode']}")
    print(f"Engine mode: {payload['engine_mode']}")
    print(f"JSON: {RAW_RESULTS_JSON}")
    print(f"MD: {REPORT_MD}")
    print(f"HTML: {REPORT_HTML}")


if __name__ == "__main__":
    run()
