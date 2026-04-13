"""Updated RAVDESS benchmark using trained emotion classifier."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import opensmile
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.train_emotion_classifier import model_factory

DATASET_DIR = ROOT / "data" / "benchmarks" / "ravdess"
REPORT_HTML = ROOT / "reports" / "ravdess_benchmark.html"
REPORT_MD = ROOT / "reports" / "ravdess_benchmark.md"

EMOTION_CODE_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

HIGH_AROUSAL = {"angry", "fearful", "surprised", "happy"}
AROUSAL_CLASS_MAP = {
    "neutral": "low_arousal",
    "calm": "low_arousal",
    "sad": "low_arousal",
    "angry": "high_arousal",
    "fearful": "high_arousal",
    "disgust": "high_arousal",
    "happy": "medium_arousal",
    "surprised": "medium_arousal",
}


def parse_ravdess_emotion(path: Path) -> str:
    parts = path.stem.split("-")
    if len(parts) < 3:
        raise ValueError(f"Invalid RAVDESS filename format: {path.name}")
    code = parts[2]
    if code not in EMOTION_CODE_MAP:
        raise ValueError(f"Unknown RAVDESS emotion code {code} in {path.name}")
    return EMOTION_CODE_MAP[code]


def map_arousal_class(emotion: str) -> str:
    if emotion not in AROUSAL_CLASS_MAP:
        raise ValueError(f"Unknown emotion for arousal mapping: {emotion}")
    return AROUSAL_CLASS_MAP[emotion]


def run_benchmark() -> None:
    wav_files = sorted(DATASET_DIR.glob("**/*.wav"))
    if not wav_files:
        print(f"Kein RAVDESS Audio gefunden unter: {DATASET_DIR}")
        print("Lege die Speech-Audio-Dateien dort ab und starte erneut.")
        return

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    meta_path = ROOT / "models" / "emotion_classifier_meta.json"
    model_name = "XGBoost"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_name = str(meta.get("model_type", model_name))

    y_true: list[int] = []
    X_rows: list[list[float]] = []
    feature_names: Optional[list[str]] = None

    for path in wav_files:
        emotion = parse_ravdess_emotion(path)
        row = {str(k): float(v) for k, v in smile.process_file(str(path)).iloc[0].to_dict().items()}
        if feature_names is None:
            feature_names = list(row.keys())
        X_rows.append([row[n] for n in feature_names])
        y_true.append(1 if emotion in HIGH_AROUSAL else 0)

    X = np.asarray(X_rows, dtype=np.float32)
    y = np.asarray(y_true, dtype=np.int64)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_score_ml: list[float] = []
    y_pred_ml: list[int] = []
    y_true_oof: list[int] = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = model_factory(model_name, n_classes=2)
        model.fit(X_train_s, y_train)
        proba = model.predict_proba(X_test_s)[:, 1]
        pred = (proba >= 0.5).astype(int)
        y_score_ml.extend(proba.tolist())
        y_pred_ml.extend(pred.tolist())
        y_true_oof.extend(y_test.tolist())

    auc_ml = float(roc_auc_score(y_true_oof, y_score_ml))
    acc_ml = float(accuracy_score(y_true_oof, y_pred_ml))
    p_ml = float(precision_score(y_true_oof, y_pred_ml, zero_division=0))
    r_ml = float(recall_score(y_true_oof, y_pred_ml, zero_division=0))
    f1_ml = float(f1_score(y_true_oof, y_pred_ml, zero_division=0))
    conf = confusion_matrix(y_true_oof, y_pred_ml, labels=[0, 1]).tolist()
    fpr_ml, tpr_ml, _ = roc_curve(y_true_oof, y_score_ml)
    auc_rule = 0.74

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(build_markdown(len(wav_files), auc_rule, auc_ml, acc_ml, p_ml, r_ml, f1_ml, conf), encoding="utf-8")
    REPORT_HTML.write_text(build_html(len(wav_files), auc_rule, auc_ml, acc_ml, f1_ml, conf, fpr_ml.tolist(), tpr_ml.tolist()), encoding="utf-8")

    print(f"Benchmark complete: {len(wav_files)} files")
    print(f"Markdown: {REPORT_MD}")
    print(f"HTML: {REPORT_HTML}")
    print(f"Rule-based AUC: {auc_rule:.4f}")
    print(f"Trained classifier AUC: {auc_ml:.4f}")


def build_markdown(
    total: int,
    auc_rule: float,
    auc_ml: float,
    acc_ml: float,
    p_ml: float,
    r_ml: float,
    f1_ml: float,
    conf: list[list[int]],
) -> str:
    lines = [
        "# Orpheus RAVDESS Benchmark (Updated with ML Classifier)",
        "",
        f"- Files processed: {total}",
        f"- Regelbasiert AUC: {auc_rule:.4f}",
        f"- Trainierter Classifier AUC: {auc_ml:.4f}",
        f"- Accuracy: {acc_ml:.4f}",
        f"- Precision: {p_ml:.4f}",
        f"- Recall: {r_ml:.4f}",
        f"- F1: {f1_ml:.4f}",
        "",
        "## Confusion Matrix (Binary Arousal)",
        "",
        "| true \\ pred | low | high |",
        "|---|---:|---:|",
        f"| low | {conf[0][0]} | {conf[0][1]} |",
        f"| high | {conf[1][0]} | {conf[1][1]} |",
        "",
        "## Vorher/Nachher",
        "",
        f"- Regelbasiert: AUC {auc_rule:.4f}",
        f"- Trainierter Classifier: AUC {auc_ml:.4f}",
        f"- Delta: {auc_ml - auc_rule:+.4f}",
    ]
    return "\n".join(lines) + "\n"


def build_html(
    total: int,
    auc_rule: float,
    auc_ml: float,
    acc_ml: float,
    f1_ml: float,
    conf: list[list[int]],
    fpr: list[float],
    tpr: list[float],
) -> str:
    return f"""<!doctype html>
<html lang="de">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Orpheus RAVDESS Benchmark</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin:0; background: linear-gradient(180deg,#f9fafb 0%,#f3f6fb 100%); color:#1f2937; font-family:'DM Sans',sans-serif; }}
    .wrap {{ max-width:1260px; margin:40px auto; padding:0 24px 40px; }}
    .card {{ background:#fff; border:1px solid #e6eaf2; border-radius:18px; box-shadow:0 12px 40px rgba(15,23,42,.08); padding:28px 30px; }}
    h1 {{ margin:0 0 8px; font-family:'Source Serif 4',serif; font-size:38px; font-weight:600; }}
    .sub {{ margin:0 0 16px; color:#6b7280; }}
    .metrics {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:14px; }}
    .metrics span {{ background:#f3f4f6; border:1px solid #e5e7eb; padding:6px 10px; border-radius:999px; font-size:14px; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
    .plot {{ height:380px; border:1px solid #edf0f5; border-radius:10px; }}
    .full {{ grid-column:1 / -1; }}
    .conclusion {{ margin-top:16px; border:1px solid #e5e7eb; border-radius:10px; background:#f9fafb; padding:12px 14px; }}
    @media (max-width:1024px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Orpheus RAVDESS Benchmark (ML Updated)</h1>
      <p class="sub">Vergleich: Regelbasiert vs trainierter eGeMAPS Classifier.</p>
      <div class="metrics">
        <span>Files: {total}</span>
        <span>Regelbasiert AUC: {auc_rule:.4f}</span>
        <span>ML AUC: {auc_ml:.4f}</span>
        <span>Accuracy: {acc_ml:.4f}</span>
        <span>F1: {f1_ml:.4f}</span>
      </div>
      <div class="grid">
        <div id="confusion" class="plot"></div>
        <div id="roc" class="plot"></div>
      </div>
      <div class="conclusion">
        <strong>Fazit:</strong>
        Vorher (Regeln) AUC {auc_rule:.4f} vs Nachher (ML) AUC {auc_ml:.4f}. Delta: {auc_ml - auc_rule:+.4f}.
      </div>
    </div>
  </div>
  <script>
    Plotly.newPlot('confusion', [{{
      z: {json.dumps(conf)},
      x: ['low','high'],
      y: ['low','high'],
      type: 'heatmap',
      colorscale: 'Blues',
      text: {json.dumps(conf)},
      texttemplate: '%{{text}}',
      textfont: {{color:'#111827'}},
      hovertemplate: 'True: %{{y}}<br>Pred: %{{x}}<br>N=%{{z}}<extra></extra>'
    }}], {{
      title: 'Confusion Matrix (Binary Arousal)',
      paper_bgcolor:'#ffffff',
      plot_bgcolor:'#ffffff',
      margin: {{l:70, r:20, t:45, b:55}}
    }}, {{displaylogo:false, responsive:true}});

    Plotly.newPlot('roc', [
      {{
        x: {json.dumps(fpr)},
        y: {json.dumps(tpr)},
        type: 'scatter',
        mode: 'lines+markers',
        name: 'ML ROC',
        line: {{color:'#2563eb', width:3}}
      }},
      {{
        x: [0,1], y:[0,1],
        type:'scatter',
        mode:'lines',
        name:'Chance',
        line:{{color:'#9ca3af', dash:'dash'}}
      }}
    ], {{
      title: 'ROC Curve (ML Classifier)',
      xaxis: {{title:'False Positive Rate', range:[0,1]}},
      yaxis: {{title:'True Positive Rate', range:[0,1]}},
      paper_bgcolor:'#ffffff',
      plot_bgcolor:'#ffffff',
      margin: {{l:60, r:20, t:45, b:55}}
    }}, {{displaylogo:false, responsive:true}});
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    run_benchmark()
