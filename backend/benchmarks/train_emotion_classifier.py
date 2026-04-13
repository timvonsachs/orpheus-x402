"""Train paper-grade emotion classifiers on RAVDESS using eGeMAPS."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import opensmile
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "data" / "benchmarks" / "ravdess"
REPORT_MD = ROOT / "reports" / "classifier_benchmark.md"
REPORT_HTML = ROOT / "reports" / "classifier_benchmark.html"
MODEL_PATH = ROOT / "models" / "emotion_classifier.joblib"
SCALER_PATH = ROOT / "models" / "feature_scaler.joblib"
META_PATH = ROOT / "models" / "emotion_classifier_meta.json"

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
RANDOM_STATE = 42


@dataclass
class ModelResult:
    name: str
    auc_mean: float
    auc_std: float
    acc_mean: float
    acc_std: float
    precision_mean: float
    precision_std: float
    recall_mean: float
    recall_std: float
    f1_mean: float
    f1_std: float
    oof_true: list[int]
    oof_proba: list[float] | list[list[float]]
    oof_pred: list[int]


def parse_emotion_from_filename(path: Path) -> str:
    parts = path.stem.split("-")
    if len(parts) < 3:
        raise ValueError(f"Invalid filename format: {path.name}")
    code = parts[2]
    if code not in EMOTION_CODE_MAP:
        raise ValueError(f"Unknown emotion code {code} in {path.name}")
    return EMOTION_CODE_MAP[code]


def extract_egemaps_features(path: Path, smile: Optional[opensmile.Smile] = None) -> dict[str, float]:
    smiler = smile or opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    df = smiler.process_file(str(path))
    row = df.iloc[0]
    return {str(k): float(v) for k, v in row.to_dict().items()}


def load_dataset() -> tuple[np.ndarray, list[str], list[str], list[str]]:
    wav_files = sorted(DATASET_DIR.glob("**/*.wav"))
    if not wav_files:
        raise RuntimeError(f"No RAVDESS wav files found under {DATASET_DIR}")

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    rows: list[list[float]] = []
    emotions: list[str] = []
    feature_names: Optional[list[str]] = None

    for p in wav_files:
        feats = extract_egemaps_features(p, smile=smile)
        if feature_names is None:
            feature_names = list(feats.keys())
        rows.append([float(feats[name]) for name in feature_names])
        emotions.append(parse_emotion_from_filename(p))

    if feature_names is None:
        raise RuntimeError("Feature extraction returned no rows")
    return np.asarray(rows, dtype=np.float32), emotions, feature_names, [p.name for p in wav_files]


def model_factory(name: str, n_classes: int) -> Any:
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1)
    if name == "XGBoost":
        if n_classes == 2:
            return XGBClassifier(
                random_state=RANDOM_STATE,
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=-1,
            )
        return XGBClassifier(
            random_state=RANDOM_STATE,
            n_estimators=350,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=-1,
        )
    if name == "SVM":
        return SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)
    raise ValueError(f"Unknown model name: {name}")


def evaluate_models(X: np.ndarray, y: np.ndarray, multiclass: bool) -> dict[str, ModelResult]:
    names = ["RandomForest", "XGBoost", "SVM", "LogisticRegression"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    n_classes = int(np.max(y)) + 1
    out: dict[str, ModelResult] = {}

    for name in names:
        auc_scores: list[float] = []
        acc_scores: list[float] = []
        p_scores: list[float] = []
        r_scores: list[float] = []
        f1_scores: list[float] = []
        oof_true: list[int] = []
        oof_pred: list[int] = []
        oof_proba: list[Any] = []

        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = model_factory(name, n_classes=n_classes)
            model.fit(X_train_s, y_train)
            proba = model.predict_proba(X_test_s)
            pred = np.argmax(proba, axis=1) if multiclass else (proba[:, 1] >= 0.5).astype(int)

            if multiclass:
                y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
                auc = roc_auc_score(y_test_bin, proba, average="macro", multi_class="ovr")
                precision = precision_score(y_test, pred, average="macro", zero_division=0)
                recall = recall_score(y_test, pred, average="macro", zero_division=0)
                f1 = f1_score(y_test, pred, average="macro", zero_division=0)
                oof_proba.extend(proba.tolist())
            else:
                auc = roc_auc_score(y_test, proba[:, 1])
                precision = precision_score(y_test, pred, average="binary", zero_division=0)
                recall = recall_score(y_test, pred, average="binary", zero_division=0)
                f1 = f1_score(y_test, pred, average="binary", zero_division=0)
                oof_proba.extend(proba[:, 1].tolist())

            acc = accuracy_score(y_test, pred)

            auc_scores.append(float(auc))
            acc_scores.append(float(acc))
            p_scores.append(float(precision))
            r_scores.append(float(recall))
            f1_scores.append(float(f1))
            oof_true.extend(y_test.tolist())
            oof_pred.extend(pred.tolist())

        out[name] = ModelResult(
            name=name,
            auc_mean=float(np.mean(auc_scores)),
            auc_std=float(np.std(auc_scores, ddof=1)),
            acc_mean=float(np.mean(acc_scores)),
            acc_std=float(np.std(acc_scores, ddof=1)),
            precision_mean=float(np.mean(p_scores)),
            precision_std=float(np.std(p_scores, ddof=1)),
            recall_mean=float(np.mean(r_scores)),
            recall_std=float(np.std(r_scores, ddof=1)),
            f1_mean=float(np.mean(f1_scores)),
            f1_std=float(np.std(f1_scores, ddof=1)),
            oof_true=oof_true,
            oof_proba=oof_proba,
            oof_pred=oof_pred,
        )
    return out


def top_feature_importance(model: Any, feature_names: list[str], top_k: int = 15) -> list[dict[str, float]]:
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=np.float64)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=np.float64)
        vals = np.mean(np.abs(coef), axis=0) if coef.ndim == 2 else np.abs(coef)
    else:
        vals = np.zeros(len(feature_names), dtype=np.float64)
    idx = np.argsort(vals)[::-1][:top_k]
    return [{"feature": feature_names[i], "importance": float(vals[i])} for i in idx]


def format_pm(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def build_markdown(
    binary_results: dict[str, ModelResult],
    multi_results: dict[str, ModelResult],
    best_name: str,
    best_auc: float,
    best_std: float,
    top_features: list[dict[str, float]],
    conf: list[list[int]],
    labels: list[str],
) -> str:
    lines = [
        "# Paper-Grade Emotion Classifier Benchmark (RAVDESS, eGeMAPS)",
        "",
        "## Binary Arousal (High vs Low)",
        "",
        "| Modell | AUC Mean±Std | Accuracy Mean±Std | F1 Mean±Std | Precision Mean±Std | Recall Mean±Std |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in binary_results.items():
        lines.append(
            f"| {name} | {format_pm(r.auc_mean, r.auc_std)} | {format_pm(r.acc_mean, r.acc_std)} | "
            f"{format_pm(r.f1_mean, r.f1_std)} | {format_pm(r.precision_mean, r.precision_std)} | "
            f"{format_pm(r.recall_mean, r.recall_std)} |"
        )

    lines.extend(
        [
            "",
            "## 8-Class Emotion",
            "",
            "| Modell | AUC Mean±Std | Accuracy Mean±Std | F1 Mean±Std | Precision Mean±Std | Recall Mean±Std |",
            "|---|---|---|---|---|---|",
        ]
    )
    for name, r in multi_results.items():
        lines.append(
            f"| {name} | {format_pm(r.auc_mean, r.auc_std)} | {format_pm(r.acc_mean, r.acc_std)} | "
            f"{format_pm(r.f1_mean, r.f1_std)} | {format_pm(r.precision_mean, r.precision_std)} | "
            f"{format_pm(r.recall_mean, r.recall_std)} |"
        )

    lines.extend(
        [
            "",
            "## Vergleich Baseline",
            "",
            f"- Random Baseline: 0.50",
            f"- Regelbasiert (aktueller Stand): 0.74",
            f"- Trainiertes bestes Modell ({best_name}): {best_auc:.4f} ± {best_std:.4f}",
            "- Literatur (eGeMAPS auf RAVDESS): ~0.80-0.85",
            "",
            "## Top 15 Feature Importance (bestes Modell)",
            "",
            "| Feature | Importance |",
            "|---|---:|",
        ]
    )
    for row in top_features:
        lines.append(f"| {row['feature']} | {row['importance']:.6f} |")

    lines.extend(["", "## Confusion Matrix (Binary, aggregiert OOF)", "", "| true \\ pred | low | high |", "|---|---:|---:|"])
    lines.append(f"| low | {conf[0][0]} | {conf[0][1]} |")
    lines.append(f"| high | {conf[1][0]} | {conf[1][1]} |")
    return "\n".join(lines) + "\n"


def build_html(
    binary_results: dict[str, ModelResult],
    top_features: list[dict[str, float]],
    conf: list[list[int]],
    roc_data: dict[str, dict[str, list[float]]],
) -> str:
    model_names = list(binary_results.keys())
    auc_vals = [binary_results[m].auc_mean for m in model_names]
    f_names = [r["feature"] for r in top_features][::-1]
    f_vals = [r["importance"] for r in top_features][::-1]

    return f"""<!doctype html>
<html lang="de">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Classifier Benchmark</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ margin:0; background: linear-gradient(180deg,#f9fafb 0%,#f3f6fb 100%); color:#1f2937; font-family:'DM Sans',sans-serif; }}
    .wrap {{ max-width:1260px; margin:40px auto; padding:0 24px 40px; }}
    .card {{ background:#fff; border:1px solid #e6eaf2; border-radius:18px; box-shadow:0 12px 40px rgba(15,23,42,.08); padding:28px 30px; }}
    h1 {{ margin:0 0 8px; font-family:'Source Serif 4',serif; font-size:38px; font-weight:600; }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
    .plot {{ height:380px; border:1px solid #edf0f5; border-radius:10px; }}
    @media (max-width:1024px) {{ .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Paper-Grade Emotion Classifier Benchmark</h1>
      <div class="grid">
        <div id="auc" class="plot"></div>
        <div id="cm" class="plot"></div>
        <div id="imp" class="plot"></div>
        <div id="roc" class="plot"></div>
      </div>
    </div>
  </div>
  <script>
    Plotly.newPlot('auc', [{{
      x: {json.dumps(model_names)},
      y: {json.dumps(auc_vals)},
      type: 'bar',
      marker: {{color:'#2563eb'}}
    }}], {{
      title:'Binary Arousal AUC Vergleich',
      yaxis:{{title:'AUC', range:[0.4,1.0]}}
    }}, {{displaylogo:false, responsive:true}});

    Plotly.newPlot('cm', [{{
      z: {json.dumps(conf)},
      x: ['low','high'],
      y: ['low','high'],
      type: 'heatmap',
      colorscale: 'Blues',
      text: {json.dumps(conf)},
      texttemplate: '%{{text}}'
    }}], {{
      title:'Confusion Matrix (OOF)',
    }}, {{displaylogo:false, responsive:true}});

    Plotly.newPlot('imp', [{{
      x: {json.dumps(f_vals)},
      y: {json.dumps(f_names)},
      type: 'bar',
      orientation: 'h',
      marker: {{color:'#e11d48'}}
    }}], {{
      title:'Top 15 Feature Importance',
    }}, {{displaylogo:false, responsive:true}});

    const rocData = {json.dumps(roc_data)};
    const traces = [];
    for (const [name, vals] of Object.entries(rocData)) {{
      traces.push({{
        x: vals.fpr,
        y: vals.tpr,
        type:'scatter',
        mode:'lines',
        name
      }});
    }}
    traces.push({{
      x:[0,1], y:[0,1], type:'scatter', mode:'lines', name:'Chance', line:{{dash:'dash', color:'#9ca3af'}}
    }});
    Plotly.newPlot('roc', traces, {{
      title:'ROC Curves (Binary Arousal)',
      xaxis:{{title:'FPR', range:[0,1]}},
      yaxis:{{title:'TPR', range:[0,1]}},
    }}, {{displaylogo:false, responsive:true}});
  </script>
</body>
</html>
"""


def train_best_binary_model(
    X: np.ndarray,
    y: np.ndarray,
    best_name: str,
    feature_names: list[str],
    best_metrics: ModelResult,
) -> dict[str, Any]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = model_factory(best_name, n_classes=2)
    model.fit(Xs, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    meta = {
        "model_type": best_name,
        "n_features": len(feature_names),
        "feature_set": "eGeMAPSv02",
        "task": "binary_arousal",
        "auc_mean": round(best_metrics.auc_mean, 4),
        "auc_std": round(best_metrics.auc_std, 4),
        "accuracy_mean": round(best_metrics.acc_mean, 4),
        "trained_on": "RAVDESS",
        "n_samples": int(X.shape[0]),
        "cv_folds": 5,
        "date": str(date.today()),
        "feature_names": feature_names,
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def run() -> None:
    X, emotions, feature_names, filenames = load_dataset()
    print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features.")
    if X.shape[1] != 88:
        print(f"[WARN] Expected 88 eGeMAPS features, got {X.shape[1]}.")

    y_binary = np.asarray([1 if e in HIGH_AROUSAL else 0 for e in emotions], dtype=np.int64)

    labels8 = sorted(set(emotions))
    le8 = LabelEncoder()
    y_8 = le8.fit_transform(np.asarray(emotions))

    binary_results = evaluate_models(X, y_binary, multiclass=False)
    multi_results = evaluate_models(X, y_8, multiclass=True)

    best_name = max(binary_results, key=lambda k: binary_results[k].auc_mean)
    best_metrics = binary_results[best_name]

    scaler = StandardScaler()
    model_for_imp = model_factory(best_name, n_classes=2)
    model_for_imp.fit(scaler.fit_transform(X), y_binary)
    top_features = top_feature_importance(model_for_imp, feature_names, top_k=15)

    conf = confusion_matrix(best_metrics.oof_true, best_metrics.oof_pred, labels=[0, 1]).tolist()

    roc_data: dict[str, dict[str, list[float]]] = {}
    for name, result in binary_results.items():
        fpr, tpr, _ = roc_curve(result.oof_true, result.oof_proba)
        roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    meta = train_best_binary_model(X, y_binary, best_name, feature_names, best_metrics)

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(
        build_markdown(
            binary_results=binary_results,
            multi_results=multi_results,
            best_name=best_name,
            best_auc=best_metrics.auc_mean,
            best_std=best_metrics.auc_std,
            top_features=top_features,
            conf=conf,
            labels=["low", "high"],
        ),
        encoding="utf-8",
    )
    REPORT_HTML.write_text(
        build_html(binary_results=binary_results, top_features=top_features, conf=conf, roc_data=roc_data),
        encoding="utf-8",
    )

    summary = {
        "samples": int(X.shape[0]),
        "features": int(X.shape[1]),
        "best_model": best_name,
        "best_auc_mean": best_metrics.auc_mean,
        "best_auc_std": best_metrics.auc_std,
        "best_acc_mean": best_metrics.acc_mean,
        "meta": meta,
        "binary_results": {
            n: {
                "auc_mean": r.auc_mean,
                "auc_std": r.auc_std,
                "accuracy_mean": r.acc_mean,
                "f1_mean": r.f1_mean,
                "precision_mean": r.precision_mean,
                "recall_mean": r.recall_mean,
            }
            for n, r in binary_results.items()
        },
    }
    out_json = ROOT / "data" / "benchmarks" / "classifier_results.json"
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Emotion Classifier Training Complete ===")
    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    for n, r in binary_results.items():
        print(f"{n}: AUC={r.auc_mean:.4f}±{r.auc_std:.4f}, ACC={r.acc_mean:.4f}, F1={r.f1_mean:.4f}")
    print(f"Best: {best_name} (AUC {best_metrics.auc_mean:.4f} ± {best_metrics.auc_std:.4f})")
    print(f"Model: {MODEL_PATH}")
    print(f"Scaler: {SCALER_PATH}")
    print(f"Meta: {META_PATH}")
    print(f"Report MD: {REPORT_MD}")
    print(f"Report HTML: {REPORT_HTML}")


if __name__ == "__main__":
    run()
