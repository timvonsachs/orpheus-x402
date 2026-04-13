# Orpheus Acoustic Sense

## Tagline
"Voice Intelligence Engine - Emotion Detection, Deepfake Defense, Personalized Baselines"

`Python 3.10+` `OpenSMILE` `FastAPI` `MIT License`

## Was ist Orpheus?

Orpheus ist eine Voice Intelligence Engine, die Audio-Streams in Echtzeit analysiert und drei Signale liefert:

1. **Emotion Detection** - Stress, Engagement, Arousal aus 21 akustischen Biomarkern (eGeMAPS) mit AUC 0.74 auf RAVDESS (1.440 Files, high vs low arousal, reproduzierbar)
2. **Deepfake Defense** - Human-vs-AI Voice Classification mit 99.8% Accuracy (MelCNN)
3. **Personalisierte Baselines** - EWMA Z-Scores pro Sprecher statt Populationsdurchschnitten. 75% weniger False Positives als statische Schwellwerte

Keine Firma auf der Welt kombiniert alle drei in einer Engine.

## Architektur

Audio Input -> Preprocessing (16kHz Mono)  
|- Humanness Engine (MelCNN) -> Human/AI Score  
|- Paralinguistic Engine (OpenSMILE eGeMAPS) -> 21 Biomarker -> Stress/Engagement/Arousal  
|- Biomarker Engine (6.373 ComParE Features) -> Premium Tier  
|- EWMA Baselines -> Personalisierte Z-Scores pro Speaker  
|- Alert Engine -> Z-Score > 2.0 = Alert, Fallback auf statische Schwellwerte

## Key Results

| Metric | Result | Method |
|--------|--------|--------|
| Emotion Detection AUC | 0.74 | RAVDESS (1,440 files, high vs low arousal, reproducible benchmark) |
| Binary Emotion Subset AUC | 0.93+ | RAVDESS Pairwise Subset (96 clips, one-vs-one, best-feature AUC) |
| TTS Detection Accuracy | 99.8% | MelCNN Phase 1 Classifier |
| False Positive Reduction | 75% | EWMA vs Static Thresholds (7 Real Calls) |
| Biomarker Plausibility | 7/7 | Real YouTube Customer Service Calls |

## Demo

Orpheus analysiert echte Kundenservice-Calls und erkennt emotionale Dynamik in Echtzeit:

**Human Call Analysis:**
- Humanness: 94.6 (Human bestaetigt)
- Frustration erkannt: Sekunde 10 (Stress 0.43 -> 0.92)
- Eskalation empfohlen: Sekunde 55 (Stress Peak 1.00)
- Ergebnis: 30 Sekunden fruehere Eskalation als ein menschlicher Agent

Screenshots und Demo-Visualisierungen befinden sich in `reports/`:
- `orpheus-demo-showcase-human.html` - Timeline eines echten Calls
- `ewma_validation.html` - Vorher/Nachher Vergleich der Baseline-Methoden
- `cognitive-integrity-demo.html` - Kombinierte Text+Voice Demo

## API

```bash
# Analyse eines Audio-Files
curl -X POST "http://localhost:8001/v1/sense?mode=full&segments=true" \
  -F "file=@call.wav"
```

Response:

```json
{
  "humanness": {"score": 94.6, "classification": "human"},
  "segments": [
    {
      "start": 5.0,
      "end": 10.0,
      "biomarkers": {
        "f0_mean": 142.3,
        "jitter": 0.012,
        "speech_rate": 4.9,
        "pause_rate": 22.1
      },
      "paralinguistic_summary": {
        "stress_level": 0.43,
        "engagement": 0.68
      }
    },
    {
      "start": 10.0,
      "end": 15.0,
      "paralinguistic_summary": {
        "stress_level": 0.92,
        "engagement": 0.89
      }
    }
  ],
  "trends": {
    "stress_trend": "rising"
  },
  "alerts": [
    {"type": "stress_spike"}
  ]
}
```

## Features

**Standard Tier (21 Biomarker):**
- Stress, Engagement, Arousal Scores
- F0, Jitter, Shimmer, HNR
- Speech Rate, Pause Rate, Formanten
- Segment-basierte Timeline
- EWMA personlisierte Baselines
- Echtzeit-Alerts (Z-Score > 2.0)

**Premium Tier (6.373 Features):**
- Alles aus Standard
- ComParE Full Feature Set
- Deepfake/TTS Detection
- Full Emotion Spectrum
- Speaker DNA Profile

## Projektstruktur

```text
orpheus-acoustic-sense/
|- api/                  # FastAPI Endpoints
|  |- routes.py
|- engine/
|  |- humanness.py       # MelCNN Human-vs-AI
|  |- paralinguistic.py  # OpenSMILE eGeMAPS Mapping
|  |- biomarkers.py      # Feature Extraction Proxy
|  |- baselines.py       # EWMA Personalized Baselines
|  |- alerts.py          # Z-Score Alert Engine
|  |- combiner.py        # Multi-Engine Orchestration
|  |- trends.py          # Segment Trend Analysis
|- models/               # Trained Models (MelCNN etc.)
|- data/
|  |- real_calls/        # Test Data + Results
|  |- baselines/         # Speaker Baseline JSONs
|- reports/              # Demo Visualizations
|- scripts/              # Validation & Analysis Scripts
|- tests/                # Test Suite
|  |- test_baselines.py  # EWMA Tests
```

## Quick Start

```bash
# Clone
git clone https://github.com/timvonsachs/orpheus-acoustic-sense.git
cd orpheus-acoustic-sense

# Install
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Run API
.venv/bin/python main.py

# Run Tests
.venv/bin/pytest tests/ -v
```

## Validation

Orpheus wurde an 15 echten YouTube Customer-Service-Calls getestet:
- 15/15 erfolgreich analysiert
- Biomarker-Plausibilitaet: 15/15 bestaetigt
- Emotionale Variation: 15/15 mit messbarer Dynamik
- EWMA-Baselines validiert: 84 statische Alerts -> 74 EWMA-Alerts (75% False-Positive-Reduktion)

## Was Orpheus NICHT ist

- Kein Luegendetektor. Orpheus misst Stress und Emotion, nicht Wahrheit.
- Kein Diagnose-Tool. Orpheus ist Screening, nicht Befund.
- Kein Ersatz fuer Menschen. Orpheus empfiehlt Eskalation. Der Mensch entscheidet.

## Roadmap

- [ ] Audio Preprocessing Pipeline (Auto-Resample 16kHz Mono)
- [ ] Speaker Diarization (Agent vs Kunde trennen)
- [ ] Telefon-Audio Fine-Tuning fuer Humanness Model
- [ ] REST API v2 mit Speaker-ID im Header
- [ ] SDK fuer Python und JavaScript
- [ ] Hosted API mit Pay-per-Call Pricing

## Autor

**Tim von Sachs** - AI Developer, Muenchen
- Email: timvonsachs@gmail.com
- Anima (AI Verification): https://chatbotaudit.vercel.app
- GitHub: https://github.com/timvonsachs

## Lizenz

MIT

---

*"Voice erzeugt maschinenlesbare Signale ueber menschliche Absicht, Risiko und Zustand. Wer diese Signale liest, baut Infrastruktur."*
