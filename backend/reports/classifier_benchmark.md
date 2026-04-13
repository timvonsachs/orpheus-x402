# Paper-Grade Emotion Classifier Benchmark (RAVDESS, eGeMAPS)

## Binary Arousal (High vs Low)

| Modell | AUC Mean±Std | Accuracy Mean±Std | F1 Mean±Std | Precision Mean±Std | Recall Mean±Std |
|---|---|---|---|---|---|
| RandomForest | 0.9177 ± 0.0151 | 0.8278 ± 0.0181 | 0.8388 ± 0.0158 | 0.8386 ± 0.0282 | 0.8398 ± 0.0237 |
| XGBoost | 0.9371 ± 0.0129 | 0.8583 ± 0.0108 | 0.8675 ± 0.0108 | 0.8657 ± 0.0151 | 0.8698 ± 0.0236 |
| SVM | 0.9117 ± 0.0139 | 0.8215 ± 0.0140 | 0.8340 ± 0.0114 | 0.8284 ± 0.0203 | 0.8399 ± 0.0132 |
| LogisticRegression | 0.9027 ± 0.0100 | 0.8181 ± 0.0119 | 0.8291 ± 0.0125 | 0.8304 ± 0.0129 | 0.8281 ± 0.0205 |

## 8-Class Emotion

| Modell | AUC Mean±Std | Accuracy Mean±Std | F1 Mean±Std | Precision Mean±Std | Recall Mean±Std |
|---|---|---|---|---|---|
| RandomForest | 0.9244 ± 0.0105 | 0.6451 ± 0.0351 | 0.6319 ± 0.0387 | 0.6644 ± 0.0281 | 0.6310 ± 0.0378 |
| XGBoost | 0.9314 ± 0.0131 | 0.6556 ± 0.0313 | 0.6428 ± 0.0373 | 0.6492 ± 0.0374 | 0.6439 ± 0.0368 |
| SVM | 0.9093 ± 0.0088 | 0.5965 ± 0.0259 | 0.5799 ± 0.0271 | 0.5912 ± 0.0305 | 0.5808 ± 0.0260 |
| LogisticRegression | 0.9064 ± 0.0137 | 0.6097 ± 0.0202 | 0.6013 ± 0.0217 | 0.6063 ± 0.0267 | 0.6029 ± 0.0207 |

## Vergleich Baseline

- Random Baseline: 0.50
- Regelbasiert (aktueller Stand): 0.74
- Trainiertes bestes Modell (XGBoost): 0.9371 ± 0.0129
- Literatur (eGeMAPS auf RAVDESS): ~0.80-0.85

## Top 15 Feature Importance (bestes Modell)

| Feature | Importance |
|---|---:|
| spectralFluxV_sma3nz_amean | 0.110181 |
| loudness_sma3_meanFallingSlope | 0.091118 |
| spectralFlux_sma3_amean | 0.023476 |
| loudness_sma3_percentile50.0 | 0.018437 |
| spectralFluxV_sma3nz_stddevNorm | 0.018099 |
| mfcc1V_sma3nz_amean | 0.017584 |
| jitterLocal_sma3nz_amean | 0.015799 |
| StddevUnvoicedSegmentLength | 0.014852 |
| mfcc3_sma3_stddevNorm | 0.014222 |
| slopeV0-500_sma3nz_stddevNorm | 0.013927 |
| loudness_sma3_pctlrange0-2 | 0.013554 |
| F1bandwidth_sma3nz_amean | 0.012660 |
| F0semitoneFrom27.5Hz_sma3nz_amean | 0.012237 |
| mfcc4V_sma3nz_amean | 0.012154 |
| spectralFluxUV_sma3nz_amean | 0.012152 |

## Confusion Matrix (Binary, aggregiert OOF)

| true \ pred | low | high |
|---|---:|---:|
| low | 568 | 104 |
| high | 100 | 668 |
