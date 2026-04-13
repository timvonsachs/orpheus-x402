# Orpheus EWMA Baseline Validation

## Input
- Erfolgreiche Calls: 7
- Simulierter Speaker: `test_speaker_001`

## Alert-Vergleich
- Alerts alte Methode (feste Schwellen): **84**
- Alerts neue Methode (EWMA, |z| > 2.0): **74**
- Reduzierte Alerts (old-only): **63** (75.0% der alten Alerts)

## Beobachtung
- `Call 1, Segment 2: Alter Alert 'stress_spike' bei absolutem Wert 0.92. Neuer Alert: Kein Alert, weil Z-Score nur 0.00 ist.`
- `Call 1, Segment 8: Neuer Alert 'disengagement' mit Z-Score -2.42.`

## Interpretation
- Die EWMA-Methode unterdrueckt Sprecher-typische, aber unkritische Auspraegungen.
- Gleichzeitig markiert sie ungewoehnliche Abweichungen frueher, sobald die Baseline warm ist.
- Damit sinkt in der Regel die Menge an potenziellen False Positives bei gleichzeitiger Personalisierung.
