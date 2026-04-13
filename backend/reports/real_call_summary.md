# Orpheus Real-World Validation: Telephone Audio

## Dataset
- 15 successful analyses out of 15 collected calls
- Sources: sales calls, cold calls, customer service (YouTube)
- Audio quality: telephone compression, noise, spontaneous speech

## Key Question
Do the Orpheus biomarkers work on real telephone audio?

## Results

### Biomarker Validity
- 15 of 15 calls reached >=80% plausible biomarker checks
- Check ranges:
  - f0_mean: 50-400 Hz
  - jitter: 0.001-0.1
  - shimmer: 0.01-0.3
  - hnr: 0-30
  - speech_rate: 1-10
  - pause_rate: 0-80

### Emotional Variation
- Average stress range across segments: 0.548
- Average engagement range across segments: 0.541
- Calls with detectable emotional dynamics (>0.1 stress or engagement range): 15 of 15

### Alert Activity
- Total alerts fired: 427
- Alert plausibility requires manual listening around alert timestamps

### Humanness Scores
- Average: 50.49
- Range: 15.00 - 94.70
- Classified as human: 7 of 15

## Conclusion
PARTIALLY on telephone audio.

## Best Demo Calls
1. LIVE CALL: A Masterclass in Real-Time Objection Handling: Stress range 0.72, engagement range 0.73, alerts 48
2. 'OUR PRICES HAVE NEVER BEEN LOWER!' - The Office US: Stress range 0.86, engagement range 0.59, alerts 13
3. Sales Agent Gets Fired On The Spot - Sales Call Gone Wrong!: Stress range 0.8, engagement range 0.61, alerts 37
