# Orpheus vs. Hume TADA: Kann Ohr 2 State-of-the-Art TTS erkennen?

- TADA Durchschnitt: **3.72/100**
- Human Durchschnitt: **51.99/100**
- AUC: **1.0000**
- Accuracy @50: **0.8500**
- Korrekt erkannt: **10/10 TADA als synthetic**, **7/10 Human als human**
- Human-Quelle: `preexisting`
- TADA-Erzeugung: `blog_demo_fallback`
- Engine-Modus: `melcnn`

## Samples

| Sample | Label | Humanness Score | Prediction |
|---|---|---:|---|
| tada_01.wav | tada | 3.00 | synthetic |
| tada_06.wav | tada | 0.50 | synthetic |
| tada_07.wav | tada | 2.30 | synthetic |
| tada_08.wav | tada | 3.00 | synthetic |
| tada_09.wav | tada | 1.30 | synthetic |
| tada_10.wav | tada | 0.20 | synthetic |
| tada_blog_01.wav | tada | 11.60 | synthetic |
| tada_blog_02.wav | tada | 2.60 | synthetic |
| tada_blog_03.wav | tada | 11.20 | synthetic |
| tada_blog_04.wav | tada | 1.50 | synthetic |
| human_01.wav | human | 51.50 | human |
| human_02.wav | human | 70.10 | human |
| human_03.wav | human | 50.90 | human |
| human_04.wav | human | 51.30 | human |
| human_05.wav | human | 38.10 | synthetic |
| human_06.wav | human | 41.20 | synthetic |
| human_07.wav | human | 56.10 | human |
| human_08.wav | human | 20.90 | synthetic |
| human_09.wav | human | 80.20 | human |
| human_10.wav | human | 59.60 | human |

## Confusion Matrix

| true \\ pred | synthetic | human |
|---|---:|---:|
| synthetic | 10 | 0 |
| human | 3 | 7 |

## Fazit

Orpheus erkennt TADA zuverlaessig. Ohr 2 funktioniert gegen State-of-the-Art.

## Hinweise/Fehler bei TADA-Erzeugung

- local_generation_failed: Could not load model HumeAI/tada-1b with any of the following classes: (<class 'transformers.models.auto.modeling_auto.AutoModelForTextToWaveform'>, <class 'transformers.models.auto.modeling_auto.AutoModelForTextToSpectrogram'>). See the original errors:

while loading with AutoModelForTextToWaveform, an error is thrown:
Traceback (most recent call last):
  File "/Users/elpatron/orpheus-acoustic-sense/.venv/lib/python3.9/site-packages/transformers/pipelines/base.py", line 293, in infer_framework_load_model
    model = model_class.from_pretrained(model, **kwargs)
  File "/Users/elpatron/orpheus-acoustic-sense/.venv/lib/python3.9/site-packages/transformers/models/auto/auto_factory.py", line 607, in from_pretrained
    raise ValueError(
ValueError: Unrecognized configuration class <class 'transformers.models.llama.configuration_llama.LlamaConfig'> for this kind of AutoModel: AutoModelForTextToWaveform.
Model type should be one of BarkConfig, CsmConfig, FastSpeech2ConformerConfig, FastSpeech2ConformerWithHifiGanConfig, MusicgenConfig, MusicgenMelodyConfig, Qwen2_5OmniConfig, Qwen3OmniMoeConfig, SeamlessM4TConfig, SeamlessM4Tv2Config, VitsConfig.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/elpatron/orpheus-acoustic-sense/.venv/lib/python3.9/site-packages/transformers/pipelines/base.py", line 311, in infer_framework_load_model
    model = model_class.from_pretrained(model, **fp32_kwargs)
  File "/Users/elpatron/orpheus-acoustic-sense/.venv/lib/python3.9/site-packages/transformers/models/auto/auto_factory.py", line 607, in from_pretrained
    raise ValueError(
ValueError: Unrecognized configuration class <class 'transformers.models.llama.configuration_llama.LlamaConfig'> for this kind of AutoModel: AutoModelForTextToWaveform.
Model type should be one of BarkConfig, CsmConfig, FastSpeech2ConformerConfig, FastSpeech2ConformerWithHifiGanConfig, MusicgenConfig, MusicgenMelodyConfig, Qwen2_5OmniConfig, Qwen3OmniMoeConfig, SeamlessM4TConfig, SeamlessM4Tv2Config, VitsConfig.

while loading with AutoModelForTextToSpectrogram, an error is thrown:
Traceback (most recent call last):
  File "/Users/elpatron/orpheus-acoustic-sense/.venv/lib/python3.9/site-packages/transformers/pipelines/base.py", line 293, in infer_framework_load_model
    model = model_class.from_pretrained(model, **kwargs)
  File "/Users/elpatron/orpheus-acoustic-sense/.venv/lib/python3.9/site-packages/transformers/models/auto/auto_factory.py", line 607, in from_pretrained
    raise ValueError(
ValueError: Unrecognized configuration class <class 'transformers.models.llama.configuration_llama.LlamaConfig'> for this kind of AutoModel: AutoModelForTextToSpectrogram.
Model type should be one of FastSpeech2ConformerConfig, SpeechT5Config.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/elpatron/orpheus-acoustic-sense/.venv/lib/python3.9/site-packages/transformers/pipelines/base.py", line 311, in infer_framework_load_model
    model = model_class.from_pretrained(model, **fp32_kwargs)
  File "/Users/elpatron/orpheus-acoustic-sense/.venv/lib/python3.9/site-packages/transformers/models/auto/auto_factory.py", line 607, in from_pretrained
    raise ValueError(
ValueError: Unrecognized configuration class <class 'transformers.models.llama.configuration_llama.LlamaConfig'> for this kind of AutoModel: AutoModelForTextToSpectrogram.
Model type should be one of FastSpeech2ConformerConfig, SpeechT5Config.



- inference_generation_failed: 
