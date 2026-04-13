from pathlib import Path

from benchmarks.ravdess_benchmark import map_arousal_class, parse_ravdess_emotion


def test_filename_parser_recognizes_all_emotions():
    examples = {
        "03-01-01-01-01-01-01.wav": "neutral",
        "03-01-02-01-01-01-02.wav": "calm",
        "03-01-03-01-01-01-03.wav": "happy",
        "03-01-04-01-01-01-04.wav": "sad",
        "03-01-05-01-01-01-05.wav": "angry",
        "03-01-06-01-01-01-06.wav": "fearful",
        "03-01-07-01-01-01-07.wav": "disgust",
        "03-01-08-01-01-01-08.wav": "surprised",
    }
    for filename, expected in examples.items():
        assert parse_ravdess_emotion(Path(filename)) == expected


def test_arousal_mapping_consistency():
    assert map_arousal_class("angry") == "high_arousal"
    assert map_arousal_class("fearful") == "high_arousal"
    assert map_arousal_class("disgust") == "high_arousal"

    assert map_arousal_class("neutral") == "low_arousal"
    assert map_arousal_class("calm") == "low_arousal"
    assert map_arousal_class("sad") == "low_arousal"

    assert map_arousal_class("happy") == "medium_arousal"
    assert map_arousal_class("surprised") == "medium_arousal"
