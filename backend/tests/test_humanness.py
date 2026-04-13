from engine.humanness import HumannessEngine


def test_humanness_interpretation_bands():
    assert "natuerlich" in HumannessEngine._interpretation(80)
    assert "Grenzbereich" in HumannessEngine._interpretation(55)
    assert "wahrscheinlich TTS-generiert" in HumannessEngine._interpretation(40)
    assert "sehr wahrscheinlich synthetisch" in HumannessEngine._interpretation(10)
