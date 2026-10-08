from streamlit.testing.v1 import AppTest

from musicgen.paths import ROOT


def test_app_loads_trained_model_and_generates_downloadable_music(tmp_path, monkeypatch):
    monkeypatch.setattr("musicgen.paths.GENERATED", tmp_path)
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Make room for a new melody."
    assert app.selectbox[0].value == "Standard · 128 notes"
    app.selectbox[0].select("Sketch · 64 notes")
    app.number_input[0].set_value(71)
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert app.success[0].value == "Your composition is ready. Press play to listen."
    result = app.session_state["composition"]
    assert result["metadata"]["note_count"] == 64
    assert result["metadata"]["settings"]["seed"] == 71
    assert result["midi"].exists()
    assert result["wav"].read_bytes()[:4] == b"RIFF"


def test_app_missing_model_shows_training_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr("musicgen.paths.MODEL_DIR", tmp_path)
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert not app.exception
    assert app.info[0].value == "Train your piano model to begin composing."
    assert "musicgen.train" in app.code[0].value
