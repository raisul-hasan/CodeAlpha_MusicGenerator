from pathlib import Path

import numpy as np
import pytest
import torch

from musicgen.generate import Generator, export
from musicgen.midi import CARDINALITIES, read_midi
from musicgen.model import ModelConfig, MusicLSTM
from musicgen.paths import MODEL_DIR
from musicgen.train import PieceWindows


def test_windows_never_cross_piece_boundaries(tmp_path):
    data = np.zeros((14, 4), dtype=np.int64)
    data[:7, 0] = 1
    data[7:, 0] = 2
    path = tmp_path / "notes.npz"
    np.savez(path, notes=data, offsets=np.array([0, 7, 14]))
    windows = PieceWindows(path, sequence_length=4, stride=1)
    assert len(windows) == 6
    for inputs, targets in windows:
        assert torch.unique(torch.cat([inputs[:, 0], targets[:, 0]])).numel() == 1


@pytest.fixture(scope="module")
def trained_generator():
    assert (MODEL_DIR / "piano_lstm.pt").exists(), "Train the real model before running acceptance tests."
    return Generator()


def test_lstm_backpropagates_into_all_four_prediction_heads():
    torch.set_num_threads(2)
    model = MusicLSTM(ModelConfig(hidden_size=16))
    inputs = torch.stack([torch.randint(0, size, (2, 5)) for size in CARDINALITIES], dim=-1)
    predictions, _ = model(inputs)
    loss = sum(torch.nn.functional.cross_entropy(head.reshape(-1, size), inputs[..., index].reshape(-1))
               for index, (head, size) in enumerate(zip(predictions, CARDINALITIES)))
    loss.backward()
    assert all(torch.isfinite(head.weight.grad).all() and head.weight.grad.abs().sum() > 0
               for head in model.heads)
    assert model.lstm.weight_ih_l0.grad.abs().sum() > 0


def test_saved_model_reproducibility_variation_and_polyphony_constraints(trained_generator):
    first = trained_generator.generate(note_count=64, seed=7)
    assert first == trained_generator.generate(note_count=64, seed=7)
    assert first != trained_generator.generate(note_count=64, seed=8)
    assert len(first) == 64
    assert first[0].start == 0
    assert first[-1].start > first[0].start
    for note in first:
        assert 21 <= note.pitch <= 108 and note.duration > 0
        assert 1 <= note.velocity <= 127
        assert sum(other.start == note.start for other in first) <= 3
    for index, note in enumerate(first):
        assert all(other.pitch != note.pitch or other.start + other.duration <= note.start + 1e-8
                   for other in first[:index])
        if index:
            spacing = note.start - first[index - 1].start
            assert spacing == 0 or spacing >= .25 - 1e-8
            if spacing == 0:
                assert note.pitch > first[index - 1].pitch


def test_generation_export_roundtrip(trained_generator, tmp_path):
    notes = trained_generator.generate(note_count=64, seed=2026, bpm=110)
    result = export(notes, tmp_path, "test_music", 110,
                    {"seed": 2026, "temperature": .9, "notes": 64, "top_k": 12})
    recovered = read_midi(result["midi"])
    assert len(recovered) == len(notes)
    assert result["wav"].read_bytes()[:4] == b"RIFF"
    assert sorted(n.pitch for n in recovered) == sorted(n.pitch for n in notes)
    for original, actual in zip(sorted(notes, key=lambda n: (n.start, n.pitch)), recovered):
        assert actual.start == pytest.approx(original.start, abs=.001)
        assert actual.duration == pytest.approx(original.duration, abs=.001)


@pytest.mark.parametrize("settings", [{"note_count": 0}, {"temperature": float("nan")},
                                     {"temperature": 0}, {"top_k": 0}, {"bpm": 0},
                                     {"seed": -1}, {"max_seconds": 999}])
def test_invalid_generation_controls_rejected(trained_generator, settings):
    with pytest.raises(ValueError):
        trained_generator.generate(**settings)


def test_missing_model_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="musicgen.train"):
        Generator(tmp_path / "missing.pt")
