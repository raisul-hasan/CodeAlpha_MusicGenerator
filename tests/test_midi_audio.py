import io
import wave

import mido
import numpy as np
import pytest

from musicgen.audio import render_audio
from musicgen.midi import Note, decode_notes, encode_notes, read_midi, write_midi


def test_midi_export_preserves_polyphony_timing_velocity_and_tempo(tmp_path):
    notes = [Note(60, 0, 1, 64), Note(64, 0, .5, 96), Note(67, .5, 1, 80)]
    path = tmp_path / "chord.mid"
    write_midi(notes, path, 120)
    recovered = read_midi(path)
    assert [(n.pitch, n.start, n.duration, n.velocity) for n in recovered] == [
        (n.pitch, n.start, n.duration, n.velocity) for n in notes]
    tempos = [message.tempo for message in mido.merge_tracks(mido.MidiFile(path).tracks)
              if message.type == "set_tempo"]
    assert tempos == [500000]


def test_encoding_preserves_simultaneous_notes_and_musical_intervals():
    notes = [Note(60, 2, .25, 64), Note(64, 2, .5, 80), Note(67, 2.5, 1, 96)]
    recovered = decode_notes(encode_notes(notes))
    assert [n.pitch for n in recovered] == [60, 64, 67]
    assert [n.start for n in recovered] == [0, 0, .5]
    assert [n.duration for n in recovered] == [.25, .5, 1]


def test_audio_has_correct_tempo_duration_and_non_silent_finite_pcm():
    notes = [Note(60, 0, 1), Note(67, 1, 1)]
    durations = []
    for bpm in [60, 120]:
        with wave.open(io.BytesIO(render_audio(notes, bpm)), "rb") as wav:
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
            assert np.max(np.abs(samples.astype(np.int32))) > 1000
            assert np.isfinite(samples).all()
            assert np.max(np.abs(samples.astype(np.int32))) < 32767
            durations.append(wav.getnframes() / wav.getframerate())
    assert durations[0] == pytest.approx(2.45, abs=.001)
    assert durations[1] == pytest.approx(1.45, abs=.001)


def test_read_handles_zero_velocity_off_and_excludes_drums(tmp_path):
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.extend([mido.Message("note_on", note=60, velocity=80, time=0),
                  mido.Message("note_on", note=36, channel=9, velocity=100, time=0),
                  mido.Message("note_on", note=60, velocity=0, time=480),
                  mido.Message("note_off", note=36, channel=9, time=0)])
    path = tmp_path / "mixed.mid"
    midi.save(path)
    assert read_midi(path) == [Note(60, 0, 1, 80)]


def test_read_closes_notes_left_on_at_end_of_file(tmp_path):
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack([mido.Message("note_on", note=60, velocity=70),
                           mido.MetaMessage("end_of_track", time=240)])
    midi.tracks.append(track)
    path = tmp_path / "unclosed.mid"
    midi.save(path)
    assert read_midi(path) == [Note(60, 0, .5, 70)]


@pytest.mark.parametrize("operation", [lambda: encode_notes([]), lambda: render_audio([])])
def test_empty_music_rejected(operation):
    with pytest.raises(ValueError):
        operation()
