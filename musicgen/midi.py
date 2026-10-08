"""MIDI note extraction, categorical musical timing, and music21 export."""
from dataclasses import asdict, dataclass
from pathlib import Path

import mido
import numpy as np

MIN_PITCH = 21
MAX_PITCH = 108
STEPS = np.array([0, 1/24, 1/12, 1/8, 1/6, 1/4, 1/3, 1/2,
                  2/3, 3/4, 1, 1.5, 2, 3, 4, 6, 8], dtype=np.float64)
DURATIONS = STEPS[1:].copy()
CARDINALITIES = (88, len(STEPS), len(DURATIONS), 16)
FEATURES = ("pitch", "step", "duration", "velocity")


@dataclass(frozen=True)
class Note:
    pitch: int
    start: float  # quarter-note beats
    duration: float
    velocity: int = 80

    def to_dict(self):
        return asdict(self)


def read_midi(path: Path) -> list[Note]:
    midi = mido.MidiFile(path)
    if midi.type == 2:
        raise ValueError("Asynchronous type-2 MIDI files are unsupported.")
    if midi.ticks_per_beat <= 0:
        raise ValueError("SMPTE time division is unsupported.")
    active, notes, ticks = {}, [], 0
    for event in mido.merge_tracks(midi.tracks):
        ticks += event.time
        if event.type not in ("note_on", "note_off") or event.channel == 9:
            continue
        key = (event.channel, event.note)
        is_on = event.type == "note_on" and event.velocity > 0
        if key in active:
            start, velocity = active.pop(key)
            if MIN_PITCH <= event.note <= MAX_PITCH and ticks > start:
                notes.append(Note(event.note, start / midi.ticks_per_beat,
                                  (ticks - start) / midi.ticks_per_beat, velocity))
        if is_on:
            active[key] = (ticks, event.velocity)
    for (_, pitch), (start, velocity) in active.items():
        if MIN_PITCH <= pitch <= MAX_PITCH:
            notes.append(Note(pitch, start / midi.ticks_per_beat,
                              max(1/24, (ticks - start) / midi.ticks_per_beat), velocity))
    return sorted(notes, key=lambda note: (note.start, note.pitch))


def encode_notes(notes: list[Note]) -> np.ndarray:
    if not notes:
        raise ValueError("The MIDI file contains no usable piano notes.")
    encoded, previous = [], notes[0].start
    for note in notes:
        if not MIN_PITCH <= note.pitch <= MAX_PITCH or note.duration <= 0:
            raise ValueError("Notes must have a piano pitch and positive duration.")
        delta = max(0, note.start - previous)
        encoded.append((note.pitch - MIN_PITCH, int(np.argmin(abs(STEPS - delta))),
                        int(np.argmin(abs(DURATIONS - note.duration))),
                        min(15, max(0, (note.velocity - 1) // 8))))
        previous = note.start
    return np.asarray(encoded, dtype=np.int64)


def decode_notes(events: np.ndarray) -> list[Note]:
    notes, onset = [], 0.0
    for pitch, step, duration, velocity in events:
        onset += float(STEPS[step])
        notes.append(Note(int(pitch) + MIN_PITCH, onset, float(DURATIONS[duration]),
                          min(127, int(velocity) * 8 + 4)))
    return notes


def write_midi(notes: list[Note], destination: Path, bpm: int = 100) -> None:
    from music21 import instrument, note as musical_note, stream, tempo

    if not notes or not 30 <= bpm <= 240:
        raise ValueError("Provide nonempty notes and a tempo between 30 and 240 BPM.")
    score = stream.Stream()
    score.insert(0, instrument.Piano())
    score.insert(0, tempo.MetronomeMark(number=bpm))
    for event in notes:
        item = musical_note.Note(event.pitch, quarterLength=event.duration)
        item.volume.velocity = event.velocity
        score.insert(event.start, item)
    destination.parent.mkdir(parents=True, exist_ok=True)
    score.write("midi", fp=str(destination))
