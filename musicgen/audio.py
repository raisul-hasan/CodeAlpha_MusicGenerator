"""Portable piano-like additive synthesis, without an external soundfont."""
import io
import wave
from pathlib import Path

import numpy as np

from .midi import Note

SAMPLE_RATE = 22050


def render_audio(notes: list[Note], bpm: int = 100, sample_rate: int = SAMPLE_RATE) -> bytes:
    if not notes or not 30 <= bpm <= 240:
        raise ValueError("Provide notes and a tempo between 30 and 240 BPM.")
    seconds_per_beat = 60 / bpm
    total = max(note.start + note.duration for note in notes) * seconds_per_beat + 0.45
    if total > 600:
        raise ValueError("Audio rendering is limited to ten minutes.")
    audio = np.zeros(int(np.ceil(total * sample_rate)), dtype=np.float64)
    for note in notes:
        held = note.duration * seconds_per_beat
        length = int((held + 0.35) * sample_rate)
        t = np.arange(length, dtype=np.float64) / sample_rate
        fundamental = 440 * 2 ** ((note.pitch - 69) / 12)
        tone = np.zeros(length)
        # Harmonics decay separately, with a gentle release after key-off.
        for harmonic, weight in [(1, 1), (2, .38), (3, .18), (4, .08), (6, .025)]:
            if fundamental * harmonic >= sample_rate / 2:
                continue
            tone += weight * np.sin(2 * np.pi * fundamental * harmonic * t) * np.exp(-t * (.8 + harmonic * .32))
        attack = np.minimum(1, t / .008)
        release = np.exp(-np.maximum(0, t - held) * 18)
        tone *= attack * release * (note.velocity / 127) ** 1.3
        start = int(round(note.start * seconds_per_beat * sample_rate))
        end = min(start + length, len(audio))
        audio[start:end] += tone[:end - start]
    peak = np.max(np.abs(audio))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("The generated audio is silent or invalid.")
    pcm = (audio / peak * .88 * 32767).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return output.getvalue()


def write_audio(notes: list[Note], destination: Path, bpm: int = 100) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(render_audio(notes, bpm))
