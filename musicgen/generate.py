"""Sample novel note events from the trained model and export MIDI/WAV."""
import argparse
import math
from pathlib import Path

import numpy as np
import torch

from .audio import write_audio
from .io import write_json
from .midi import CARDINALITIES, DURATIONS, MIN_PITCH, STEPS, Note, write_midi
from .model import ModelConfig, MusicLSTM
from .paths import GENERATED, MODEL_DIR


class Generator:
    def __init__(self, checkpoint_path: Path = MODEL_DIR / "piano_lstm.pt"):
        if not checkpoint_path.exists():
            raise FileNotFoundError("No trained model found. Run python -m musicgen.dataset and python -m musicgen.train.")
        torch.set_num_threads(2)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1 or checkpoint.get("cardinalities") != list(CARDINALITIES):
            raise ValueError("This checkpoint uses an incompatible music representation.")
        self.model = MusicLSTM(ModelConfig(**checkpoint["model_config"]))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.contexts = checkpoint["seed_contexts"]
        self.metadata = {key: value for key, value in checkpoint.items()
                         if key not in ("state_dict", "seed_contexts")}

    @staticmethod
    def sample(logits: torch.Tensor, temperature: float, top_k: int,
               rng: torch.Generator, blocked: list[int] | None = None) -> int:
        scores = logits.detach().clone() / temperature
        if blocked:
            scores[blocked] = -torch.inf
        k = min(top_k, int(torch.isfinite(scores).sum()))
        if k < 1:
            raise ValueError("No valid next note can be sampled.")
        values, indices = torch.topk(scores, k)
        selection = torch.multinomial(torch.softmax(values, dim=0), 1, generator=rng)
        return int(indices[selection].item())

    def generate(self, note_count: int = 128, temperature: float = .9, top_k: int = 12,
                 seed: int = 42, bpm: int = 100, max_seconds: float = 120) -> list[Note]:
        if not 32 <= note_count <= 512:
            raise ValueError("Length must be between 32 and 512 notes.")
        if not math.isfinite(temperature) or not .2 <= temperature <= 2:
            raise ValueError("Temperature must be between 0.2 and 2.0.")
        if not 1 <= top_k <= CARDINALITIES[0] or not 30 <= bpm <= 240:
            raise ValueError("Top-k must be 1-88 and tempo must be 30-240 BPM.")
        if not 0 <= seed <= 2**32 - 1 or not 5 <= max_seconds <= 120:
            raise ValueError("Seed or maximum duration is out of range.")
        rng = torch.Generator().manual_seed(seed)
        context = self.contexts[seed % len(self.contexts)].unsqueeze(0)
        notes, active, onset, simultaneous = [], {}, 0., 0
        max_beats = max_seconds * bpm / 60
        with torch.inference_mode():
            predictions, state = self.model(context)
            for _ in range(note_count):
                rhythm_temperature = max(.8, temperature)
                # Suppress human microtiming categories to avoid rushed bursts.
                blocked_steps = [index for index, value in enumerate(STEPS) if 0 < value < .25]
                if simultaneous >= 3 or (notes and notes[-1].pitch >= 104):
                    blocked_steps.append(0)
                step = self.sample(predictions[1][0, -1], rhythm_temperature, min(top_k, len(STEPS)),
                                   rng, blocked_steps)
                # Start the new composition at beat zero; the seed is only model context.
                if not notes:
                    step = 0
                proposed_onset = onset + float(STEPS[step])
                if proposed_onset >= max_beats - 1/24:
                    break
                onset = proposed_onset
                simultaneous = simultaneous + 1 if step == 0 else 1
                active = {pitch: end for pitch, end in active.items() if end > onset + 1e-8}
                blocked_pitches = set(active)
                if notes and step == 0:
                    # Follow the low-to-high chord ordering used in preprocessing.
                    blocked_pitches.update(range(notes[-1].pitch - MIN_PITCH + 1))
                if len(blocked_pitches) == CARDINALITIES[0]:
                    step = int(np.flatnonzero(STEPS == .25)[0])
                    onset += .25
                    if onset >= max_beats - 1/24:
                        break
                    simultaneous = 1
                    active = {pitch: end for pitch, end in active.items() if end > onset + 1e-8}
                    blocked_pitches = set(active)
                pitch = self.sample(predictions[0][0, -1], temperature, top_k, rng, sorted(blocked_pitches))
                duration = self.sample(predictions[2][0, -1], max(.5, temperature * .8),
                                       min(top_k, len(DURATIONS)), rng,
                                       [index for index, value in enumerate(DURATIONS) if value < .125])
                velocity = self.sample(predictions[3][0, -1], rhythm_temperature, min(top_k, 16), rng)
                length = min(float(DURATIONS[duration]), max_beats - onset)
                event = Note(pitch + MIN_PITCH, onset, length, min(127, velocity * 8 + 4))
                notes.append(event)
                active[pitch] = onset + length
                next_input = torch.tensor([[[pitch, step, duration, velocity]]], dtype=torch.long)
                predictions, state = self.model(next_input, state)
        if len(notes) < 2:
            raise ValueError("Generation ended too early; try a longer duration or a different seed.")
        return notes


def export(notes: list[Note], directory: Path, name: str, bpm: int, settings: dict) -> dict:
    if not name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in name):
        raise ValueError("Output name must contain only letters, numbers, hyphens, or underscores.")
    directory.mkdir(parents=True, exist_ok=True)
    midi_path, wav_path = directory / f"{name}.mid", directory / f"{name}.wav"
    write_midi(notes, midi_path, bpm)
    write_audio(notes, wav_path, bpm)
    metadata = {"settings": settings, "bpm": bpm, "note_count": len(notes),
                "duration_seconds": max(note.start + note.duration for note in notes) * 60 / bpm,
                "notes": [note.to_dict() for note in notes]}
    write_json(directory / f"{name}.json", metadata)
    return {"midi": midi_path, "wav": wav_path, "metadata": metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notes", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=.9)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bpm", type=int, default=100)
    parser.add_argument("--name", default="composition")
    parser.add_argument("--output-dir", type=Path, default=GENERATED)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_DIR / "piano_lstm.pt")
    args = parser.parse_args()
    generator = Generator(args.checkpoint)
    settings = {"notes": args.notes, "temperature": args.temperature, "top_k": args.top_k, "seed": args.seed}
    notes = generator.generate(args.notes, args.temperature, args.top_k, args.seed, args.bpm)
    result = export(notes, args.output_dir, args.name, args.bpm, settings)
    print(f"Generated {len(notes)} notes: {result['midi']} and {result['wav']}")


if __name__ == "__main__":
    main()
