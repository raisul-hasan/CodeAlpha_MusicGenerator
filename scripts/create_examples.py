"""Regenerate the three listening examples from the saved checkpoint."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from musicgen.generate import Generator, export
from musicgen.paths import EXAMPLES
from musicgen.visualize import piano_roll
import matplotlib.pyplot as plt


def main():
    generator = Generator()
    for name, seed, creativity, bpm in [("focused", 18, .65, 90),
                                         ("balanced", 42, .9, 100),
                                         ("exploratory", 2026, 1.15, 110)]:
        settings = {"seed": seed, "notes": 128, "temperature": creativity, "top_k": 12}
        notes = generator.generate(128, creativity, 12, seed, bpm)
        export(notes, EXAMPLES, name, bpm, settings)
        fig = piano_roll(notes, bpm)
        fig.savefig(EXAMPLES / f"{name}.png", dpi=160, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved {name}: {len(notes)} notes", flush=True)


if __name__ == "__main__":
    main()
