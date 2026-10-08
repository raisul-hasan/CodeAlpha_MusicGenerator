"""Piano-roll visualization shared by the app and exported examples."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from .midi import Note


def piano_roll(notes: list[Note], bpm: int = 100):
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor="#151c16")
    ax.set_facecolor("#151c16")
    seconds = 60 / bpm
    patches = [Rectangle((note.start * seconds, note.pitch - .38), note.duration * seconds, .76)
               for note in notes]
    collection = PatchCollection(patches, cmap="summer", linewidth=.2, edgecolor="#b7f397")
    collection.set_array([note.velocity for note in notes])
    collection.set_clim(1, 127)
    ax.add_collection(collection)
    ax.set_xlim(0, max(note.start + note.duration for note in notes) * seconds + .2)
    ax.set_ylim(min(note.pitch for note in notes) - 3, max(note.pitch for note in notes) + 3)
    low, high = ax.get_ylim()
    octaves = [pitch for pitch in range(24, 109, 12) if low <= pitch <= high]
    ax.set_yticks(octaves, [f"C{pitch // 12 - 1}" for pitch in octaves])
    ax.set_xlabel("Time · seconds", color="#93a58f", labelpad=12)
    ax.tick_params(colors="#93a58f", labelsize=9, length=0)
    ax.grid(axis="x", color="#ffffff", alpha=.06)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=1.5)
    return fig
