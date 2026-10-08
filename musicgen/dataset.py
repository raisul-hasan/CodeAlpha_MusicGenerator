"""Verified MAESTRO collection and piece-separated sequence preparation."""
import argparse
import hashlib
import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .io import read_json, write_json
from .midi import encode_notes, read_midi
from .paths import PROCESSED, RAW

DATASET_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
DATASET_SHA256 = "70470ee253295c8d2c71e6d9d4a815189e35c89624b76d22fce5a019d5dde12c"
DATASET_NAME = "maestro-v3.0.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_dataset(destination: Path = RAW) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination / DATASET_NAME
    metadata = root / f"{DATASET_NAME}.json"
    receipt = destination / "download_receipt.json"
    if metadata.exists() and receipt.exists():
        info = read_json(receipt)
        if info.get("sha256") == DATASET_SHA256:
            print("Using previously verified MAESTRO download.", flush=True)
            return root
    archive = destination / f"{DATASET_NAME}-midi.zip"
    if not archive.exists() or sha256(archive) != DATASET_SHA256:
        partial = archive.with_suffix(".zip.part")
        request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "MusicGeneration/1.0"})
        print("Downloading the MAESTRO v3 MIDI-only archive...", flush=True)
        with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as output:
            downloaded = 0
            while block := response.read(1024 * 1024):
                output.write(block)
                downloaded += len(block)
                if downloaded % (10 * 1024 * 1024) == 0:
                    print(f"Downloaded {downloaded // (1024 * 1024)} MiB", flush=True)
        if sha256(partial) != DATASET_SHA256:
            raise ValueError("Dataset checksum mismatch; the partial archive was not extracted.")
        partial.replace(archive)
    with zipfile.ZipFile(archive) as bundle:
        resolved = destination.resolve()
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(resolved):
                raise ValueError("Unsafe archive member.")
        bundle.extractall(destination)
    if not metadata.exists():
        raise FileNotFoundError("The verified archive does not contain the expected metadata.")
    write_json(receipt, {"dataset": DATASET_NAME, "url": DATASET_URL,
                         "sha256": DATASET_SHA256, "license": "CC BY-NC-SA 4.0"})
    print("Checksum verified; MIDI files extracted.", flush=True)
    return root


def metadata_rows(path: Path) -> list[dict]:
    metadata = read_json(path)
    if isinstance(metadata, list):
        return metadata
    # MAESTRO uses pandas' column-oriented JSON format.
    keys = list(metadata["midi_filename"])
    return [{field: values[key] for field, values in metadata.items()} for key in keys]


def prepare_dataset(root: Path, train_pieces: int = 96, validation_pieces: int = 16,
                    test_pieces: int = 16, max_notes: int = 1536, seed: int = 42,
                    destination: Path = PROCESSED) -> dict:
    if min(train_pieces, validation_pieces, test_pieces) < 1 or max_notes < 64:
        raise ValueError("Use at least one piece per split and at least 64 notes per piece.")
    rows = metadata_rows(root / f"{DATASET_NAME}.json")
    rng = random.Random(seed)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": DATASET_NAME, "source": DATASET_URL, "seed": seed,
                "max_notes_per_piece": max_notes, "representation": "piano note events v1",
                "splits": {}, "skipped": []}
    compositions: dict[str, set[tuple[str, str]]] = {}
    for split, limit in [("train", train_pieces), ("validation", validation_pieces), ("test", test_pieces)]:
        candidates = [row for row in rows if row["split"] == split]
        rng.shuffle(candidates)
        sequences, selected = [], []
        for row in candidates:
            if len(sequences) >= limit:
                break
            relative = Path(row["midi_filename"])
            source = (root / relative).resolve()
            if not source.is_relative_to(root.resolve()):
                raise ValueError("Unsafe dataset path.")
            try:
                notes = read_midi(source)[:max_notes]
                encoded = encode_notes(notes)
                if len(encoded) < 64:
                    raise ValueError("Fewer than 64 usable piano notes.")
            except (ValueError, OSError, EOFError) as error:
                manifest["skipped"].append({"file": str(relative), "reason": str(error)})
                continue
            sequences.append(encoded)
            selected.append({"file": str(relative), "composer": row["canonical_composer"],
                             "title": row["canonical_title"], "notes": len(encoded)})
            if len(sequences) % 16 == 0:
                print(f"{split}: prepared {len(sequences)} pieces", flush=True)
        if len(sequences) < limit:
            raise ValueError(f"Only {len(sequences)} usable {split} pieces; requested {limit}.")
        compositions[split] = {(piece["composer"], piece["title"]) for piece in selected}
        # Flat storage with boundaries; training windows never cross pieces.
        offsets = np.cumsum([0] + [len(sequence) for sequence in sequences], dtype=np.int64)
        np.savez_compressed(destination / f"{split}.npz", notes=np.concatenate(sequences), offsets=offsets)
        manifest["splits"][split] = {"pieces": selected, "piece_count": len(selected),
                                      "note_count": int(offsets[-1])}
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if compositions[left] & compositions[right]:
            raise ValueError(f"Composition leakage between {left} and {right}.")
    manifest["composition_split_verified"] = True
    write_json(destination / "manifest.json", manifest)
    print("Prepared all three splits; composition separation verified.", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pieces", type=int, default=96)
    parser.add_argument("--validation-pieces", type=int, default=16)
    parser.add_argument("--test-pieces", type=int, default=16)
    parser.add_argument("--max-notes", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-dir", type=Path)
    args = parser.parse_args()
    root = args.dataset_dir or download_dataset()
    prepare_dataset(root, args.train_pieces, args.validation_pieces, args.test_pieces,
                    args.max_notes, args.seed)


if __name__ == "__main__":
    main()
