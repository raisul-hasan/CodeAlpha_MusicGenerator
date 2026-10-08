"""Train, validate, and evaluate the LSTM on verified MIDI performances."""
import argparse
import copy
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .dataset import sha256
from .io import read_json, write_json
from .midi import CARDINALITIES, FEATURES
from .model import ModelConfig, MusicLSTM
from .paths import MODEL_DIR, PROCESSED, REPORTS

LOSS_WEIGHTS = (1, .5, .5, .15)


class PieceWindows(Dataset):
    def __init__(self, path: Path, sequence_length: int = 48, stride: int = 24):
        if sequence_length < 2 or stride < 1:
            raise ValueError("Invalid sequence length or stride.")
        with np.load(path, allow_pickle=False) as data:
            self.notes = torch.from_numpy(data["notes"].astype(np.int64))
            self.offsets = data["offsets"].copy()
        self.sequence_length = sequence_length
        self.starts = []
        for left, right in zip(self.offsets[:-1], self.offsets[1:]):
            self.starts.extend(range(int(left), int(right) - sequence_length, stride))
        if not self.starts:
            raise ValueError(f"No training windows in {path}; preprocess longer pieces.")

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        sequence = self.notes[start:start + self.sequence_length + 1]
        return sequence[:-1], sequence[1:]


def evaluate(model, loader, device) -> dict:
    model.eval()
    losses = np.zeros(4)
    correct = np.zeros(4)
    count = 0
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions, _ = model(inputs)
            batch_count = targets.shape[0] * targets.shape[1]
            for index, prediction in enumerate(predictions):
                truth = targets[..., index]
                losses[index] += nn.functional.cross_entropy(
                    prediction.reshape(-1, prediction.shape[-1]), truth.reshape(-1),
                    reduction="sum").item()
                correct[index] += (prediction.argmax(-1) == truth).sum().item()
            count += batch_count
    return {"loss": float(np.dot(losses / count, LOSS_WEIGHTS)),
            "cross_entropy": dict(zip(FEATURES, (losses / count).tolist())),
            "accuracy": dict(zip(FEATURES, (correct / count).tolist())),
            "pitch_perplexity": float(math.exp(min(20, losses[0] / count))),
            "target_events": count}


def baseline_metrics(training: PieceWindows, loader) -> dict:
    probabilities = []
    for index, count in enumerate(CARDINALITIES):
        totals = torch.bincount(training.notes[:, index], minlength=count).double() + 1
        probabilities.append(totals / totals.sum())
    losses, correct, total = np.zeros(4), np.zeros(4), 0
    for _, targets in loader:
        total += targets.shape[0] * targets.shape[1]
        for index, probability in enumerate(probabilities):
            truth = targets[..., index]
            losses[index] += -probability[truth].log().sum().item()
            correct[index] += (truth == probability.argmax()).sum().item()
    return {"loss": float(np.dot(losses / total, LOSS_WEIGHTS)),
            "cross_entropy": dict(zip(FEATURES, (losses / total).tolist())),
            "accuracy": dict(zip(FEATURES, (correct / total).tolist())),
            "pitch_perplexity": float(math.exp(losses[0] / total)), "target_events": total}


def plot_history(history: list[dict], destination: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [entry["epoch"] for entry in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(epochs, [entry["train_loss"] for entry in history], label="Train")
    axes[0].plot(epochs, [entry["validation"]["loss"] for entry in history], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Weighted cross-entropy", title="Learning progress")
    axes[0].legend()
    axes[1].plot(epochs, [entry["validation"]["accuracy"]["pitch"] for entry in history], color="#53852f")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Validation pitch prediction", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def train(epochs: int = 30, batch_size: int = 64, learning_rate: float = .002,
          sequence_length: int = 48, hidden_size: int = 128, seed: int = 42,
          patience: int = 7, device_name: str = "auto", data_dir: Path = PROCESSED,
          model_dir: Path = MODEL_DIR, reports_dir: Path = REPORTS) -> dict:
    if min(epochs, batch_size, patience) < 1 or learning_rate <= 0:
        raise ValueError("Epochs, batch size, patience, and learning rate must be positive.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    config = ModelConfig(hidden_size=hidden_size)
    manifest = read_json(data_dir / "manifest.json")
    manifest_hash = sha256(data_dir / "manifest.json")
    training = PieceWindows(data_dir / "train.npz", sequence_length, max(1, sequence_length // 2))
    validation = PieceWindows(data_dir / "validation.npz", sequence_length, sequence_length)
    testing = PieceWindows(data_dir / "test.npz", sequence_length, sequence_length)
    generator = torch.Generator().manual_seed(seed)
    training_loader = DataLoader(training, batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation, batch_size=batch_size)
    testing_loader = DataLoader(testing, batch_size=batch_size)
    model = MusicLSTM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=.5)
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    baseline = baseline_metrics(training, testing_loader)
    best_loss, best_epoch, stale, history = float("inf"), 0, 0, []
    context_indices = np.random.default_rng(seed).choice(len(training), min(64, len(training)), replace=False)
    seed_contexts = torch.stack([training[int(index)][0] for index in context_indices])
    started = time.perf_counter()
    print(f"Training {sum(p.numel() for p in model.parameters()):,} parameters on {device}; "
          f"{len(training):,} training windows.", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, count = 0., 0
        for inputs, targets in training_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions, _ = model(inputs)
            loss = sum(weight * nn.functional.cross_entropy(
                prediction.reshape(-1, prediction.shape[-1]), targets[..., index].reshape(-1))
                for index, (weight, prediction) in enumerate(zip(LOSS_WEIGHTS, predictions)))
            if not torch.isfinite(loss):
                raise RuntimeError("Training produced a non-finite loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total_loss += loss.item() * len(inputs)
            count += len(inputs)
        metrics = evaluate(model, validation_loader, device)
        entry = {"epoch": epoch, "train_loss": total_loss / count,
                 "validation": metrics, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(entry)
        scheduler.step(metrics["loss"])
        if metrics["loss"] < best_loss - .0001:
            best_loss, best_epoch, stale = metrics["loss"], epoch, 0
            checkpoint = {"format_version": 1, "model_config": config.to_dict(),
                          "state_dict": copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()}),
                          "seed_contexts": seed_contexts, "sequence_length": sequence_length,
                          "cardinalities": list(CARDINALITIES), "seed": seed, "epoch": epoch,
                          "validation": metrics, "dataset_manifest_sha256": manifest_hash,
                          "dataset": manifest["dataset"],
                          "dataset_counts": {name: {"pieces": item["piece_count"], "notes": item["note_count"]}
                                             for name, item in manifest["splits"].items()}}
            temporary = model_dir / "piano_lstm.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(model_dir / "piano_lstm.pt")
        else:
            stale += 1
        write_json(reports_dir / "training_history.json", history)
        print(f"Epoch {epoch:02d}: train={entry['train_loss']:.4f}, val={metrics['loss']:.4f}, "
              f"pitch accuracy={metrics['accuracy']['pitch']:.1%}", flush=True)
        if stale >= patience:
            print("Early stopping: validation loss stopped improving.", flush=True)
            break
    checkpoint = torch.load(model_dir / "piano_lstm.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test = evaluate(model, testing_loader, device)
    report = {"dataset": manifest["dataset"], "dataset_manifest_sha256": manifest_hash,
              "data_counts": checkpoint["dataset_counts"], "model_config": config.to_dict(),
              "parameters": sum(p.numel() for p in model.parameters()), "sequence_length": sequence_length,
              "seed": seed, "device": str(device),
              "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "torch_version": str(torch.__version__), "batch_size": batch_size,
              "initial_learning_rate": learning_rate, "epochs_completed": len(history),
              "best_epoch": best_epoch, "elapsed_seconds": round(time.perf_counter() - started, 2),
              "validation": checkpoint["validation"], "test": test, "unigram_baseline_test": baseline,
              "test_loss_improvement_percent": 100 * (baseline["loss"] - test["loss"]) / baseline["loss"],
              "checkpoint_sha256": sha256(model_dir / "piano_lstm.pt")}
    write_json(reports_dir / "evaluation.json", report)
    write_json(reports_dir / "dataset_manifest.json", manifest)
    plot_history(history, reports_dir / "training_curves.png")
    print(f"Saved best checkpoint from epoch {best_epoch}; test loss={test['loss']:.4f}, "
          f"baseline loss={baseline['loss']:.4f}.", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=.002)
    parser.add_argument("--sequence-length", type=int, default=48)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    train(args.epochs, args.batch_size, args.learning_rate, args.sequence_length,
          args.hidden_size, args.seed, args.patience, args.device)


if __name__ == "__main__":
    main()
