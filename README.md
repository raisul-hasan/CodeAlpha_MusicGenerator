# Cadenza - Music Generation with AI

A working CodeAlpha Task 3 project: a two-layer LSTM trained on real classical piano MIDI, with a local music studio, new sequence generation, MIDI downloads, and synthesized WAV playback. The trained checkpoint and three example compositions are included; you can use the studio without downloading the dataset or training again.

## Start the studio

In PowerShell, from this project folder:

```powershell
.\start.ps1
```

Open **http://127.0.0.1:8501**. Choose a length, tempo, creativity level, and variation seed. Click **Generate composition**, play the audio, and download MIDI or WAV. The same model, seed, and settings reproduce the same composition on the same software environment. The examples tab contains three saved variations.

If PowerShell blocks the script, use the equivalent direct command:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Stop the server with `Ctrl+C`. Use `.\start.ps1 -Port 8502` if port 8501 is occupied.

## Install on another computer

Use Python 3.11-3.13. A GPU is optional; generation and training both support CPU operation.

```powershell
.\setup.ps1
.\start.ps1
```

Setup creates a virtual environment, installs CPU PyTorch if needed, and installs the project and test dependencies. For a compatible NVIDIA GPU, use `setup.ps1 -TorchBuild cu126`. The existing environment on this computer reuses the already installed CUDA-enabled PyTorch; `setup.ps1 -ReuseInstalledPackages` offers that option when creating a new environment.

Manual setup on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

On macOS/Linux, use `.venv/bin/python` instead of `.venv\Scripts\python.exe` and run the same Python module commands. No API key, paid service, external MIDI player, or soundfont is required.

## How the project meets Task 3

| Assignment requirement | Implementation |
| --- | --- |
| Collect MIDI music data | Official MAESTRO v3 MIDI-only archive, downloaded and checked against its published SHA-256 |
| Preprocess note sequences | `musicgen/dataset.py` and `musicgen/midi.py`: extract and encode piano pitches, onset spacing, durations, and velocities |
| Build an RNN/LSTM or GAN | `musicgen/model.py`: four categorical embeddings, a two-layer LSTM, and four prediction heads |
| Train on the dataset | `musicgen/train.py`: actual training, validation, early stopping, checkpoint saving, and independent test evaluation |
| Export MIDI and play/save audio | `musicgen/generate.py`, music21 MIDI export, local WAV synthesis, and Streamlit playback/download controls |

## Reproduce the training pipeline

The default preparation downloads the roughly 56 MB **MIDI-only** MAESTRO archive, verifies its checksum before extraction, and uses a reproducible subset of the official splits: 96 training, 16 validation, and 16 test performances. It retains up to the first 1,536 usable notes from each performance. A dataset manifest records every selected file and skipped file. This is a subset-trained demonstration model.

```powershell
.\.venv\Scripts\python.exe -m musicgen.dataset
.\.venv\Scripts\python.exe -m musicgen.train --device auto
.\.venv\Scripts\python.exe scripts/create_examples.py
```

Training selects the best validation checkpoint and evaluates it once on the untouched test split. Windows never cross performance boundaries; the official composition-level separation is also verified. The test split is not used for checkpoint selection or seed contexts. Seeds control data selection, shuffling, initialization, and sampling. Floating-point results can differ across hardware or library versions.

For a larger run, choose more performances or longer excerpts, then retrain:

```powershell
.\.venv\Scripts\python.exe -m musicgen.dataset --train-pieces 384 --validation-pieces 48 --test-pieces 48 --max-notes 3072
.\.venv\Scripts\python.exe -m musicgen.train --epochs 40 --hidden-size 192 --device auto
```

These commands replace the local processed data and trained checkpoint. Model-only inference remains possible without `data/raw` or `data/processed`; the checkpoint stores a small set of training contexts for initialization.

## Recorded training results

The included 266,265-parameter model trained on **143,759 notes from 96 real MIDI performances**. It completed 17 epochs with early stopping and retained the checkpoint from epoch 10. The validation and test subsets contain 23,906 and 24,414 notes respectively.

| Held-out test metric | Trained LSTM | Training-frequency baseline |
| --- | ---: | ---: |
| Weighted cross-entropy (lower is better) | 5.5066 | 6.5425 |
| Pitch accuracy | 12.31% | 2.54% |
| Pitch perplexity (lower is better) | 29.75 | 57.05 |

The weighted test loss improved by **15.83%** over a frequency-only baseline fitted on training data. The loss combines pitch, spacing, duration, and velocity cross-entropy with weights `1, 0.5, 0.5, 0.15`. Accuracy means exact next-event category prediction; it is not a subjective music quality score. Metrics use next-note targets in non-overlapping evaluation windows; target counts are recorded in the report.

Full results, hardware, configuration, manifest/checkpoint hashes, and per-feature metrics are in `reports/evaluation.json`. Epoch-by-epoch results are in `reports/training_history.json`; `reports/training_curves.png` contains the learning curves. `reports/dataset_manifest.json` preserves the exact dataset selection.

The exact direct dependency versions used for this run are recorded in `requirements-tested.txt`. For that configuration, install with `pip install -r requirements-tested.txt`. PyTorch's CUDA or CPU wheel index determines the installed build.

## Generate from the command line

```powershell
.\.venv\Scripts\python.exe -m musicgen.generate --notes 128 --temperature 0.9 --top-k 12 --seed 42 --bpm 100 --name my_composition
```

Output goes to `outputs/generated/`: a `.mid`, a `.wav`, and a `.json` containing settings and note events. Length supports 32-512 notes, temperature 0.2-2.0, top-k 1-88, and tempo 30-240 BPM. The browser offers a narrower range of practical controls. Generation stops at two minutes of musical time, so the actual note count can be lower than the requested count.

## Representation and audio

- Notes retain their pitch and velocity; spacing and duration are quantized to a small vocabulary of musical beat values, including simultaneous onsets, triplet values, and rests up to eight beats.
- MIDI extraction uses `mido` to pair note-on/off events across channels and tracks. Drum events and pitches outside the piano range (21-108) are excluded. Zero-velocity note-ons are treated as note-offs. Dangling notes are closed at the end of the file. Sustain pedal and tempo curves are not learned; note duration means key-held duration, and playback uses the selected fixed tempo.
- The model warms its hidden state using a training excerpt and samples new notes afterward; the excerpt itself is not copied into the exported music. Sampling limits chords to three simultaneous attacks, orders chord tones from low to high, and prevents overlapping repetitions of the same pitch. For a clearer sketch, playback sampling suppresses human microtiming categories below a quarter beat and durations below an eighth beat; it still samples the remaining rhythm categories from learned probabilities. These are generation safeguards, not additional learned model features.
- `music21` writes the resulting score to MIDI. The WAV renderer uses velocity-sensitive additive synthesis with harmonic decay, attack, and release. It is a piano-like preview rather than a sampled grand piano. Import MIDI into a DAW or notation application to choose a different instrument.
- This compact model creates short experimental sketches. It can repeat patterns, wander harmonically, and lacks explicit long-form composition planning. Generation does not guarantee that a passage is unique relative to the training repertoire.

## Verify the project

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover polyphonic MIDI round trips and tempo, quantization, valid non-silent WAV output, velocity-zero note-offs, dangling notes, piece boundaries, gradient flow into all model heads, real-checkpoint loading, reproducible generation, variation, safe note constraints, and the Streamlit generation flow. Acceptance tests require the included checkpoint. Browser testing uses a separate optional script described in `scripts/browser_check.py`.

## Project layout

```text
app.py                  Local music studio
setup.ps1 / start.ps1    Windows setup and launch helpers
musicgen/               Data pipeline, LSTM, training, generation, MIDI, audio, plotting
models/piano_lstm.pt     Best trained checkpoint and seed contexts
data/raw/               Verified MAESTRO download (not needed for inference)
data/processed/         Cached sequences and piece boundaries
reports/                Actual training metrics, dataset manifest, learning curves
outputs/examples/       Three generated MIDI/WAV examples and piano rolls
outputs/generated/      Your generated compositions
tests/                  Unit and end-to-end acceptance tests
scripts/                Example generation and browser verification
```

## Troubleshooting

- **No trained model:** restore `models/piano_lstm.pt`, or run the data and training commands.
- **CUDA unavailable:** use `--device cpu`. The studio generates on CPU even when the model trained on a GPU.
- **Dependency missing:** install with the project's `.venv` Python, not a different Python installation.
- **Download interrupted:** rerun dataset preparation; partial downloads are never extracted, and the official checksum is verified.
- **Port occupied:** choose another port with `start.ps1 -Port 8502`.
- **Audio sounds synthetic:** use the MIDI download with a sampled piano instrument. The WAV preview deliberately needs no external soundfont.

## Dataset attribution

**MAESTRO v3.0.0**, provided by Google LLC, is available under **Creative Commons Attribution Non-Commercial Share-Alike 4.0**. Dataset source and citation: <https://magenta.tensorflow.org/datasets/maestro>. License: <https://creativecommons.org/licenses/by-nc-sa/4.0/>. Retain the attribution and observe the dataset license when using or redistributing dataset material. See `DATA_ATTRIBUTION.md` for the paper citation and archive checksum.
