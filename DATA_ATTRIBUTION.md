# Training data attribution

Dataset: MAESTRO v3.0.0 (MIDI and Audio Edited for Synchronous TRacks and Organization).

Provider: Google LLC, in partnership with the International Piano-e-Competition organizers.

Source: https://magenta.tensorflow.org/datasets/maestro

License: Creative Commons Attribution Non-Commercial Share-Alike 4.0 (CC BY-NC-SA 4.0).

License text: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

Archive: https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip

Published and verified SHA-256:

```text
70470ee253295c8d2c71e6d9d4a815189e35c89624b76d22fce5a019d5dde12c
```

Citation: Curtis Hawthorne, Andriy Stasyuk, Adam Roberts, Ian Simon, Cheng-Zhi Anna Huang, Sander Dieleman, Erich Elsen, Jesse Engel, and Douglas Eck. "Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset." International Conference on Learning Representations, 2019.

Paper: https://openreview.net/forum?id=r1lYRjC9F7

The recorded training run uses 96 training performances, with separate 16-performance validation and test subsets from the official MAESTRO splits. Exact selections appear in `reports/dataset_manifest.json`. Raw MIDI downloads and processed datasets are optional retraining files and have been removed during cleanup. Run `python -m musicgen.dataset` to recreate them. The checkpoint contains short encoded training contexts used to initialize generation, so ordinary generation does not require those files.
