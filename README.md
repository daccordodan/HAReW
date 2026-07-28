# Wi-Fi HAR — SHARP Baseline Reproduction & Extension (PyTorch)

NNDL course project (University of Padova) reproducing and extending **SHARP**
(Sensing Human Activities through Wi-Fi Radio Propagation, Meneghello et al.,
IEEE TMC 2023).

## Project Constraints (binding for this codebase)

1. **PyTorch only.** All models/training/evaluation code uses `torch`.
2. **No raw-CFR preprocessing.** Doppler traces are already computed and
   provided as pickled NumPy arrays, organized per the SHARP paper's
   `S<set><repetition>` folder convention (e.g. `S1a/`, `S1b/`), with one
   `_stream_{0..3}.txt` file per monitor antenna. `src/data/doppler_trace_dataset.py`
   reads these directly.
3. **Baseline first, extensions separated.** The base SHARP architecture is
   reproduced in `src/training/train_baseline.py` / `evaluate_baseline.py`.
   The three extension tasks live in `src/tasks/task{1,2,3}_*/`, each with
   its own config file and output directory -- none of them are imported by
   or entangled with the baseline code.
4. **Must run on Google Colab.** See `notebooks/00_colab_setup.ipynb`.

## Resolved Label Scope (baseline)

The actual Doppler-trace data contains 8 raw activity codes: `E, H, J1, J2,
L, R, S, W`. Per the project owner's decision, the baseline targets a
**5-class task**: `E` (empty), `W` (walking), `R` (running), `J` (jumping,
merging `J1`+`J2`), `L` (sitting still). `H` and `S` are recorded but
**excluded** from the baseline. See `src/data/label_mapping.py` for the
single source of truth on this mapping.

## Quick Start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Place your Doppler trace folders (S1a/, S1b/, ...) under data/doppler_traces/

# 1. Train the baseline on set S1
python scripts/run_baseline_training.py --config config/base_config.yaml

# 2. Evaluate across whichever of S1-S7 you have data for
python scripts/run_baseline_evaluation.py \
    --config config/base_config.yaml \
    --checkpoint outputs/baseline/checkpoints/sharp_baseline_best.pt
```

## Quick Start (Google Colab)

Open `notebooks/00_colab_setup.ipynb` first -- it installs dependencies,
detects the runtime/device, and (optionally) mounts Google Drive for
persistent storage. Then run `01_data_exploration.ipynb` →
`02_baseline_training.ipynb` → `03_model_evaluation.ipynb` in order.

## Extension Tasks (Phase 2 — scaffolded, not yet implemented)

| Task | Config | Code |
|---|---|---|
| 1. Cross-subject generalization (contrastive/self-supervised) | `config/task1_cross_subject.yaml` | `src/tasks/task1_cross_subject/` |
| 2. Cross-environment/day robustness (domain adaptation) | `config/task2_cross_environment.yaml` | `src/tasks/task2_cross_environment/` |
| 3. Person identification | `config/task3_person_id.yaml` | `src/tasks/task3_person_id/` |

Each scaffold currently raises `NotImplementedError` at the point where a
design decision (augmentation strategy, domain-adaptation method, person-ID
label source) is still open -- see `docs/PROJECT_STATUS.md` for the current
list of gaps before starting implementation.

## Documentation

- `docs/DEVELOPMENT_GUIDE.md` — repository architecture, coding standards, workflow.
- `docs/INDEX.md` — maps every topic to its exact source (paper/section/slide).
- `docs/KNOWLEDGE_BASE.md` — technical deep-dive on dataset, methodology, results.
- `docs/GLOSSARY_AND_VARIABLES.md` — acronyms and variable dictionary.
- `docs/PROJECT_STATUS.md` — current phase, open gaps, next actions.

## Testing

```bash
pytest tests/
```

`tests/test_label_mapping.py` and `tests/test_doppler_trace_dataset.py` run
without `torch`-heavy dependencies beyond what's in `requirements.txt`.
`tests/test_models.py` requires `torch` to be installed (true in Colab, or
after `pip install -r requirements.txt` locally).
