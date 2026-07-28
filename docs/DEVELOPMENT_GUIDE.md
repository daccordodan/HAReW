# DEVELOPMENT GUIDE — Wi-Fi HAR / SHARP Reproduction & Extension (PyTorch)

**Revision note:** this guide replaces an earlier version of the repo that
assumed raw-CFR preprocessing (`.mat` parsing, Lasso phase sanitization) and
a framework-agnostic model stub. It was rebuilt to satisfy four binding
project constraints from the owner:

1. **PyTorch only.**
2. **No raw-CFR preprocessing** — Doppler traces are already computed,
   pickled, and organized per the SHARP paper's convention.
3. **Baseline first, extension tasks structurally separated.**
4. **Must run on Google Colab.**

All architecture decisions below were validated directly against a real
sample of the owner's data (see "Data Format" section), not assumed from
the papers alone.

---

## 1. Repository Directory Tree

```
project_workspace/
├── README.md
├── requirements.txt
├── config/
│   ├── base_config.yaml               # baseline SHARP reproduction
│   ├── task1_cross_subject.yaml       # extension 1: contrastive/self-supervised
│   ├── task2_cross_environment.yaml   # extension 2: domain adaptation
│   └── task3_person_id.yaml           # extension 3: person identification
├── docs/
│   ├── INDEX.md
│   ├── KNOWLEDGE_BASE.md
│   ├── GLOSSARY_AND_VARIABLES.md
│   ├── PROJECT_STATUS.md
│   └── DEVELOPMENT_GUIDE.md
├── data/
│   └── doppler_traces/          # drop your S1a/, S1b/, S6a/, ... folders here
├── notebooks/
│   ├── 00_colab_setup.ipynb     # Colab bootstrap: deps, device, optional Drive mount
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_training.ipynb
│   └── 03_model_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── label_mapping.py           # THE single source of truth for class scope
│   │   └── doppler_trace_dataset.py   # PyTorch Dataset reading pickled traces directly
│   ├── models/
│   │   ├── __init__.py
│   │   ├── inception_module.py        # real PyTorch 3-branch Inception block
│   │   ├── sharp_classifier.py        # real PyTorch classifier (~128K params)
│   │   └── decision_fusion.py         # majority-vote + summed-vector fusion
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_baseline.py          # BASELINE ONLY -- trains on S1
│   │   ├── evaluate_baseline.py       # BASELINE ONLY -- zero-shot eval on S1-S7
│   │   └── losses.py
│   ├── tasks/                          # EXTENSION TASKS -- structurally isolated
│   │   ├── __init__.py
│   │   ├── task1_cross_subject/
│   │   │   ├── __init__.py
│   │   │   └── contrastive_encoder.py
│   │   ├── task2_cross_environment/
│   │   │   ├── __init__.py
│   │   │   └── domain_adaptation.py
│   │   └── task3_person_id/
│   │       ├── __init__.py
│   │       └── person_classifier.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py                 # accuracy/F1/confusion matrix
│   └── utils/
│       ├── __init__.py
│       ├── config_loader.py
│       ├── logger.py
│       └── colab_utils.py             # Colab detection, Drive mount, device selection
├── scripts/
│   ├── run_baseline_training.py
│   ├── run_baseline_evaluation.py
│   ├── run_task1.py
│   ├── run_task2.py
│   └── run_task3.py
├── outputs/
│   ├── baseline/{checkpoints,logs,figures}/
│   ├── task1_cross_subject/{checkpoints,logs}/
│   ├── task2_cross_environment/{checkpoints,logs}/
│   └── task3_person_id/{checkpoints,logs}/
└── tests/
    ├── test_label_mapping.py
    ├── test_doppler_trace_dataset.py
    └── test_models.py
```

Note there is **no `src/preprocessing/` folder** in this version -- it was
deliberately removed. Raw-CFR parsing (`.mat` files), Lasso phase
sanitization, and STFT-based Doppler computation are **not needed**, since
Doppler traces are supplied pre-computed. If a future phase of the project
needs to work from raw CFR after all, that preprocessing code can be
reintroduced as its own separate module -- it should not be resurrected
inside `src/data/` or `src/models/`.

---

## 2. Data Format (verified against a real sample, not assumed)

Confirmed directly by loading the owner's uploaded files:

- **Folder convention:** `S<SetNumber><RepetitionLetter>/`, e.g. `S1a/`, `S1b/`
  — matches Paper 2 Table 1's set/repetition structure.
- **Filename convention:** `{Set}{Letter}_{ActivityCode}_stream_{AntennaIdx}.txt`,
  e.g. `S1a_W_stream_0.txt`. `AntennaIdx` ∈ {0,1,2,3} → `Nant=4`.
- **File contents:** a **pickled NumPy `float64` array**, shape
  `(n_time_steps, 100)`. `100` = `ND` (velocity bins). `n_time_steps` varies
  per recording and is **not** pre-windowed — `doppler_trace_dataset.py`
  performs the `Nw=340` sliding-window step itself.
- **Values:** already normalized, roughly in `[0.06, 1.0]`.
- **Known data issue:** at least one file in the owner's sample
  (`S1b_L_stream_3.txt`) is **0 bytes** (corrupted/truncated). The loader
  catches this per-recording (skips that whole 4-antenna group) and reports
  it via `DopplerTraceDataset.summary()["corrupt_files"]` rather than
  crashing. Always check this field after building a dataset.
- **Activity codes actually present:** `E, H, J1, J2, L, R, S, W` — 8 raw
  codes, not the 7-code set (`W,R,J,L,S,C,G`) listed in Paper 1. See
  `docs/PROJECT_STATUS.md` for this newly-confirmed discrepancy.
- **Resolved label scope (owner's decision):** baseline targets 5 classes
  — `E, W, R, J, L` — with `J1`+`J2` merged into `J`, and `H`/`S` excluded
  entirely. This is centralized in `src/data/label_mapping.py`; no other
  module should hardcode raw activity codes.
- **Data completeness gap:** only `S1a`/`S1b` were confirmed present in the
  owner's initial upload, despite it being named for "Scenario 1 and 6" —
  `S6` was not actually included. Zero-shot evaluation on missing sets is
  handled gracefully (`evaluate_baseline.py` logs a warning and skips a set
  with no data) but won't produce a real S6 result until that data arrives.

---

## 3. Module Rationale

### `src/data/label_mapping.py`
The single source of truth for which raw activity codes count as target
classes. Every other module (dataset, training, evaluation) imports from
here rather than hardcoding `"W"`, `"H"`, etc. Changing the label scope
later (e.g. moving to Paper 2's 8-class extended task, Sec. 6.7) means
editing exactly one file.

### `src/data/doppler_trace_dataset.py`
A `torch.utils.data.Dataset` that:
1. Discovers and parses stream files by regex (verified against real filenames).
2. Groups the 4 per-antenna files for each (set, repetition, activity) recording.
3. Skips groups with a missing or corrupt antenna file (see "Data Format" above).
4. Slides an `Nw=340` window over the time axis to produce individual samples.
5. Returns `(Nant, Nw, ND)` tensors — antennas stay grouped, since
   `decision_fusion.py` needs all 4 predictions for the *same* physical window.
6. Enforces Paper 2's train/test protocol: `build_train_val_split()` only
   ever touches set `S1` (60/20/20 split); `build_zero_shot_test_set()`
   explicitly rejects `S1` to prevent it from being mistakenly used as a
   "zero-shot" set.

### `src/models/`
Real, working PyTorch modules (not stubs):
- `inception_module.py` — 3 branches (1x1→3x3 conv, 1x1→5x5 conv, maxpool→1x1
  conv), concatenated to 15 channels, matching the paper's stated feature-map
  count before reduction.
- `sharp_classifier.py` — adds the 1x1 reduction (15→3), a **2×2 max-pool**
  (added specifically to bring the flattened dense-layer size down to a
  parameter count close to the paper's reported 128,535 — without it, a
  naive flatten of a 340×100 feature map would produce a dense layer with
  ~4x too many parameters), dropout, and the final linear classifier head.
  Validated analytically at 128,443 params (99.9% of the paper's figure).
- `decision_fusion.py` — majority vote (≥ Nant−1 agreement) falling back to
  summed-probability argmax, per Paper 2 Sec. 4.2.

### `src/training/train_baseline.py` / `evaluate_baseline.py`
- **Training** flattens the `(batch, Nant, Nw, ND)` batch into
  `(batch*Nant, 1, Nw, ND)` — the paper trains **one shared classifier**
  applied per-antenna, not 4 independent networks. Each antenna's window
  becomes its own training example with the same label.
- **Evaluation** keeps antennas grouped and applies `decision_fusion.py` to
  produce one fused prediction per sample, matching actual SHARP inference.
- Both are Colab-safe: device selection and output paths route through
  `src/utils/colab_utils.py`.

### `src/tasks/task{1,2,3}_*/`
Structurally isolated per project constraint #3:
- Each has its own config file (`config/task{N}_*.yaml`) and output
  directory (`outputs/task{N}_*/`).
- `task2` and `task3` may **read** a baseline checkpoint as a starting
  point (one-directional dependency), but `train_baseline.py` never
  imports anything from `src/tasks/`.
- All three are currently scaffolds (`NotImplementedError`) pending open
  design decisions — see `docs/PROJECT_STATUS.md` for exactly what's
  blocking each one.

### `src/utils/colab_utils.py`
Centralizes the only two things that differ between a local run and Colab:
device selection (`cuda` vs `cpu`) and optional Google Drive mounting for
persistent data/output paths. No other module should call
`google.colab.drive` directly.

---

## 4. Coding Standards

- **Language/version:** Python ≥ 3.10, PyTorch ≥ 2.2.
- **Style:** PEP 8 via `black`/`flake8`. Type hints on all public signatures.
- **Docstrings:** every module states which paper/section it implements,
  and explicitly notes when a design choice (e.g. kernel sizes) is a
  reconstruction rather than a disclosed paper parameter.
- **No hardcoded label scope:** always route through `label_mapping.py`.
- **No cross-imports from `src/tasks/` into `src/training/` or `src/data/`
  in the reverse direction** — tasks may depend on the baseline; the
  baseline must never depend on tasks.
- **Testing:** `pytest tests/`. Dataset/label tests use small synthetic
  pickled files (not the real dataset, which isn't checked into the repo)
  and specifically cover the corrupt-file case found in real data.

## 5. Dependency Management

See `requirements.txt` — trimmed to only what's needed given constraint #2
(no `scipy`/`h5py`/`osqp` for `.mat`/Lasso work). Install via
`pip install -r requirements.txt`; Colab already ships `torch`, so that line
is typically a no-op there.

## 6. Execution Workflow

**Local:**
```bash
pip install -r requirements.txt
# copy your S1a/, S1b/, ... folders into data/doppler_traces/
python scripts/run_baseline_training.py --config config/base_config.yaml
python scripts/run_baseline_evaluation.py --config config/base_config.yaml \
    --checkpoint outputs/baseline/checkpoints/sharp_baseline_best.pt
```

**Google Colab:** run `notebooks/00_colab_setup.ipynb`, then
`01_data_exploration.ipynb` → `02_baseline_training.ipynb` →
`03_model_evaluation.ipynb`, in order.

**Extension phase:** once baseline S1–S7 accuracy is validated against
Paper 2 Table 3 (>95% mean, 95.99% on S7), resolve the open design
questions in `docs/PROJECT_STATUS.md` for whichever of the 3 tasks you
pick up first, then implement into the corresponding `src/tasks/task{N}_*/`
scaffold and its own config/output directory.

---

**Sources Referenced:**
- Real data sample provided by the project owner (Doppler-trace pickled arrays, S1a/S1b)
- `KNOWLEDGE_BASE.md, Section 3 (Methodology & Technical Architecture)`
- `INDEX.md, Topic Location Matrix (Model Architecture, Evaluation rows)`
- `GLOSSARY_AND_VARIABLES.md, Section 2 (Dataset Variable Dictionary)`
- `PROJECT_STATUS.md, Section 4 (Recommended Next Actions)`
