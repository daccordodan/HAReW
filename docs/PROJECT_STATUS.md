# PROJECT STATUS & ARCHITECTURE OVERVIEW

## 1. Project Context & Objectives

- **What this is:** A Neural Networks and Deep Learning (NNDL) course project at the University of Padova, centered on **Wi-Fi-based Human Activity Recognition (HAR)**. The project is built on two published research artifacts — a public CSI/CFR dataset and the **SHARP** methodology — which students are expected to understand, reproduce, and extend.
- **Primary goal:** Achieve **environment- and person-independent** activity recognition from commodity Wi-Fi CSI data, using the Doppler-shift feature extraction + Inception-based classifier pipeline (SHARP) as the **baseline** to match or improve upon.
- **Core technical bet:** Doppler shift (not raw amplitude/phase) is the environment-agnostic feature that enables generalization across new rooms, people, and days without retraining.

**Key stakeholders:**
- **Course staff (project setters):** Riccardo Mazzieri, Daniele Mari, Jacopo Pegoraro (Univ. of Padova) — authors of the Presentation deck, not of the source papers.
- **Original researchers (methodology/dataset authors):** Francesca Meneghello, Nicolò Dal Fabbro, Domenico Garlisi, Ilenia Tinnirello, Michele Rossi (Univ. of Padova / Univ. of Palermo / CNIT).
- **End users:** Students (or AI agents) tasked with reproducing/extending the SHARP baseline.

**High-level milestones (per course framing):**
1. Understand wireless sensing rationale and Doppler-spectrogram feature extraction.
2. Onboard onto the public dataset (or pre-extracted Doppler traces).
3. Reproduce the SHARP baseline (Inception classifier + decision fusion).
4. Extend the baseline along one or more open research directions (cross-subject generalization, cross-environment robustness, person identification).

---

## 2. Knowledge Architecture & File Inventory

| File | Role |
| :--- | :--- |
| `INDEX.md` | Navigation map / topic-locator matrix. Maps every subject (goals, dataset schema, preprocessing, model architecture, results, roadmap) to its exact source, page, section, or slide. Includes a discrepancy log. Use this first to find **where** something is documented. |
| `KNOWLEDGE_BASE.md` | Deep-dive technical synthesis of all three sources: dataset specs and limitations (Paper 1), SHARP methodology and validation metrics (Paper 2), and strategic framing/roadmap (Presentation). Use this to find **what** the answer is without re-reading the source PDFs. |
| **Paper 1 (PDF)** — *A CSI Dataset for Wireless Human Sensing on 80 MHz Wi-Fi Channels* | Source dataset specification. Defines hardware, environments, collection methodology, folder structure, file-naming conventions, and known limitations for the public AR/PI/PC CSI dataset. |
| **Paper 2 (PDF)** — *SHARP: Environment and Person Independent Activity Recognition With Commodity IEEE 802.11 Access Points* | Source methodology and experimental results. Defines the phase-sanitization algorithm, Doppler extraction pipeline, Inception-based neural network classifier, decision fusion strategy, and all benchmark/ablation results. |
| **Presentation (PDF)** — *Human Activity Recognition with Wi-Fi* (NNDL Project Proposals, 9 slides) | Source strategic deck. Introduces the task to students, summarizes Papers 1 & 2 at a high level, links to the dataset/GitHub, and frames the open challenges/objectives for the course project. |

---

## 3. Work Completed To Date

- [x] Initial document ingest and cross-document mapping.
- [x] Extraction of dataset schema and limitations into `KNOWLEDGE_BASE.md`.
- [x] Mapping of methodology and validation metrics into `KNOWLEDGE_BASE.md`.
- [x] Alignment check between high-level presentation and technical papers (`INDEX.md` Discrepancy & Gap Log).

---

## 4. Current Status & Next Steps

- **Current Phase:** Baseline Implementation (PyTorch) — architecture rebuilt around four binding project constraints: (1) PyTorch only, (2) no raw-CFR preprocessing (Doppler traces supplied pre-computed), (3) baseline implemented first with the 3 extension tasks structurally separated, (4) must run on Google Colab. Core baseline modules (dataset, models, training/eval loops) are implemented and unit-tested; extension tasks (`src/tasks/task{1,2,3}_*/`) remain scaffolds pending design decisions below.

- **Resolved Decisions (this phase):**
  - **Baseline label scope:** confirmed with the project owner as 5 classes — `E, W, R, J, L` — with `J1`+`J2` merged into a single `J` class, and `H`/`S` excluded from the baseline entirely. Centralized in `src/data/label_mapping.py`.
  - **Data format confirmed against a real sample:** pickled NumPy `float64` arrays, shape `(n_time_steps, 100)`; `ND=100` verified directly (not just inferred from the papers). Filename convention `{Set}{Letter}_{ActivityCode}_stream_{AntennaIdx}.txt` confirmed, `AntennaIdx` ∈ {0,1,2,3} confirming `Nant=4`.
  - **Raw-CFR preprocessing modules removed:** `src/preprocessing/` (phase sanitization, Doppler/STFT computation, `.mat` parsing) deleted from the repo per constraint #2 — not needed since Doppler traces are supplied pre-computed.

- **New Open Gaps (discovered during this phase):**
  - **New activity-code discrepancy:** the real dataset drop contains raw codes `E, H, J1, J2, L, R, S, W` — 8 codes, none of which is `C` or `G` from Paper 1's letter-code table, and `J` is split into two sub-variants (`J1`/`J2`, meaning unconfirmed — e.g. two-footed vs. one-footed jump). `H` has no definition anywhere in Paper 1 or Paper 2's activity tables; this is a NEW instance of the H-ambiguity (related to, but distinct from, the previously logged "H in a `.h5` model filename" gap) and remains unresolved.
  - **Missing Scenario 6 data:** the owner's upload was named suggesting "Scenario 1 and 6" but only contained `S1a`/`S1b` — no `S6` folders were actually present. Needed before any real S6 zero-shot evaluation can run.
  - **Corrupt file found in real data:** `S1b_L_stream_3.txt` is 0 bytes (truncated/corrupted). The dataset loader now handles this gracefully (skips the affected recording, logs it in `summary()["corrupt_files"]`), but the underlying file itself is still missing/broken and may need re-uploading if `S1b` "L" (sitting-still) samples are needed for a complete S1 evaluation.
  - **Extension task design choices still open** (blocking `src/tasks/` implementation):
    - Task 1: contrastive-learning augmentation strategy for Doppler traces not yet chosen.
    - Task 2: domain-adaptation strategy (adversarial vs. batch-norm recalibration vs. environment-conditioned augmentation) not yet chosen.
    - Task 3: person-ID label source not yet confirmed against real filenames (unlike activity codes, no person-ID suffix has been verified in the actual data drop).

- **Pending Questions / Gaps (carried over from earlier phases):**
  - **Subject/environment count mismatch:** Presentation states "10 people, 7 environments," which matches neither Paper 1's AR subset (4 subjects, 6 environments) nor SHARP's own validation set (3 subjects, 3 environments) — needs clarification before quoting headline numbers to stakeholders.
  - **GitHub repo mismatch:** Paper 2 cites `github.com/signetlabdei/SHARP`; the Presentation cites `github.com/francescamen/SHARP`. Unclear if these are the same repo (fork/rename) — needs verification before pointing students/agents to a single canonical source.
  - **No quantitative baseline in the deck:** The Presentation gives no accuracy/F1 numbers, so any target metrics for the student project must be sourced from Paper 2 directly (Tables 3–6).
  - **Activity-class scope ambiguity:** unclear whether the course project's baseline should target the primary 5-class SHARP task (4 activities + empty room) or the extended 8-activity variant (which requires subject-specific training, per Paper 2 Sec. 6.7).

- **Recommended Next Actions:**
  1. **Obtain missing data:** upload the missing `S6` folders (and any of `S2-S5`, `S7` needed for full zero-shot evaluation per Paper 2 Table 3), and re-upload a working `S1b_L_stream_3.txt` if complete S1 "sitting still" coverage is needed.
  2. ~~Decide between raw CFR vs. pre-extracted Doppler traces~~ — **resolved**: project works exclusively from pre-computed Doppler traces (constraint #2); `src/preprocessing/` removed from the repo.
  3. **Run baseline training/evaluation** (`scripts/run_baseline_training.py` → `scripts/run_baseline_evaluation.py`) once sufficient S1 data is confirmed clean, and validate against Paper 2's Table 3 metrics (>95% mean accuracy, 95.99% on S7) before starting any extension task.
  4. **Scope the extension track:** for whichever of the three tasks is picked up first, resolve its specific open design gap (see "New Open Gaps" above) before writing implementation code into the corresponding `src/tasks/task{N}_*/` scaffold.
