# KNOWLEDGE BASE & CONTENT SUMMARY

## 1. Executive Summary

- **Core problem:** Existing Wi-Fi-based Human Sensing (WHS) datasets and algorithms are **obsolete (20/40 MHz bands)**, lack **domain diversity** (environments, people, hardware), and existing HAR (Human Activity Recognition) algorithms **fail to generalize** across new environments, subjects, and days without retraining.
- **Core contribution (two-part):**
  - **Paper 1** delivers a large, domain-diverse **80 MHz IEEE 802.11ac CSI/CFR dataset** (public, on IEEE DataPort) to enable reproducible WHS research across Activity Recognition (AR), Person Identification (PI), and People Counting (PC) tasks.
  - **Paper 2** delivers **SHARP**, a novel signal-processing + deep-learning pipeline that achieves **environment- and person-independent HAR** using only commodity/off-the-shelf (COTS) Wi-Fi routers — no dedicated radar/camera hardware.
- **Key technical insight:** Raw CFR amplitude/phase is **environment-dependent** and unreliable for generalization; the **Doppler shift** (extracted via a novel phase-sanitization technique) isolates *dynamic* (moving-scatterer) information, which is **environment-agnostic** and robust to static-object/furniture changes.
- **Overall impact:** Enables **passive, contactless, non-intrusive, privacy-preserving** (no video/camera) sensing that can be integrated into **commercial, already-deployed Wi-Fi infrastructure**, paving the way for the emerging **IEEE 802.11bf** sensing-enabled Wi-Fi standard (expected devices ~2024).
- **Primary target audience / use cases:**
  - Researchers developing/benchmarking domain-adaptive WHS algorithms (academic reproducibility).
  - Smart-building applications: **assisted living / elderly care, intrusion detection, keyless access, occupancy-based safe-distancing / people counting, smart entertainment, energy optimization**.
  - Course context: used as a **Neural Networks and Deep Learning (NNDL) student project** at the University of Padova, framed around reproducing/extending the SHARP baseline.

---

## 2. Dataset Deep Dive (Paper 1 Focus)

### Dataset Overview
- **Name/Source:** *A CSI Dataset for Wireless Human Sensing on 80 MHz Wi-Fi Channels* — hosted on **IEEE DataPort**, DOI `10.21227/xbhv-f125`.
- **License:** Published under **Creative Commons Attribution 4.0**; access to the DataPort listing noted (per Presentation) as **IEEE-members-only**, though key metadata and pre-extracted Doppler traces are separately linked.
- **Volume:** **>13 hours** of CFR collections, totaling **23.6 GB**.
- **Domain diversity (headline):** **13 subjects** (10 male, 3 female — P1–P13), **7 environments**, **up to 7 activities**, **3 hardware setups**, multiple measurement **days** (April–Dec. 2020 and Jan.–Sept. 2022).
- **Applications supported:** **AR** (Activity Recognition), **PI** (Person Identification), **PC** (People Counting).
- **Standard/Bandwidth:** IEEE **802.11ac**, **80 MHz** channel bandwidth — first dataset of this kind to offer 80 MHz AR/PI/PC data with this level of domain diversity (vs. prior 20/40 MHz datasets in literature, see Table 1 comparison).

### Data Collection & Preprocessing
- **Network setup:** Single Wi-Fi link (one AP/Tx, one station/Rx) built on **OpenWrt**; **IEEE 802.11ac channel 42**, center frequency **5,210 MHz**, **80 MHz** bandwidth.
- **Traffic generation:** **iPerf3** tool; packet rate **173 packets/sec**; new channel estimate every **Tc ≈ 6×10⁻³ s**.
- **Radio configuration:** **MCS 4** (modulation and coding scheme), **no frame aggregation**, **single antenna** on Tx and Rx (to avoid multi-antenna cross-interference at the monitor).
- **Extraction tool:** **Nexmon-CSI** firmware patch — extracts CFR from Broadcom/Cypress IEEE 802.11ac chipsets.
- **Monitor device:** Asus RT-AC86U with **Nant = 4** antennas (used across all three hardware setups).
- **Interference handling:** In all environments except the semi-anechoic chamber, **uncontrolled ambient Wi-Fi interference** was present (CSMA/CA causes packet gaps under interference) — intentionally left in to mirror real-world conditions.
- **File format:** Each acquisition saved as a `.mat` **CFR trace**; a **(N·Nant) × M** dimensional complex matrix (rows = CFR vectors, columns = M monitored OFDM sub-channels).
- **Post-processing script:** A **Python script** for further data processing is made available (from prior SHARP work).

### Data Schema & Variables
- **Trace duration:** 40–300 seconds per acquisition → **N = 6,600–50,000** CFR vectors per monitor antenna per trace.
- **Sub-channels (M):** Of the 256 theoretically available at 80 MHz, Nexmon returns only **M = 242 data sub-channels** (no info on control sub-channels).
- **Folder structure:** **26 sub-folders**, naming convention = `[Prefix][Number][Letter]`:
  - **Prefix**: `AR` / `PI` / `PC` (target application).
  - **Number**: specific combination of environment, hardware, position, day, person.
  - **Letter**: distinct measurement campaign repetition under the same setting.
- **Activity labels (AR, single-letter codes):** `W` = walking, `R` = running, `J` = jumping, `L` = sitting still, `S` = standing still, `C` = sitting down/standing up, `G` = arm exercises; `E` = empty room (no person).
- **Person identifiers:** `P1–P13` (P1, P4, P6–P13 = male; P2, P3, P5 = female).
- **PI files:** suffix `p<ID>` per volunteer (free movement); `p00` = empty room.
- **PC files:** suffix `n<count>` for 1–10 concurrent people; `n00` = empty room (no single-person case, since PI data covers it).
- **Environment metadata (Table 2):** environment type, room dimensions (w × l × h in meters), presence/type of obstruction, device position codes (`M1–M4`, `Tx`, `Rx`), Tx/Rx hardware brand, involved person(s), furniture inventory.
- **7 environments:** bedroom, living room, kitchen, university laboratory, university office, semi-anechoic chamber, meeting room.
- **Hardware sets:** (1) Netgear X4S AC2600 ↔ Netgear X4S AC2600; (2) Asus RT-AC86U ↔ Asus RT-AC86U; (3) Netgear X4S AC2600 ↔ TP-Link AD7200.
- **Obstructions:** wood **bookcase** (bedroom, M2 position) and **20 cm concrete block wall** (meeting room, Tx2 links) — enables NLOS vs. LOS comparison.
- **Semi-anechoic chamber:** absorption factor **110 dB** (1–10 GHz), no furniture/reflectors — near multi-path-free control condition.

### Known Limitations & Edge Cases
- **No localization ground truth:** the precise location of the subject within the room was **not recorded** — dataset is **not suitable for localization/tracking tasks** (only AR/PI/PC).
- **Uncontrolled interference:** most environments (except semi-anechoic chamber) had **uncontrolled co-channel interference** from other Wi-Fi devices, which is a feature (realism) but also a confound for controlled comparisons.
- **Access barrier:** full IEEE DataPort listing is gated as **IEEE-members-only** (per Presentation), though pre-extracted Doppler traces and reproducibility code (CodeOcean capsule `4811042`) are separately available.
- **AR domain diversity is narrower than headline PI/PC numbers:** only **4 volunteers** (not 13) and **6 environments** used for AR; full 13-subject/7-environment diversity applies mainly to PI/PC.
- **Sign-inversion artifact:** CFR values on sub-channels **−63 to 122** require a sign inversion, likely due to hardware artifacts (documented in Paper 2, applies to the same Nexmon extraction pipeline).
- **Domain-shift sensitivity (motivating use case):** CFR traces from the *same empty room* on different days are **almost uncorrelated** (Pearson coefficient analysis, Fig. 3) — demonstrates why domain-robust algorithms are essential, and why naive amplitude/phase features are unreliable across time/environment.

---

## 3. Methodology & Technical Architecture (Paper 2 Focus)

### System/Model Architecture
- **System name:** **SHARP** (Sensing Human Activities through Wi-Fi Radio Propagation).
- **Pipeline (2 major stages):**
  1. **CSI Data Processing** (signal-level, per antenna):
     - **CFR model:** `H_k(n) = A_k(n)·e^{jφ_k(n)}`, sum over multipath components; real CFR includes an unwanted phase offset `φ_offs,k` (Eq. 2).
     - **Phase offset decomposition:** CFO (carrier freq. offset), PPO (phase-locked-loop offset), PA (phase ambiguity) — constant across sub-channels; **SFO** (sampling freq. offset) and **PDD** (packet detection delay) — sub-channel dependent (Eq. 3).
     - **Novel Phase Sanitization method:** formulates path separation as a **Lasso / compressive-sensing minimization problem** (Eq. 8, solved via quadratic programming, e.g. OSQP), decomposing the complex CFR into `T·r` (path-delay dictionary × sparse path-amplitude vector). The **strongest path** is used as the phase reference (no need for a separate reference antenna or cable-connected reference signal) — this is the key novelty enabling full antenna-array spatial diversity to be retained.
     - **Doppler Trace Computation:** Doppler vector `d_i(u)` obtained via **Short-Time Fourier Transform (STFT)** over N subsequent sanitized CFR samples per observation window (Eq. 20); analogous to FMCW radar fast-time/slow-time processing. Result = **Doppler spectrogram**, robust to static objects (walls/furniture), sensitive only to moving scatterers.
  2. **Learning Architecture for HAR** (per-antenna classification + fusion):
     - **Input:** Nw × ND-dimensional Doppler trace matrix per antenna (**Nw=340 stacked vectors ≈ 2 s window; ND=100 velocity bins**).
     - **Feature extractor:** simplified **Inception module** (inspired by Inception-v4 reduction block) with **3 parallel branches** combining MAXPOOL and CONV layers at different kernel sizes (multi-scale feature extraction) — chosen over a full 43M-parameter Inception-v4 or deep stacked-CNN alternatives (MSDNet, RANet, ELASTIC) for lightweight, low-cost-device deployability.
     - **Dimensionality reduction:** 1×1 conv layer reduces feature maps from **15 → 3**.
     - **Classifier head:** Flatten → **Dropout (20%)** → **Dense layer with 5 output neurons** (one per activity class) → softmax-style **activity vector**.
     - **Total parameters:** **128,535** (single-antenna classifier).
     - **Loss function:** **cross-entropy**.
     - **Decision Fusion (multi-antenna, Section 4.2):** each of the **Nant=4** antennas independently classified; if ≥ Nant−1 antennas agree → majority-vote winner; otherwise, activity vectors are **summed element-wise** and the argmax activity is chosen. (Alternative "data fusion" approach — merging antenna data at the network input — was tested and found **less robust**, esp. under NLOS or full environment+subject change, since it forces equal weighting of all antennas including "bad" ones.)

### Experimental Setup
- **Hardware:** Two **Netgear X4S AC2600** routers as Tx/Rx (single antenna each); one **Asus RT-AC86U** (Nant=4 antennas) as passive monitor, using **Nexmon CSI**.
- **Channel:** IEEE 802.11ac channel 42; **MCS 4**; **173 packets/s**; **Tc ≈ 6×10⁻³ s**.
- **Environments (SHARP's own validation subset — smaller than full Paper-1 dataset):** bedroom (with bookcase obstruction), living room, university laboratory — **3 environments**, **3 subjects (P1 male, P2/P3 female)**.
- **Measurement sets S1–S7** (environment-day-person triplets):
  - **S1:** training scenario (baseline env/day/person).
  - **S2:** same env/person, different day (tests time-robustness).
  - **S3:** same env, different day AND different person.
  - **S4/S5:** person moves through bookcase-obstructed NLOS areas of the bedroom.
  - **S6/S7:** different environments/days; **S7 = worst case** — no overlap in room, day, or person vs. training.
- **Per-set recording:** 120 s per activity + 120 s empty-room trace; ~**120 minutes** total CSI data collected across all 4 monitor antennas.
- **Preprocessing parameters (Table 2 in Paper 2):**
  - Amplitude normalized by mean over 242 sub-channels.
  - Phase sanitization regularization `λ = 10⁻¹`.
  - Reconstructed 3 central sub-channels → **245-component** CFR vector.
  - Doppler vector computed from **N = 31** subsequent sanitized CFR samples.
  - Zero-padded to **ND = 100** velocity bins for finer resolution.
  - Noise threshold: remove Doppler contributions **< −12 dB**.
  - Doppler trace = **Nw = 340** stacked vectors (~2 s).
  - **Nant = 4** monitoring antennas.
- **Train/test split:** **60%** of S1 = training set; remaining **40%** split evenly between validation/test; sets **S2–S7 used only for testing** (zero-shot generalization, no retraining).
- **Benchmarks compared:** **DeepSense** [11], **EI** [12], **MatNet-eCSI** [13] — all retrained on the same S1 training portion for fairness.

### Evaluation & Results
- **Primary classification task:** 5-class (4 activities: **walking, running, jumping, sitting** + **empty room**).
- **Headline accuracy:** **>95%** mean accuracy across all test sets; near **100%** when environment/monitor position match training (regardless of day/person).
- **NLOS robustness:** **~97%** accuracy when direct path blocked by bookcase (S4/S5) — main confusion is **running ↔ walking**.
- **Worst-case generalization (S7 — new environment + new person):** **95.99%** average accuracy (this is the paper's core "environment- and person-independent" claim). Confusion matrix (Fig. 7) again shows **walking misclassified as running** as the main error mode.
- **Phase-sanitization ablation (Section 6.3, Table 4):** SHARP's "ref. main path" method vs. literature-standard "ref. one antenna" conjugate-multiplication approach — both similar on S1 (same env/person), but **"ref. one antenna" degrades substantially** on S2–S7 while SHARP's method **stays above 95%**.
- **Fusion ablation (Section 6.4, Table 5):** **Decision fusion outperforms data fusion**, especially in NLOS (S4/S5) and full-domain-shift (S7) cases.
- **Antenna-count ablation (Section 6.5, Fig. 9a):** accuracy scales with number of antennas (**Nsub = 1→4**); even **1 antenna achieves >80%** accuracy; more antennas → more spatial diversity → higher robustness.
- **Benchmark comparison (Section 6.6, Figs. 9b–9c):** DeepSense, EI, MatNet-eCSI perform comparably to SHARP **only on S1** (matching train/test conditions); all **substantially degrade** on S2–S7 (even same-environment/different-day cases), confirming that **amplitude-only or amplitude+phase (non-Doppler) features do not generalize**.
- **8-activity extension (Section 6.7, Table 6):** adding **standing, sitting down/standing up, arm gymnastics** (single-subject, LOS-only evaluation) — SHARP maintains **>95%** accuracy on the expanded activity set. **Note:** standing-up and sitting-down produce **almost indistinguishable Doppler signatures** given system bandwidth/frequency limits; the added 3 activities require **subject-specific** (not fully environment-independent) training to separate reliably.
- **Overall conclusion:** SHARP's **Doppler-shift feature + Inception-based classifier + decision fusion** combination is the key to environment/person-independent generalization; competing amplitude/phase-based methods fail to generalize under domain shift.

---

## 4. Strategic Context & Roadmap (Presentation Focus)

### Business/Research Goals & KPIs
- **Framing (Slide 3):** Wi-Fi routers can be **repurposed as passive radar** for human-movement monitoring with **high accuracy** and **no video recording** — positioned for **healthcare, security, and monitoring** applications.
- **Pedagogical goal:** Present SHARP as the **course baseline model** (Slide 8) that students are expected to **understand, reproduce, and extend**.
- **Implicit KPI (from Paper 2, inherited by the course framing):** achieve **environment- and person-independent classification accuracy > 95%** in worst-case (new room + new person) test conditions, matching or improving on the SHARP benchmark.
- **Key differentiator emphasized to students (Slide 7, "Diversity is key"):** success is measured not just by raw accuracy but by **robustness across environments and people** — i.e., generalization, not overfitting to a single training domain.
- **Success criteria implied by "Challenges and possible objectives" (Slide 9):**
  1. **Cross-subject generalization** — can the model correctly classify activities for unseen people? (proposed approach: contrastive / self-supervised learning).
  2. **Cross-environment & cross-day robustness** — does performance hold as the room or the measurement day changes?
  3. **Person identification** — can subject-discriminative features (from objective 1) be repurposed to identify *who* is performing the activity, not just *what* activity is being performed?

### Project Roadmap
- **Phase 0 — Problem framing (Slides 1–3):** Introduce course staff (Mazzieri, Mari, Pegoraro), cite foundational papers (Paper 1, Paper 2) and GitHub source, establish the wireless-sensing/Doppler-shift rationale.
- **Phase 1 — Conceptual foundation (Slides 4–5):** Teach the Doppler-spectrogram-as-image abstraction (STFT of channel estimates) and its per-activity variability, setting up the CNN-based processing intuition.
- **Phase 2 — Dataset onboarding (Slide 6):** Point students to the IEEE DataPort dataset (members-only) and the **pre-extracted Doppler traces** (a lower-friction entry point bypassing raw CFR preprocessing) — headline stats: 7 activities, 10 people, 7 environments.
- **Phase 3 — Core theme reinforcement (Slide 7):** "Diversity is key" — reiterate that environment and people diversity is the central axis the project must address.
- **Phase 4 — Baseline delivery (Slide 8):** Present the SHARP architecture (Inception-based classifier + decision fusion) as the **starting point / floor** for student work.
- **Phase 5 — Open research directions (Slide 9, "Challenges and possible objectives") — planned future development paths for student projects:**
  - Explore **contrastive / self-supervised learning** for subject-invariant representations.
  - Test/improve robustness to **environment and day changes** beyond SHARP's reported benchmarks.
  - Extend the pipeline toward **person identification** as a secondary task built on the same Doppler features.
- **Longer-horizon research directions (carried over from Papers 1 & 2, beyond the immediate course scope):**
  - **Cross-hardware generalization** (train on Netgear, test on Asus, and vice versa) — flagged in both Paper 1 and Paper 2 as needing further research.
  - **Multi-person concurrent activity recognition** (Paper 2, Sec. 7).
  - **Domain adaptation / few-shot retraining** strategies for rapid adaptation to new domains (Paper 1).
  - **Sub-sampling studies:** impact of reduced sampling time, sub-channel subset selection, and antenna-count trade-offs on sensing accuracy (Paper 1, "Examples of Use").
  - **Sensor fusion with ground truth:** pairing CFR data with **webcam-based localization ground truth** to enable future localization/tracking applications (Paper 1 conclusions) — notably, this is **out of scope** for the current AR/PI/PC dataset, which lacks positional ground truth.
