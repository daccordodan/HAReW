# GLOSSARY & VARIABLE DICTIONARY

## 1. Domain Terminology & Acronyms

| Term / Acronym | Source File(s) | Full Definition / Meaning | Alternative Terms / Aliases Used |
| :--- | :--- | :--- | :--- |
| **HAR** | Paper 1, Paper 2, Presentation | Human Activity Recognition — the task of classifying human movements/activities from sensor data. | "activity recognition"; "AR" (dataset-application prefix in Paper 1) |
| **WHS** | Paper 1, Paper 2 (implicitly) | Wi-Fi-based / Wireless Human Sensing — the broader research area covering AR, PI, PC. | "wireless sensing" (Presentation, Slide 3) |
| **CSI** | Paper 1, Paper 2 | Channel State Information — the estimated channel parameters (amplitude/phase) for each sub-channel. | Used **interchangeably with CFR** in Paper 2 (explicit footnote, Sec. 3) |
| **CFR** | Paper 1, Paper 2, Presentation (implicitly) | Channel Frequency Response — complex-valued vector describing attenuation and phase shift per OFDM sub-channel; the raw signal from which sensing features are derived. | "channel estimate" (Presentation, Slide 4); synonymous with CSI |
| **AR** | Paper 1, Presentation | Activity Recognition — one of the three dataset applications; also the sub-folder prefix in Paper 1's dataset. | "activity recognition" |
| **PI** | Paper 1 | Person Identification — dataset application; sub-folder prefix. | "identity recognition" (Paper 1, "Examples of Use") |
| **PC** | Paper 1 | People Counting — dataset application; sub-folder prefix. | — |
| **OFDM** | Paper 1, Paper 2 | Orthogonal Frequency Division Multiplexing — the Wi-Fi modulation scheme transmitting over K/M partially overlapping, orthogonal sub-channels. | — |
| **COTS** | Paper 1 (implicitly), Paper 2 | Commercial/Commodity Off-The-Shelf — standard, unmodified Wi-Fi hardware (no dedicated radar/sensing device). | "commodity Wi-Fi"; "commercial Wi-Fi" |
| **MCS** | Paper 1, Paper 2 | Modulation and Coding Scheme — radio configuration parameter; both papers use **MCS 4**. | — |
| **Nexmon / Nexmon-CSI** | Paper 1, Paper 2 | Firmware patch/tool enabling CFR/CSI extraction from Broadcom/Cypress IEEE 802.11ac chipsets. | "Nexmon CSI extraction tool" |
| **CFO** | Paper 2 | Carrier Frequency Offset — a phase-offset component, constant across sub-channels within an antenna. | Part of `φ_offs,k` decomposition |
| **PPO** | Paper 2 | Phase-locked loop offset (as literally labeled in Paper 2's decomposition of the phase offset) — constant across sub-channels. | — |
| **PA** | Paper 2 | Phase Ambiguity — constant-across-sub-channels phase-offset component. | — |
| **SFO** | Paper 2 | Sampling Frequency Offset — sub-channel-dependent phase-offset component. | — |
| **PDD** | Paper 2 | Packet Detection Delay — sub-channel-dependent phase-offset component. | — |
| **Phase Sanitization** | Paper 2 | SHARP's novel method to remove the unwanted phase offset (`φ_offs,k`) from raw CFR via Lasso/compressive-sensing path separation, using the strongest multipath component as reference. | "phase cleaning," "phase correction" |
| **STFT** | Paper 2 | Short-Time Fourier Transform — used to compute the Doppler vector from a window of N sanitized CFR samples. | Referenced simply as "Fourier Transform" in Presentation, Slide 4 |
| **FMCW** | Paper 2 | Frequency-Modulated Continuous-Wave (radar) — analogy used to explain the fast-time/slow-time structure of the CFR-to-Doppler transformation. | — |
| **Doppler shift / micro-Doppler** | Paper 1, Paper 2, Presentation | Frequency shift caused by moving scatterers (e.g., body parts); the core environment-agnostic sensing feature. | "Doppler shifts" (Presentation, Slide 3 diagram) |
| **Doppler trace / Doppler spectrogram** | Paper 2, Presentation | The matrix formed by stacking consecutive Doppler vectors over time; explicitly used **interchangeably** in Paper 2 (Sec. 3.2). | "Doppler spectrogram" (Presentation, Slides 4–5); treated as an "image" in Presentation |
| **LOS** | Paper 1, Paper 2 | Line-of-Sight — unobstructed direct path between transmitter and monitor/receiver. | — |
| **NLOS** | Paper 1, Paper 2 | Non-Line-of-Sight — obstructed direct path (e.g., by bookcase or concrete wall). | — |
| **SHARP** | Paper 1 (referenced as [4]), Paper 2, Presentation | Acronym for "Sensing Human Activities through Wi-Fi Radio Propagation" — the environment/person-independent HAR system that is the paper's/course's central algorithm. | "our algorithm" (Paper 1); "baseline model" (Presentation, Slide 8) |
| **Signals of opportunity** | Paper 2 | Passive sensing model where the monitor exploits ambient/already-occurring Wi-Fi traffic (rather than dedicated probe packets) for CFR estimation. | — |
| **Decision Fusion** | Paper 2 | SHARP's strategy of independently classifying each antenna's Doppler trace, then combining outputs via majority vote or summed activity vectors. | Contrasted with "Data Fusion" (input-level antenna merging) |
| **Inception module** | Paper 2, Presentation | Multi-branch (MAXPOOL + CONV) neural network feature extractor, inspired by Inception-v4's reduction block; core of SHARP's classifier. | "CNN" (Presentation, Slide 4–5, simplified framing) |
| **DeepSense, EI, MatNet-eCSI** | Paper 2 | Three prior-art HAR algorithms from the literature, used as benchmarks against SHARP. | — |
| **IEEE 802.11bf** | Paper 1, Paper 2 | Emerging Wi-Fi standard (working group est. Sept. 2020) that will natively support joint communication + sensing. | — |
| **OSQP** | Paper 2 | "Operator Splitting Quadratic Program" solver — used to solve the Lasso/compressive-sensing phase-sanitization minimization problem. | — |
| **IEEE DataPort** | Paper 1, Presentation | Public repository hosting the released CSI/CFR dataset. | — |
| **NNDL** | Presentation (implicitly, via context) | Neural Networks and Deep Learning — the University of Padova course for which the Presentation deck was authored. | — |
| **LOCUS** | Paper 1, Paper 2 | EU Horizon 2020 project (Grant No. 871249) that partially funded both papers' research. | — |
| **Tx / Rx / M(onitor)** | Paper 1, Paper 2 | Transmitter / Receiver / Monitor — the three roles in the Wi-Fi sensing network topology. | Monitor also abbreviated as "M" with numeric suffix (M1–M4) denoting *position*, distinct from the sub-channel-count variable "M" (see Section 3, Discrepancy #1) |

---

## 2. Dataset Variable Dictionary

| Variable Name | Data Type | Description | Source File / Section | Equivalent Term in Presentation |
| :--- | :--- | :--- | :--- | :--- |
| **Prefix (AR / PI / PC)** | Categorical (string) | Identifies the target application for a sub-folder of the dataset. | Paper 1, "Dataset Organization" | Not explicitly named; implied by activity/people/environment counts (Slide 6) |
| **Number** (folder-naming component) | Numeric/categorical code | Encodes a specific combination of environment, hardware, position, day, and person. | Paper 1, "Dataset Organization" | Not covered |
| **Letter** (folder-naming component) | Alphabetic | Identifies distinct measurement-campaign repetitions under the same setting (e.g., `a`, `b`, `c`). | Paper 1, "Dataset Organization" | Not covered |
| **Activity code** (`W`,`R`,`J`,`L`,`S`,`C`,`G`,`E`) | Categorical (single letter) | Walking, Running, Jumping, sitting (L=still), Standing still, sitting down/standing up (C), arm exercises (G), Empty room (E). | Paper 1, "Dataset Organization," p.150 | "7 different activities" (Slide 6) — headline count only, no letter codes given |
| **Person ID** (`P1`–`P13`) | Categorical | Volunteer identifier; P1,P4,P6–P13 = male, P2,P3,P5 = female. | Paper 1, "Dataset Domain Diversity" | "10 different people" (Slide 6) — count only |
| **PI suffix** (`p<ID>`, `p00`) | Categorical string | Identifies which volunteer's free-movement trace is in a PI file; `p00` = empty room. | Paper 1, "Dataset Organization" | Not covered |
| **PC suffix** (`n<count>`, `n00`) | Categorical string | Number of concurrent people (1–10) in a PC trace; `n00` = empty room. | Paper 1, "Dataset Organization" | Not covered |
| **Environment type** | Categorical (7 values) | bedroom, living room, kitchen, university laboratory, university office, semi-anechoic chamber, meeting room. | Paper 1, "Measurement Setup" / Table 2 | "7 different environments" (Slide 6) — count only |
| **Room dimensions** (w × l × h) | Numeric (meters) | Physical dimensions of each measurement environment. | Paper 1, Table 2 | Not covered |
| **Obstruction flag / type** | Boolean / categorical | Presence and type of direct-path obstruction (wood bookcase, 20 cm concrete block wall, or none). | Paper 1, Table 2; Paper 2, Figs. 5–6 | Not covered |
| **Device position code** (`M1`–`M4`, `Tx`, `Tx1/2`, `Rx`, `Rx1/2`) | Categorical | Denotes physical placement of monitor/transmitter/receiver devices within an environment. | Paper 1, Table 2 & Fig. 2; Paper 2, Figs. 5–6 | Not covered |
| **Tx/Rx hardware brand** | Categorical | Netgear X4S AC2600, Asus RT-AC86U, or TP-Link AD7200. | Paper 1, "Hardware Specifications" / Table 2 | Not covered |
| **Furniture inventory** | Text/list | List of furniture items present in each environment (affects multipath). | Paper 1, Table 2 | Not covered |
| **CFR trace** (`.mat` file) | Complex-valued matrix, dims (N·Nant) × M | One full acquisition; rows = CFR vectors, columns = monitored OFDM sub-channels. | Paper 1, "Data Description" | "channel estimate" concept (Slide 4), not the file format |
| **N** (CFR vectors per trace) | Integer, range 6,600–50,000 | Number of CFR vectors collected per monitor antenna per trace (40–300 s acquisitions). | Paper 1, "Data Description" | Not covered — **Note:** a *different* variable is also named `N` in Paper 2 (see Discrepancy #2) |
| **M** (sub-channel count) | Integer = 242 | Number of monitored OFDM data sub-channels returned by Nexmon (out of 256 theoretical at 80 MHz). | Paper 1, "Data Description"; Paper 2, Sec. 5.1 (same value, same meaning) | Not covered |
| **Nant** (antenna count) | Integer = 4 | Number of monitor antennas (Asus RT-AC86U) used across all hardware setups. | Paper 1, "Hardware Specifications"; Paper 2, Table 2 | Not covered |
| **Set identifier** (`S1`–`S7`) | Categorical | Denotes a specific environment-day-person triplet used in SHARP's train/test protocol. | Paper 2, Table 1 (Sec. 5.2) | Not covered (Presentation gives no set-level breakdown) |
| **Doppler vector** `d_i(u)` | Numeric vector, length ND=100 | Per-window Doppler power spectrum computed via STFT; `u` = Doppler/velocity index. | Paper 2, Sec. 3.2, Eq. (20) | "Doppler spectrogram" concept (Slide 4), without formal notation |
| **Doppler trace input tensor** | Numeric matrix, Nw × ND = 340 × 100 | Stacked Doppler vectors forming the neural network's per-antenna input. | Paper 2, Sec. 4.1, Fig. 4 | Treated as an "image" for CNN input (Slide 4) |
| **Activity vector** | Numeric vector, length 5 (or 8 in extended task) | Per-antenna classifier output (one score per activity class), used pre- and post-decision fusion. | Paper 2, Sec. 4.1–4.2 | Not covered |
| **λ (Lasso regularization weight)** | Numeric = 10⁻¹ | Weighting parameter for the ℓ1 regularization term in the phase-sanitization minimization problem. | Paper 2, Sec. 6.1, Eq. (8) | Not covered |
| **Tc (channel sampling interval)** | Numeric ≈ 6×10⁻³ s | Time between successive CFR/channel estimates, governed by the 173 packets/s traffic rate. | Paper 1, "Wi-Fi Network Setup"; Paper 2, Table 2 | Not covered |

---

## 3. Discrepancies & Synonym Mapping

1. **"M" is overloaded across documents/contexts.** In Paper 1's Data Description and Paper 2's Sec. 5.1, `M = 242` denotes the **number of monitored OFDM sub-channels**. In Paper 1's Table 2 and Paper 2's Figs. 5–6, `M1`–`M4` denote **monitor device position codes**. These are unrelated variables sharing the same letter — readers should disambiguate by context (numeric value vs. subscript-position label).
2. **"N" has two distinct meanings between Paper 1 and Paper 2.** In Paper 1, `N` = the number of CFR vectors per monitor antenna per trace (6,600–50,000, a function of the 40–300 s acquisition duration). In Paper 2 (Sec. 3.2, Eq. 18–20; Table 2), `N` (or `N = 31` in the preprocessing parameters) denotes the number of **subsequent sanitized CFR samples used to compute one Doppler vector/observation window** — a much smaller, fixed processing-window parameter. These are not interchangeable despite the shared symbol.
3. **"CSI" and "CFR" are synonyms, but usage varies by document.** Paper 2 explicitly states (Sec. 3, footnote) that the two terms are used interchangeably. Paper 1 favors "CFR" throughout its main text but titles itself a "CSI Dataset." The Presentation uses "channel estimate" (Slide 4) as a looser, non-technical stand-in for both.
4. **"Doppler trace" and "Doppler spectrogram" are explicitly synonymous** (Paper 2, Sec. 3.2: "we refer to Doppler trace, or Doppler spectrogram, as the matrix..."). The Presentation uses only "Doppler spectrogram" (Slides 4–5) and additionally frames it informally as an "image," a framing not used in either paper.
5. **GitHub repository naming differs.** Paper 2 (Ref. [1]) cites `github.com/signetlabdei/SHARP`; the Presentation (Slide 2, Ref. [3]) cites `github.com/francescamen/SHARP`. No document clarifies whether these are the same repository (e.g., a personal fork of a lab account) or two distinct sources.
6. **"SHARP" refers to three overlapping-but-distinct scopes** depending on context: (a) the phase-sanitization + Doppler-extraction signal-processing pipeline specifically (Paper 2, Sec. 3); (b) the complete system including the Inception classifier and decision fusion (Paper 2, Secs. 3–4, and Paper 1's brief reference [4]); and (c) the entire course project umbrella, i.e., "the SHARP baseline" that students extend (Presentation, Slide 8). Readers should infer the intended scope from context.
7. **"Activities" count is presentation-simplified vs. paper-precise.** The Presentation's "7 different activities" (Slide 6) matches Paper 1's *full dataset* letter-code inventory (W,R,J,L,S,C,G, excluding E) but does not match Paper 2's *primary* SHARP evaluation, which uses only 4 activities + empty room (5-class task, Table 3), extending to 8 total only in the secondary single-subject experiment (Sec. 6.7, Table 6). The Presentation does not distinguish between these scopes.
8. **"People" count is a headline simplification.** The Presentation's "10 different people" (Slide 6) corresponds to Paper 1's PI/PC subject pool, not to the AR-specific subject count (4 volunteers) nor to SHARP's own validation subset (3 subjects, P1–P3, Paper 2 Sec. 5.2). The three documents are describing three different subject pools under the same headline phrase.
9. **"Monitor" vs. "receiver" terminology can blur roles.** In the AR/PI/PC dataset (Paper 1) and SHARP (Paper 2), the "monitor" (M) is a **passive, third device** distinct from the active Tx/Rx communication pair — it is not simply a synonym for "receiver" (Rx), even though both receive Wi-Fi signals. The Presentation's Slide 3 diagram ("Wi-Fi router" / "Laptop") simplifies this three-device topology into a generic two-node illustration, which could be misread as the full experimental setup.
10. **Real Doppler-trace data drop uses activity codes not documented in either paper.** Direct inspection of the project owner's actual pre-computed Doppler-trace files (folders `S1a`/`S1b`) found 8 raw activity codes: `E, H, J1, J2, L, R, S, W`. This does **not** match Paper 1's 7-letter AR code table (`W,R,J,L,S,C,G`, +`E`) exactly: `C` and `G` are absent from the real data, while `H` (undefined in either paper) and a two-way split of jumping into `J1`/`J2` (variant unspecified) are present instead. Per project owner decision, the baseline task uses 5 classes (`E,W,R,J,L`, with `J1`+`J2` merged and `H`/`S` excluded) — see `src/data/label_mapping.py` in the generated repo, and `PROJECT_STATUS.md`'s "New Open Gaps" for the still-unresolved meaning of `H`.
