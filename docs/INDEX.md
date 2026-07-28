# INDEX & DOCUMENT MAP

## 1. Quick Source Reference

| Document Name | Document Type | Primary Role in the Project |
| :--- | :--- | :--- |
| **Paper 1** — *A CSI Dataset for Wireless Human Sensing on 80 MHz Wi-Fi Channels* (Meneghello et al., IEEE Communications Magazine, Sept. 2023, 7 pages) | Dataset Paper | Describes the public CSI/CFR dataset (activity recognition, person identification, people counting) that underlies the project. Defines data collection hardware, environments, folder structure, and file-naming conventions. |
| **Paper 2** — *SHARP: Environment and Person Independent Activity Recognition With Commodity IEEE 802.11 Access Points* (Meneghello et al., IEEE Trans. Mobile Computing, Oct. 2023, 16 pages) | Technical/Methodology Paper | Presents the core methodology (phase sanitization → Doppler extraction → Inception-based neural network classifier → decision fusion) that the student project is expected to reproduce, benchmark, or extend. |
| **Presentation** — *Human Activity Recognition with Wi-Fi* (Project Proposals, NNDL course, Univ. of Padova, 2023–2024, 9 slides) | Presentation / Project Proposal Deck | High-level project brief introducing the task to students, summarizing Papers 1 & 2, pointing to the dataset/GitHub, and framing open challenges/objectives for the course project. |

---

## 2. Topic Location Matrix

| Information Category / Topic | Source Document | Exact Section / Page / Slide | Key Elements / Variables Covered |
| :--- | :--- | :--- | :--- |
| **Project framing / task introduction** | Presentation | Slide 1 (title/authors), Slide 2 (reference papers) | Course instructors (Mazzieri, Mari, Pegoraro), citations to Paper 1 & Paper 2, GitHub link `github.com/francescamen/SHARP` |
| **Business/Research Goals & Scope** | Paper 1 | p.146, Abstract & "Introduction" | Goal: provide a domain-diverse 80 MHz CSI dataset for AR, PI, PC; motivates need for generalizable/domain-adaptive algorithms |
| | Paper 2 | p.6160–6161, Abstract & Sec. 1 "Introduction" | Goal: build an environment- and person-independent HAR system (SHARP) using COTS Wi-Fi; scope limited to *dynamic* activities (static poses out of scope) |
| | Presentation | Slide 3 "Wireless sensing" | Restates rationale (contactless sensing, healthcare/security/monitoring use cases, no video recorded) |
| | Presentation | Slide 9 "Challenges and possible objectives" | Reframes research goals as student objectives: cross-subject generalization, cross-environment robustness, person identification |
| **Related Work / Existing Datasets Comparison** | Paper 1 | p.147, "Background and Existing Datasets" + Table 1 | Comparison vs. datasets [9]-[15]: days, environments, people, activities, Tx hardware, standard (802.11n/ac), bandwidth (20/40/80 MHz) |
| | Paper 2 | p.6161–6163, Sec. 2 "Related Work" (2.1–2.3) | CSI-based HAR literature (E-eyes, CARM, DeepSense, EI, MatNet-eCSI), CFR phase-correction literature, Doppler-based sensing literature |
| **Dataset Schema, Metadata & Variables (Paper 1's own dataset)** | Paper 1 | p.148–150, "Data Description" + "Dataset Organization" + Table 2 | 26 sub-folders; prefixes AR/PI/PC; `.mat` CFR trace files; matrix dims (N·Nant)×M; M=242 sub-channels; N=6,600–50,000 vectors/trace; activity letters W/R/J/L/S/C/G; person IDs P1–P13; environment dimensions, furniture, obstruction flags |
| **Dataset Schema, Metadata & Variables (SHARP's own sub-collection)** | Paper 2 | p.6168–6170, Sec. 5.2 "Dataset Acquisition and Organization" + Table 1 + Table 2 | Sets S1–S7 (environment-day-person triplets); 3 environments (bedroom, living room, lab); persons P1–P3; 120s/activity recordings; Table 2 lists channel/Doppler parameters (Tc, N, ND, Nw, Nant) |
| | Presentation | Slide 6 "The dataset" | Public-facing summary: IEEE DataPort link, "IEEE-members only" access caveat, pre-extracted Doppler traces link, headline counts (7 activities, 10 people, 7 environments) |
| **Hardware Specifications** | Paper 1 | p.148, "Hardware Specifications" | Netgear X4S AC2600, Asus RT-AC86U, TP-Link AD7200 routers; Nant=4 monitor antennas; Nexmon-CSI extraction tool |
| | Paper 2 | p.6168–6169, Sec. 5.1 "Nexmon Extraction Tool" | Asus RT-AC86U as monitor (Nant=4), Netgear X4S AC2600 as Tx/Rx, 242 data sub-channels, sign-inversion artifact on sub-channels −63 to 122 |
| **Data Collection Methodology & Wi-Fi Network Setup** | Paper 1 | p.147–148, "Experimental Setup" → "Wi-Fi Network Setup and CFR Data Collection" | OpenWrt-based network, IEEE 802.11ac channel 42 (5,210 MHz center freq.), 80 MHz bandwidth, iPerf3 traffic generator, 173 packets/s, MCS 4, single antenna Tx/Rx, Tc≈6×10⁻³ s |
| | Paper 2 | p.6169–6170, Sec. 5.2 | Two active terminals (Tx/Rx) + one passive monitor (signals-of-opportunity model), MCS 4, 173 packets/s, Tc≈6×10⁻³ s |
| **Measurement Environments/Obstructions** | Paper 1 | p.148–149, "Measurement Setup" + Fig. 1/Fig. 2 + Table 2 | 7 environments (bedroom, living room, kitchen, lab, office, semi-anechoic chamber, meeting room); obstructions: wood bookcase, concrete block wall; semi-anechoic chamber absorption (110 dB, 1–10 GHz) |
| | Paper 2 | p.6169, Fig. 5 & Fig. 6 | Bedroom (with bookcase obstruction), living room, university laboratory layouts for sets S1–S7 |
| **Preprocessing & CSI/Phase Cleaning (Core Algorithm)** | Paper 2 | p.6163–6166, Sec. 3 "CSI Data Processing" → 3.1 "Phase Sanitization", 3.2 "Doppler Trace Computation" | Eqs. (1)–(22): CFR model, phase offset decomposition (CFO, PPO, PA, SFO, PDD), Lasso/compressive-sensing path-separation (Eq. 8), strongest-path reference sanitization, Doppler vector via STFT (Eq. 20) |
| | Paper 1 | p.147, brief mention only ("Interested readers can find a complete description... in [4]") | Cross-reference pointer to Paper 2 for full CFR/Doppler theory — **no algorithmic detail given in Paper 1** |
| | Presentation | Slide 4 "Doppler spectrograms from Wi-Fi packets" | Simplified explanation: STFT of channel estimate → spectrogram, treated as an "image" |
| **Model Architecture & Methodology (Neural Network)** | Paper 2 | p.6167–6168, Sec. 4 "Learning Architecture for HAR" (4.1, 4.2) + Fig. 4 | Simplified Inception module (3 branches, MAXPOOL+CONV), 128,535 parameters, 1×1 conv reduction (15→3 feature maps), dense layer (5 outputs), Dropout 20%, cross-entropy loss; Decision Fusion across Nant antennas (majority vote / summed activity vectors) |
| | Presentation | Slide 8 "Baseline model" (visual/diagram slide, minimal text) | Presents SHARP's architecture as the course "baseline model" to build upon |
| | Presentation | Slide 5 "Doppler spectrogram vs activity" | Reiterates CNN-based processing of spectrograms-as-images |
| **Evaluation Metrics & Experimental Results** | Paper 2 | p.6170–6173, Sec. 6 "Experimental Results" (6.1–6.7) + Tables 3–6 + Figs. 7–9 | Accuracy & F1-score per activity/set; ~95–100% mean accuracy; worst case (S7, new person+environment) = 95.99%; walking/running confusion; benchmark comparison vs DeepSense, EI, MatNet-eCSI (Fig. 9b–9c); antenna-count ablation (Fig. 9a); 8-activity extension results (Table 6) |
| | Paper 1 | p.150–151, "Examples of Use" + Fig. 3 | Pearson correlation coefficient analysis (CFR trace similarity across days) — illustrative dataset-usage example, not a model benchmark |
| | Presentation | Not explicitly covered | No results/metrics tables presented in the deck; students are pointed to the papers for quantitative results |
| **Project Roadmap & Future Deliverables / Research Directions** | Paper 1 | p.150–151, "Examples of Use" (7 use cases) + p.152 "Conclusions" | Time/environment/hardware robustness, domain adaptation, multi-path/interference impact, obstruction impact, Tx/monitor-location impact, time-frequency-spatial diversity; future work = fusing CFR with webcam ground truth for localization/tracking |
| | Paper 2 | p.6173, Sec. 7 "Concluding Remarks" | Future work: cross-hardware generalization, multi-person concurrent recognition |
| | Presentation | Slide 9 "Challenges and possible objectives" | Explicit student project directions: (1) cross-subject generalization via contrastive/self-supervised learning, (2) cross-environment/day robustness, (3) person identification |
| **Author / Institutional Info** | Paper 1 | p.152, "Biographies" | Meneghello, Dal Fabbro, Garlisi, Tinnirello, Rossi — Univ. of Padova & Univ. of Palermo/CNIT |
| | Paper 2 | p.6174–6175, "Biographies" | Same author team, extended bios; funding via MIUR "Departments of Excellence" and EU Horizon 2020 LOCUS (Grant 871249) |
| | Presentation | Slide 1 | Course staff: Riccardo Mazzieri, Daniele Mari, Jacopo Pegoraro (PhD/postdoc, Univ. of Padova) — **distinct from the paper authors**, confirming the deck is a third-party course adaptation, not written by the original researchers |
| **Data/Code Availability** | Paper 1 | p.146 Abstract; p.151 "Examples of Use" | IEEE DataPort DOI `10.21227/xbhv-f125`; reproducibility code at `codeocean.com/capsule/4811042/tree` |
| | Paper 2 | p.6169, Sec. 5; p.6174 Ref. [1] | GitHub repo `github.com/signetlabdei/SHARP` (code + dataset) |
| | Presentation | Slide 2 (Ref. [3]), Slide 6 | GitHub `github.com/francescamen/SHARP`; IEEE DataPort link (`ieee-dataport.org/documents/csi-dataset-wireless-human-sensing-80-mhz-wi-fi-channels`), noted as members-only |

---

## 3. Discrepancy & Gap Log

1. **Dataset size/diversity figures differ between documents.** Paper 1 (the full public dataset) reports **13 subjects, 7 environments, up to 7 activities**, with domain diversity across days/hardware. Paper 2 (SHARP's own validation subset) uses a **smaller collection: 3 subjects (P1–P3), 3 environments, 4–8 activities** across sets S1–S7. The Presentation's Slide 6 states **"10 different people, 7 environments"** — this matches neither paper exactly (it is closer to Paper 1's PI/PC subject count of 10, not the AR subject count of 4, and not SHARP's 3 subjects), so a reader could conflate the two datasets. This is a genuine inconsistency, not just a simplification.
2. **GitHub repository URL mismatch.** Paper 2 cites the code/dataset repository as `github.com/signetlabdei/SHARP`, while the Presentation (Slide 2, Ref. [3]) cites `github.com/francescamen/SHARP`. These may be a fork/rename of the same project, but the discrepancy is not explained in either source.
3. **No algorithmic detail in Paper 1.** Paper 1 explicitly defers all CFR/Doppler/phase-sanitization theory to Paper 2 ("Interested readers can find a complete description of the Wi-Fi channel in [4]"). A reader relying on Paper 1 alone cannot reconstruct the SHARP methodology; the Presentation likewise gives only a simplified, non-mathematical description (Slide 4), so full derivations (Eqs. 1–22) exist **only** in Paper 2.
4. **Presentation omits quantitative results entirely.** Neither accuracy/F1 numbers nor confusion matrices from Paper 2 (Tables 3–6, Figs. 7–9) are reproduced in the deck — Slide 8 ("Baseline model") is visual/diagrammatic with no metrics, so students must consult Paper 2 directly for benchmark numbers.
5. **Scope of "activities" is ambiguous across sources.** Paper 1 lists up to seven activity classes (W, R, J, L, S, C, G) in the full dataset; Paper 2's core SHARP evaluation uses **4 activities + empty room** (Table 3) and only extends to **8 activities** in a secondary experiment (Sec. 6.7, single subject only, Table 6). The Presentation's "7 different activities" (Slide 6) matches Paper 1's dataset-wide count but not SHARP's primary 5-class evaluation — this distinction is not flagged for readers.
6. **Presentation authorship vs. source papers.** The deck's authors (Mazzieri, Mari, Pegoraro) are not authors of Paper 1 or Paper 2 (Meneghello, Dal Fabbro, Garlisi, Tinnirello, Rossi). The Presentation is a third-party pedagogical adaptation for a Neural Networks and Deep Learning (NNDL) course, which should be kept in mind when treating it as an authoritative summary of the source papers.
7. **No explicit page/slide linkage for "Diversity is key" (Slide 7).** This slide title echoes the "domain diversity" theme central to both papers but contains no citation or specific figures of its own — it functions as a transitional/discussion slide rather than a sourced content slide.
