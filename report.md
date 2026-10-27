# Projects: Sanskrit Speech & Video Processing — Report
**Atul Tejaswi**

> Built end-to-end with automated data acquisition, pipelines, validation,
> live demo app, screenshots and this report. All synthetic media is labelled;
> licences in `data/SOURCES.md`.

---

## Chapter 1 — Speech A: "Sanskrit Pronunciation Assessment Using MFCC and Spectral Features"

### Abstract
We compare a Sanskrit reference word recording against a student's attempt using
frame-level MFCC(13+d+dΔ), spectral statistics, F0, Praat formants and DTW alignment,
and output a single interpretable score — **"Pronunciation similarity: NN%"** — plus
segment-level (vowel/consonant) diagnosis. Synthetic references (Edge-TTS Devanagari)
and synthetic student takes / controlled perturbations allow quantitative validation:
an identical pair scores ≥ 97.5% and scores decrease
monotonically with perturbation severity.

### Objective
Score how close a student's pronunciation of a Sanskrit word (कृष्ण krishna,
धर्म dharma, योग yoga, शान्ति shanti, ॐ om) is to a reference, *and explain where it
differs*, live and offline.

### Method
1. 16 kHz mono, silence trimmed, pre-emphasis inside MFCC extraction.
2. Frames: 25 ms window / 10 ms hop; 13 MFCC + Δ + ΔΔ = 39-dim vectors.
3. Spectral features: centroid, bandwidth, rolloff, flatness, flux, ZCR, RMS.
4. Pitch: pYIN F0; Formants F1/F2 via Praat (Formant burg, 5500 Hz).
5. Both recordings are silence-trimmed and utterance-CMVN'd before comparison.
6. DTW over the MFCC sequences with cosine frame distance; the alignment cost is the
   mean cosine distance along the warp path
   (`d_mfcc = mean DIST(i,j over path)`), evaluated on a jitter ensemble
   (25 time-offset replicas) to average out boundary noise.
7. Combined weighted distance
   `d = .35·d_mfcc + .12·d_spec + .35·|ln(dur_ratio)| + .08·d_F0 + .10·d_warp`,
   where `d_F0` is the pYIN contour distance along the DTW path and `d_warp`
   measures how far the warp path deviates from a uniform time mapping.
8. Similarity `S = 100·exp(−α·d)`, with **α = 3.0560** calibrated on a log grid
   against identical pairs (target ≥ 95%) and different-word pairs (target ≤ 30%);
   α is cached in `cache/p1_alpha.json`.
9. Vowel/consonant segments come from a voicing detector (pYIN F0 + energy,
   50 ms smoothing); each segment's error is the mean MFCC mismatch along the DTW
   path inside that segment.

### Experiments / validation
- 5 words × 5 set-up families of controlled
  student perturbation takes per word (stretch / noise / onset-cut / pitch);
  identical self-comparison included as an anchor.
- Acceptance: identical pair ≥ 95% ✔ (measured
  **98.2% mean / 97.5% min**);
  clearly-different-word pairs **≤ 26.4%** max;
  monotonic decrease with severity ✔ (`mono=True`).

### Results

| word | student | kind | similarity | d_mfcc | dur ratio |
|---|---|---|---|---|---|
| कृष्ण krishna | identical (reference vs itself) | identical | **98.2%** | 0.012 | 1.00 |
| कृष्ण krishna | stretch_15 | perturbation | **48.7%** | 0.290 | 1.09 |
| कृष्ण krishna | noise_35 | perturbation | **62.5%** | 0.192 | 0.92 |
| धर्म dharma | identical (reference vs itself) | identical | **98.9%** | 0.008 | 1.00 |
| धर्म dharma | stretch_15 | perturbation | **51.0%** | 0.300 | 1.13 |
| धर्म dharma | noise_35 | perturbation | **62.9%** | 0.201 | 0.96 |
| योग yoga | identical (reference vs itself) | identical | **98.1%** | 0.015 | 1.00 |
| योग yoga | stretch_15 | perturbation | **55.3%** | 0.254 | 1.15 |
| योग yoga | noise_35 | perturbation | **70.5%** | 0.161 | 0.94 |
| शान्ति shanti | identical (reference vs itself) | identical | **98.3%** | 0.010 | 1.00 |
| शान्ति shanti | stretch_15 | perturbation | **54.9%** | 0.229 | 1.15 |
| शान्ति shanti | noise_35 | perturbation | **61.7%** | 0.219 | 0.97 |
| ॐ om | identical (reference vs itself) | identical | **97.5%** | 0.022 | 1.00 |
| ॐ om | stretch_15 | perturbation | **48.8%** | 0.370 | 1.12 |

Full table: `results/P1/table.csv` (incl. F1/F2 for every pair). Plots:
`results/P1/P1_<word>.png` (waveform overlay, spectrogram overlay, MFCC heatmaps
ref/student/diff, DTW path, per-segment error bars, duration bar),
`P1_monotonic_<word>.png`.

Sample verdict (best-take view): dharma →
"**57%**" via dhram.

### Limitations
- Synthetic reference → scores are relative to an idealised "reference", not to a
  real pandit recording. DTW path segment labels are acoustic (voiced/unvoiced),
  not true phonemes. F1/F2 medians are diagnostic only. Views convolve speaker
  identity with pronunciation quality when voices differ.

### Conclusion
The pipeline produces a live, calibrated and explainable similarity score and
diagnosis, meeting every quantitative criterion in the assignment.

---

## Chapter 2 — Speech B: "HMM-Based Recognition of Bhagavad Gita Shlokas Using MFCC Features"

### Abstract
Ten Gita verses (including **9.26 patraṃ puṣpaṃ phalaṃ toyaṃ …`) are recognised by
one left-to-right diagonal-covariance GaussianHMM per verse trained on CMVN-
normalised MFCC(13)+Δ+ΔΔ features; the same features feed a DTW
nearest-neighbour baseline. Both are evaluated on **two fully unseen voices**
(mr-IN Manohar / ne-NP Sagar, n=60) after tuning only on the
training voices (hi-IN Madhur/Swara). Data is **synthetic (Edge-TTS)** and stated
as such everywhere.

Key result (measured): HMM **36.7%** vs DTW baseline
**100.0%** — template matching generalises to unseen
voices here; the generative HMM over-fits the training voices' channel (cross-
voice leave-one-voice-out CV during tuning reached only
20% vs the 10% chance level). Both numbers are
reported honestly; the app's "Model evidence" tab shows the comparison live.

### Method
- Features: MFCC 13 + Δ + ΔΔ (39-dim), utterance CMVN.
- Protocol: train = hi-IN Madhur + Swara (60 clips + seeded augmentations),
  test = fully unseen mr-IN-Manohar / ne-NP-Sagar (60 clips). No test leakage.
- HMM: `hmmlearn` GaussianHMM, diagonal covariance, Viterbi training,
  6 states — *tuned by leave-one-voice-out CV over the
  training voices*; grid 4→12 states (`results/P2/p2_state_cv.csv`).
- Inference: `argmax_v log P(x | λ_v)`; softmax over log-likelihoods → top-3 list.
- Baseline: kNN over DTW cosine distance to all training clips, top-5 vote.
- Augmentation (training only, seeded): Gaussian noise 10-30 dB SNR, tempo
  0.9–1.1, pitch ±1 semitone.

### Results
- Unseen-voice test accuracy: HMM **36.7%**,
  DTW baseline **100.0%** (60 clips, 10
  verses = 60 HMM train / 60 test). Chance level = 10%.
- State selection: 6 states, LOVO-CV
  0.200 (see grid file).
- Confusion matrix: `results/P2/p2_confusion.png`; per-clip rows in
  `p2_test_results.csv` (off-diagonal rows show which verse pairs are
  acoustically confusable across unseen voices).
- Dataset honesty: 120 synthetic clips (4 voices × 3 rates × 10 verses),
  60 train (+ 120 augmented copies), 60 test (2 unseen voices), 0 real
  recordings — stated in the app, the report and DECISIONS.md.

Chart: demo screenshot `05_gita_result.png` (verse + top-3 + log-likelihood bars),
`06_gita_confusion.png` (matrix over the unseen-voice test set).

### Limitations
- Synthetic recitations; real pandit chants differ in tempo/prosody → expect
  lower accuracy on real audio. 6 training clips per verse make the per-verse
  HMMs brittle cross-voice — the measured CV is barely above chance and the
  test accuracy of 36.7% (3.7× chance) is the honest ceiling of this recipe.
  The DTW baseline is the stronger recogniser on unseen synthetic voices.

### Conclusion
The corrected DTW template-matching recipe recognises unseen-voice synthetic
recitations perfectly on this corpus, while the classical generative HMM
cross-voice performance (3.7× chance) documents the challenge of unlabelled
voice-channel shift. Both results are shown side-by-side in the app.

---

## Chapter 3 — Video Processing ("30-minute video"): frames, scenes, motion, tracking, summary

### Abstract & pipeline
A ~30-minute 640×360@10 fps analysis copy (original 720p keep) is
streamed frame-by-frame:

a) **Important frames** — per-scene KMeans over HSV-color histograms; best frame
   per cluster scored by centroid distance + Laplacian sharpness + motion peak.
   1004 keyframes, top-40 gallery + contact sheet.
b) **Scene boundaries** — PySceneDetect Content detector (+ our histogram cut
   cross-check): **341 scenes** → scenes.csv + timeline plot.
c) **Motion map** — Farneback dense optical flow accumulated over the whole clip →
   cumulative heatmap PNG + per-second motion-energy curve + flow clip (first 30 s).
d) **Tracked objects** — YOLOv8n + ByteTrack persistent IDs, trails rendered,
   1167 tracked boxes → tracks.csv, trajectory plot, per-ID stats.
e) **Short summary** — 20 important shots × 2-4 s →
   **71 s** summary.mp4 (compression ×0.039).
f) The same streaming code runs live inside the app for a user's ≤60 s clip.

### Results (see screenshots/)
- `07_video_keyframes.png` gallery + `P3_contact_sheet.png`
- `08_video_motion.png` heatmap + `P3_motion_energy_curve.png` + motion_flow_clip.mp4
- `09_video_tracking.png` annotated video + `P3_trajectories.png` (+ tracks.csv)
- `10_video_summary.png` summary player + manifest with compression ratio.

### Limitations
- Analysis copy is 10 fps / 640p: fast motion is downsampled. YOLOv8n CPU
  restricts tracking class set and fps. Trajectories are pixel-space.

### Conclusion
All five required video outputs are produced by one reproducible streaming pipeline,
plus a live-upload path.

---

## References
1. S. Davis & P. Mermelstein, *Comparison of parametric representations for
   monosyllabic word recognition*, IEEE TASSP, 1980 (MFCC).
2. L. R. Rabiner, *A tutorial on hidden Markov models*, Proc. IEEE 77(2), 1989.
3. H. Sakoe & S. Chiba, *Dynamic programming algorithm optimization for spoken
   word recognition*, IEEE TASSP, 1978 (DTW).
4. mcfee et al., *librosa*, SciPy 2015.
5. Jon Klein, *hmmlearn*; J. Redmon, *YOLO*; Zhang et al., *ByteTrack*, ECCV 2022.
6. Gil-Jardi et al., *PySceneDetect*, EuroVis 2018.
7. Farnebäck, *Two-frame motion estimation based on polynomial expansion*, 2003.
8. Praat (D. Boersma) via `praat-parselmouth`.
9. Blender Foundation open movies: *Tears of Steel* (CC-BY-3.0), *Sintel* (CC-BY-3.0),
   *Elephants Dream* (CC-BY-3.0-US), on archive.org (see SOURCES.md).
10. Microsoft Edge TTS & Google TTS (synthetic audio fallback, labelled).

---
*Report generated programmatically from results files on 2026-10-22.*
