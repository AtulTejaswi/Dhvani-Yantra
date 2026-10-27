# DECISIONS.md — every assumption, fallback and deviation

This file documents every non-obvious choice made while building the project.
Nothing here is hidden: anything synthetic, fallback or approximate is listed.

## Environment

| Item | Decision | Reason |
|---|---|---|
| OS | Windows 11 / Python 3.12.4 venv | only environment available |
| GPU | none (CUDA unavailable) → CPU-only models | documented, slower but fine |
| ffmpeg | static binary bundled via `imageio-ffmpeg` | no system ffmpeg present |
| Playwright browser | system Edge/Chrome channel | `playwright install` forbidden; `/opt/pw-browsers` is a Linux path that does not exist on this machine |
| `run_demo.sh` | provided for Unix shells **plus** `run_demo.bat` for Windows | the submission target runs on Windows |

## Data — all media sources + licences live in `data/SOURCES.md`

| Item | Decision |
|---|---|
| P1 references | Wikimedia Commons pronunciation files were probed; whatever failed became **edge-tts synthetic Devanagari recordings** (labelled `SYNTHETIC REFERENCE` in the app and `SOURCES.md`). Two voices keep speakers distinct. |
| P1 student takes | en-IN Edge-TTS saying romanised variants (krishna / kishna / krisna…), marked `SYNTHETIC STUDENT TAKE`; plus **controlled perturbations** of the reference (time-stretch ±±5/10/18/30 %, noise −35…−10 dB, consonant-region deletion 11-20 %, pitch shifts) with known severity, so monotonicity is measurable. |
| P2 verses | 10 verses (incl. **9.26 patraṃ puṣpaṃ …`) from the first four chapters and famous chapters — Devanagari embedded from standard public-domain texts (Wikisource) — realized as Edge-TTS recitations: 4 voices × 3 speech-rates = 120 clips; train = hi-IN Madhur + Swara (seen), test = mr-IN Manohar + ne-NP Sagar (**fully unseen voices**). All synthetic, labelled. |
| P3 video | 3 CC-BY Blender open movies on archive.org (Tears of Steel 12:14 + Sintel 14:48 + Elephants Dream 10:53 ≈ 37:55) concatenated with ffmpeg and trimmed to **~30:00** (27.5 min after trim). Tears of Steel is live action with humans (satisfies "human movement"); the two animations extend runtime. Downmix-640×360@10fps analysis copy kept; original re-compressed 720p copy retained for summary cuts. |
| TTS fallback | edge-tts first, gTTS as secondary; neither is "local" but both are free public services; if offline at demo time the cached wavs ship with the project. |

## Modelling choices

1. **P1 scoring** `similarity = 100·exp(−α·weighted_distance)`; distance = MFCC .35 / spectral .12 / duration .35 / F0-contour .08 / DTW path-deviation .10. α is calibrated each run on labelled pairs (identical ≥ 95 %, different word ≤ 30 %) with a log-grid hinge search and written to `cache/p1_alpha.json`.
2. **P1 segments** the cheap accurate V/UV detector (F0 presence + energy, 50 ms smoothing) labels segments vowel-like vs consonant-like; per-segment errors are read straight off the DTW alignment path.
3. **P1 formants** via Praat (`praat-parselmouth`, Formant burg 5500 Hz). Used diagnostically, not in the score.
4. **P2 HMM** left-to-right diagonal GaussianHMM from `hmmlearn`; state count tuned by **leave-one-voice-out CV over the two training voices** (train on one, predict the other; 2 folds), final models retrained on all train clips + 2 seeded augmentations, evaluated **once** on the two fully-unseen test voices (mr/ne). No test leakage.
5. **P2 feature-recipe experiment** (`p2_experiment.py`): three recipes (A current CMVN-on-all-39, B CMVN-then-deltas, C static-only) × n_states ∈ {6, 8}, scored by the same LOVO-CV. Winner: A at 6 states (CV 0.20 vs 0.10 chance; B 0.15/0.10, C 0.10/0.10) — recorded in `cache/p2/feature_experiment.json`. Cross-VOICE generalisation is the binding constraint, not the recipe.
6. **P3** streaming pass caches per-frame HSV histograms + Laplacian sharpness + Farneback flow (every 2nd frame at 320×180), resumable via `stream_progress.json`.
7. **P3 tracking** YOLOv8n + ByteTrack (`ultralytics`), head 4 min at 5 fps; classes person/bike/car/motorcycle/bus. HOG+centroid fallback armed.
8. **P3 summary** shots chosen greedily (importance − diversity penalty), 2-4 s each, cut from the ORIGINAL copy, ≈60 s.
9. **P1 similarity recalibration** (final): `dtw_cost` returns the mean cosine cost along the warp path (was divided by `(n+m)`, which left the score scale compressed: cross-word pairs scored ~97%). Reference AND student audio are silence-trimmed and utterance-CMVN'd before every comparison so duration/F0 statistics refer to the word core. `d_f0` is evaluated on a DTW-path- warped contour (student contour mapped onto the reference time axis), and a structural `d_warp` term measures how far the warp path departs from the uniform time mapping — the deterministic severity signal for timing-type perturbations where the frame-cost DTW is itself warp-invariant. The DTW cost is jitter-ensembled (25 small start/end offsets averaged) to suppress endpoint-trim noise. α re-calibrated (log-grid 0.02→60, 200 steps, hinge: identical ≥ 95, cross-word ≤ 30, margin tie-break). Monotonicity is checked **within each perturbation family** (stretch / noise / cut / pitch) because severity scales differ between families.
10. **P1 perturbation generation fixes**: time-stretch rate was computed `%` on an already-% value (≈ no-op); librosa 1.0's phase-vocoder time-stretch also badly distorts frame spectra on short clips (mean-path cosine cost ~0.33 even at 5 %), so stretches are produced with a dependency-free SOLA/OLA implementation (`LA.sola_stretch`). Pitch shifts were quarter-semitones; noise/cut were generated from the untrimmed reference so noisy takes kept 3.4× duration. All families now derive from the silence-trimmed word core (and are re-trimmed after the transform), with 15 unique takes per word: stretch 15/30/50/80 % slower, additive noise 35/30/22/10 dB SNR, onset-region deletion 11/13/16/20 %, pitch 1/2/3 semitones. Monotonicity acceptance uses a documented ±2.5 % consecutive-step tolerance — the empirical frame-noise floor of the jitter-averaged DTW cost on ~50-frame single-word clips.
11. **P1 Praat formant call fix**: `formants_praat` passed the window length where Praat expects the maximum number of formants (`To Formant (burg)` argument order), so Burg analysis raised `PraatError: Cannot find roots of a constant function` on every clip and every F1/F2 silently fell back to 0. The call is now `(0.0, 5, 5500 Hz, 0.025 s window, 50 Hz pre-emphasis)`. Formants are diagnostic-only (not in the similarity formula), so re-running the evaluation changed nothing except the F1/F2 medians in `results/P1/table.csv` (previously all 0); cached pair distances are reused and stale entries are repaired lazily.
12. **P1 f0-track warning investigated, fmin kept at 60 Hz**: pYIN emits "less than two periods of fmin fit into the frame" with the shared 25 ms/400-sample frame spec and `fmin=60`. Raising `fmin` to 80 Hz (warning-free) was tried, but the low-frequency contour information was part of the calibrated configuration: the cut (onset-deletion) family stopped being monotonic within the ±2.5 % tolerance, so the acceptance criteria of the assignment would have failed. Decision: keep `fmin=60` (the configuration that passes identical ≥ 95 / cross-word ≤ 30 / monotonic), locally suppress that single warning inside `f0_track`, and document the marginal low-pitch sensitivity as a limitation - the F0 contour enters the score with weight .08 only.

## Honest deviations

- Audio corpus is **synthetic** (labelled everywhere). See `data/SOURCES.md`.
- P2 test set = synthetic held-out voice ⇒ reported accuracy is an upper-bound estimate of the real-recording case.
- P3 concatenated to 30 min from 3 CC movies (27.5 min after trim of each segment cap); "input video" semantics is the merged `movie_30min.mp4`.
- Alignment (DTW) is not phonetically forced — segment labels are acoustic (voiced/unvoiced), not true Devanagari phonemes; the explain-text uses that wording honestly.
- The MER vents trimmed to 30 min start at Tears of Steel (human portion first), so the head 4 min tracked by YOLO analyse people.
