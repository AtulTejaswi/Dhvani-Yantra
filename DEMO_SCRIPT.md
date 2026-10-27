# DEMO_SCRIPT.md — 3-minute presentation flow + viva cheat sheet

## What to click and say (timed)

**0:00 — open.** `./run_demo.sh`, URL prints; browser opens three tabs.
*Say:* "Everything here runs offline on cached models; one command starts it."

**0:15 — Pronunciation tab.**
*Click:* word **कृष्ण krishna**, student take **1. krishna (tts-student)**, press
**Compare & score**.
*Say while spinner:* "Reference and student MFCC sequences are DTW-aligned; the
score is calibrated exponential decay of a weighted distance — MFCC, spectral,
duration and pitch; α was calibrated so identical audio scores ≈100% and a
different word drops under ~30%."
*Read the result:* "Pronunciation similarity: NN% — verdict explains v/consonant
errors and duration."

**1:00 — scroll to charts.** Waveform overlay, spectrogram overlay, MFCC heatmaps,
DTW path, per-segment error bars. Then the monotonicity plot: "scores decrease
monotonically as I inject larger perturbations — validated."

**1:30 — Gita tab.** Pick sample verse 9.26 (test voice; mention it's a voice
the model never heard), press **Identify verse**.
*Say:* "One left-to-right Gaussian HMM per verse; I score the clip against all
ten and softmax the log-likelihoods. States were chosen by leave-one-voice-out
cross-validation."
*Read:* "Verse 9.26, top-3 rank, accuracy chart; held-out-speaker accuracy NN%."

**2:10 — Video tab.** Walk through in order: keyframes gallery → scene timeline
(mention PySceneDetect + histogram cross-check) → cumulative motion heatmap and
per-second energy → tracking video with persistent IDs and trails → 60-90 s
summary player and compression ratio. Optionally: "process my uploaded clip".

**2:45 — close.** Point at synthetic-data labels; DECISIONS.md documents every
deviation.

## Q&A cheat sheet

- **Why MFCC?** Compact perceptually-warped cepstral representation; decorrelates
  spectral envelope info that carries phoneme identity; robust to gain/room.
- **What does the HMM do?** Models how the sequence of MFCC frames evolves through
  left-to-right states (emission = diagonal Gaussians); Baum-Welch trains the
  transition/emission parameters; at test the verse whose model maximises
  log-likelihood of the clip wins.
- **How does DTW work?** DP over all (i,j) frame pairs accumulating the min-cost
  alignment of two sequences regardless of speed difference; path backtraced to
  show where segments align; further used for per-segment errors in P1 and as a
  baseline classifier in P2.
- **Why calibration α?** Distance scales differ between feature groups; the
  exponential mapping needs one scale factor; supervised pairs define the
  operating point honestly.
- **Limitations?** Synthetic audio (labelled); DTW costs quadratic in length;
  voicing-based segment labels aren't true phonemes; YOLO at 5 fps CPU; scores on
  real recordings may differ; HMM diagonal covariance ignores correlations.
- **Data provenance?** All licences in data/SOURCES.md; CC-BY Blender movies for
  P3; TTS-recognition recitations labelled synthetic everywhere.
