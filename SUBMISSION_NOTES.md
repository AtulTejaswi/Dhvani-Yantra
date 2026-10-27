# SUBMISSION_NOTES.md — paste-ready short text for the upload form

Built by Atul Tejaswi — 3 projects in one package (Python 3.12, all offline, CPU
only):

**1) Sanskrit Pronunciation Assessment Using MFCC and Spectral Features**
Reference vs student takes of कृष्ण, धर्म, योग, शान्ति, ॐ (95 comparison pairs).
13 MFCC+Δ+ΔΔ (25ms/10ms), spectral centroid/bandwidth/rolloff/flatness/flux/ZCR,
pYIN F0, Praat F1/F2, DTW alignment → "Pronunciation similarity: NN%" with an
explainable verdict (vowel vs consonant errors, duration ratio, spectral shift).
Calibrated α; identical pairs score ≥95%, clearly-different words <30%, scores
decrease monotonically with perturbation severity (results/P1/summary.json).

**2) HMM-Based Recognition of Bhagavad Gita Shlokas Using MFCC Features**
10 verses incl. 9.26; CMVN-MFCC(13)ΔΔ; one left-to-right diagonal GaussianHMM per
verse; the state count is tuned with leave-one-voice-out CV, tested on voices the
model never saw (Marathi/Nepali neural voices reading the same Devanagari shlokas).
Reports accuracy, confusion matrix and a DTW nearest-neighbour baseline
(results/P2/p2_summary.json). Recitations are Edge-TTS synthetic — labelled.

**3) Video Processing of a ~30-minute video**
Concatenated CC-BY Blender open movies (Tears of Steel + Sintel + Elephants
Dream ≈30 min). Per-scene keyframes via KMeans+sharpness+motion; scene boundaries
via PySceneDetect(341 scenes) cross-checked with histogram cuts; Farneback motion
map + per-second energy; YOLOv8n+ByteTrack persistent-ID tracking with trails
(tracks.csv, annotated video, trajectory plot); a 60-90 s summary video
(compression ratio reported). Same streaming code runs live on a user clip
(≤60s) inside the app.

How to run: `./run_demo.sh` (or run_demo.bat on Windows) → Streamlit app on
http://localhost:8501 with tabs [Pronunciation] [Gita Verse] [Video]. Sample data
ships inside the package; demo responds in seconds.

Key screenshot files (screenshots/INDEX.md has captions):
01_pronunciation_home.png, 02_pronunciation_result.png, 03_pronunciation_plots.png,
04_gita_home.png, 05_gita_result.png, 06_gita_confusion.png, 07_video_keyframes.png,
08_video_motion.png, 09_video_tracking.png, 10_video_summary.png.
demo_walkthrough.mp4 is a recorded, captioned fallback if the live demo fails.

Honesty: every synthetic/fallback item is labelled in the app, report and
DECISIONS.md; all media licences in data/SOURCES.md (CC-BY + TTS).
