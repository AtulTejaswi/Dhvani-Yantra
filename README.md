# Sanskrit Speech & Video Processing — Assignment Package

**Student:** Atul Tejaswi

Three projects, one package, one live demo:

| # | Project | One-liner |
|---|---|---|
| P1 | **Sanskrit Pronunciation Assessment Using MFCC and Spectral Features** | reference vs student take of कृष्ण / धर्म / योग / शान्ति / ॐ → *"Pronunciation similarity: NN%"* + vowel/consonant/duration diagnosis |
| P2 | **HMM-Based Recognition of Bhagavad Gita Shlokas Using MFCC Features** | verse recording → "Verse 9.26" via per-verse Gaussian HMMs |
| P3 | **Video processing (30-min input)** | important frames • scene boundaries • motion map • YOLO+ByteTrack IDs • 60-90 s summary |

## Launch the live demo

```bash
./run_demo.sh        # Linux/macOS/WSL/Git-Bash   (Windows: run_demo.bat)
```

It reuses `.venv`, reinstalls only if needed, rebuilds any missing dataset/models,
then starts **Streamlit** at http://localhost:8501 (three tabs respond in seconds
on bundled samples).

## Repository tree

```
project_out/
├── app.py                  # unified live demo (Streamlit, 3 tabs)
├── run_demo.sh / .bat      # one-command launcher
├── requirements.txt        # pinned
├── README.md, report.md, report.docx, report/report.pdf, DEMO_SCRIPT.md, SUBMISSION_NOTES.md, DECISIONS.md
├── src/
│   ├── lib_audio.py        # features/SR/frame spec  (16 kHz, 25ms/10ms, 13 MFCC+ΔΔΔ)
│   ├── lib_data.py         # TTS + free-media fetchers, SOURCES.md logger, ffmpeg
│   ├── speech_a/           # P1: build_data, core (DTW/calibration), eval
│   ├── speech_b/           # P2: build_data, train_eval (HMM + DTW baseline)
│   └── video/              # P3: prepare, stream, scenes, keyframes, motion, track, summary, lite, run_all
├── tests/                  # pytest unit tests (feature spec, DTW, CMVN, …)
├── data/ (audio, gita, video) + SOURCES.md   # every file + licence (+synthetic flags)
├── results/P1, P2, P3      # CSV/JSON metrics, all plots, summary videos
├── screenshots/            # automated 1920×1080 captures + INDEX.md
├── demo_walkthrough.mp4    # recorded captioned fallback demo
└── cache/                  # alpha calibration, HMM pickles, video analysis copy (~), resumable
```

## Reproduce the numbers (optional)

```bash
# P1 (≈15 min): python src/speech_a/p1_eval.py
# P2 refresh data:  python src/speech_b/p2_build_data.py
# P2 train+eval (~15-25 min incl. DTW baseline):
#                python src/speech_b/p2_train_eval.py
# P3 (20-40 min):python src/video/p3_prepare.py && python src/video/p3_run_all.py
# screenshots + walkthrough:  python tools/capture.py
# tests:         python -m pytest tests -q
```

Everything synthetic (TTS) is labelled in the app, the report, `SOURCES.md` and
`DECISIONS.md`. All media is CC-BY (Blender open movies) or clearly-marked TTS.
