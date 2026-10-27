# data/SOURCES.md — every data source, synthetic flag and licence

Everything shipped in this project is either (a) a Creative-Commons Blender
Foundation open movie, (b) a **synthetic** machine-generated speech clip, or
(c) a derivative of (a)/(b). There is **no third-party copyrighted media**.
All synthetic assets are flagged `[SYNTHETIC …]` both here and in the app UI.

---

## P1 — Sanskrit word-pronunciation data  (`data/audio/`)

All P1 wavs are **synthetic speech** generated locally with Microsoft
`edge-tts` (Neural TTS, 16 kHz mono after ffmpeg conversion). Original
Wikimedia Commons pronunciation
probes returned no usable Sanskrit files, so the whole corpus is
[documented fallback](../DECISIONS.md) and carries an explicit
`SYNTHETIC` flag. Licence: edge-tts output (free Microsoft Neural TTS);

* **References** `ref_{word}.wav` + `ref_{word}_alt.wav` — Devanagari words
  कृष्ण (krishna), धर्म (dharma), योग (yoga), शान्ति (shanti), ॐ (om),
  read by hi-IN-MadhurNeural (ref) and hi-IN-SwaraNeural (_alt) → `[SYNTHETIC REFERENCE]`
* **Student takes** `stud_{word}.wav`, `stud_{word}_{i}_{variant}.wav` — en-IN-PrabhatNeural reading *romanised variants* (krishna/kishna/krosna, dhram/dharm, yug/yog, aum/om, shant/shanti) → `[SYNTHETIC STUDENT TAKE]`
* **Perturbations** `pert_{word}_{family}_{param}.wav` — programmatic distortions of the reference itself (time-stretch 15/30/50/80 % slower via a dependency-free SOLA implementation, additive noise at 35/30/22/10 dB SNR, onset-region deletion 11/13/16/20 %, pitch 1/2/3 semitones). Each derives from the synthetic reference, flagged `[SYNTHETIC PERTURBATION]`.

## P2 — Bhagavad Gita verse recitations (`data/gita/`)

All 120 wavs are **Edge-TTS recitations** of standard Devanagari shlokas;
the text of the Bhagavad Gita itself is an ancient public-domain scripture
(transcription via Wikisource). Every wav is flagged `[SYNTHETIC RECITATION]`;
licence: edge-tts output (Microsoft Neural TTS).

| verse | voices | rates | split |
|---|---|---|---|
| 1.1, 2.14, 2.47, 2.62, 4.7, 4.8, 9.22, 9.26, 10.20, 18.66 | hi-IN-MadhurNeural, hi-IN-SwaraNeural | −10 %, +0 %, +12 % | train (seen voices) |
| same 10 verses | mr-IN-ManoharNeural, ne-NP-SagarNeural | −10 %, +0 %, +12 % | test (**fully held-out voices**) |

60 train + 60 test clips = 120 files (`verse_{verse}_{Voicetag}_{rate}.wav`).

## P3 — video corpus (`data/video/`)

| file | source | licence | role |
|---|---|---|---|
| `raw_tears_of_steel_720p.mp4` | Blender Foundation "Tears of Steel" — https://archive.org/details/Tears-of-Steel | CC-BY-3.0 | live-action short with **real human actors** (12:14) |
| `raw_sintel_512kb.mp4` | Blender Foundation "Sintel" — https://archive.org/details/Sintel | CC-BY-3.0 | animated feature exerpt (fills length) |
| `raw_elephants_dream_512kb.mp4` | Blender Foundation "Elephants Dream" — https://archive.org/details/ElephantsDream | CC-BY-3.0-US | animated short (fills length) |
| `movie_30min.mp4` | ffmpeg concat + trim of the 3 above to exactly 1800 s, 720p, no audio | derivative of CC-BY works (attribution above) | the "input video" the whole P3 pipeline processes |

`movie_30min.mp4` starts with Tears of Steel so the YOLO tracking head
(first 4 min) sees real people. Each source is re-encoded 720 p/24 fps and
capped at 700 s before concat.

## Models / libraries bundled or downloaded at run time

| item | source | licence |
|---|---|---|
| YOLOv8n weights | auto-downloaded via `ultralytics` (github release) | AGPL-3.0 |
| ffmpeg binary | `imageio-ffmpeg` wheel | GPL (bundled binary of ffmpeg) |
| librosa / hmmlearn / scikit-learn / matplotlib / streamlit … | PyPI (see `requirements.txt`) | BSD/AGPL/PSF etc., standard OSS |
| DejaVu Sans (plot font) | matplotlib bundled | Bitstream Vera/DejaVu licence |

## Fallback/failed probes

* Wikimedia Commons Sanskrit pronunciation uploads — probed, **failed /
  unusable** → P1 corpus chose the documented edge-tts fallback
  (see `DECISIONS.md`, fallback #2).
* archive.org Gita-archive chanting probes — probed, few-permission issues →
  P2 corpus is fully synthetic instead.
