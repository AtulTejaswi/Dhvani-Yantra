#!/usr/bin/env bash
# run_demo.sh — one-command live demo for the presentation.
# Creates/activates the venv, installs deps if needed, verifies caches exist,
# rebuilds them if missing, then launches the Streamlit app and prints the URL.
set -e
cd "$(dirname "$0")"

PY=python3
command -v python3 >/dev/null 2>&1 || PY=python
export PYTHONIOENCODING=utf-8

echo "== SVP demo launcher =="
# 1) venv
if [ ! -d ".venv" ] && [ ! -d "../.venv" ]; then
  echo "[demo] creating virtual environment..."
  "$PY" -m venv .venv
fi
# activate whichever venv we have
if [ -d ".venv/bin" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PY=.venv/bin/python
elif [ -d ".venv/Scripts" ]; then
  PY=.venv/Scripts/python
elif [ -f "../.venv/Scripts/python.exe" ]; then
  PY=../.venv/Scripts/python.exe
fi

# 2) deps (only if clearly missing; no-op when the app can already start)
if ! "$PY" -c "import streamlit" >/dev/null 2>&1; then
  echo "[demo] installing requirements (no-op if present)..."
  "$PY" -m pip install -q -r requirements.txt || echo "[demo] pip failed but continuing (deps may already exist)"
fi

# 3) data / model cache check — rebuild anything missing
if [ "$(ls -1 data/audio/ref_*.wav data/gita/verse_*.wav 2>/dev/null | wc -l)" -lt 60 ]; then
  echo "[demo] rebuilding speech datasets (TTS)..."
  "$PY" -u src/speech_a/p1_build_data.py || true
  "$PY" -u src/speech_b/p2_build_data.py || true
fi
if [ ! -f cache/p2/hmm_models.pkl ]; then
  echo "[demo] training Gita HMMs (few minutes)..."
  "$PY" -u src/speech_b/p2_train_eval.py
fi
if [ ! -f cache/p1_alpha.json ] || [ ! -f results/P1/summary.json ]; then
  echo "[demo] running P1 evaluation (few minutes)..."
  "$PY" -u src/speech_a/p1_eval.py
fi
if [ ! -f cache/p3/prep.json ] || [ ! -f cache/p3/frames_stats.npz ]; then
  echo "[demo] preparing video + streaming analysis (can take ~15 min)..."
  "$PY" -u src/video/p3_prepare.py
  "$PY" -u src/video/p3_run_all.py
fi

# 4) launch
echo "[demo] starting Streamlit app..."
echo "      (browser opens automatically; URL below)"
export PYTHONPATH="$PWD/src:$PWD/src/speech_a:$PWD/src/speech_b:$PWD/src/video"
"$PY" -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
