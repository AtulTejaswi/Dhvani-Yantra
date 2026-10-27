@echo off
REM run_demo.bat - Windows equivalent of run_demo.sh
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
if not defined PY if exist "..\.venv\Scripts\python.exe" set PY=..\.venv\Scripts\python.exe
if not defined PY (
  echo [demo] no venv found; using system python
  set PY=python
)
"%PY%" -c "import streamlit" >nul 2>nul || "%PY%" -m pip install -q -r requirements.txt
if not exist cache\p2\hmm_models.pkl "%PY%" -u src\speech_b\p2_train_eval.py
if not exist results\P1\summary.json "%PY%" -u src\speech_a\p1_eval.py
if not exist results\P3\P3_summary.mp4 (
  "%PY%" -u src\video\p3_prepare.py
  "%PY%" -u src\video\p3_run_all.py
)
set PYTHONPATH=%CD%\src;%CD%\src\speech_a;%CD%\src\speech_b;%CD%\src\video
"%PY%" -m streamlit run app.py --server.port 8501
