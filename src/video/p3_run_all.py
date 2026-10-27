"""P3 orchestrator: run every P3 stage in order with checkpointing."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable


import os


def run(script: str):
    print(f"\n[p3-run-all] === {script} ===", flush=True)
    t0 = time.time()
    env = os.environ | {"PYTHONIOENCODING": "utf-8"}
    r = subprocess.run([PY, "-u", str(HERE / script)], env=env, check=False)
    if r.returncode != 0:
        raise SystemExit(f"{script} failed rc={r.returncode}")
    print(f"[p3-run-all] {script} OK in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    for s in ("p3_stream.py", "p3_scenes.py", "p3_keyframes.py",
              "p3_motion.py", "p3_track.py", "p3_summary.py"):
        run(s)
    print("[p3-run-all] ALL STAGES COMPLETE")
