"""P3: streaming analysis pass over the analysis copy.

One loop, ~10 fps 640x360, produces (checkpointed to cache/p3):
 - per-frame HSV histogram + Laplacian sharpness + motion energy
 - Farneback dense flow (every 2nd frame) -> cumulative motion map
 - first 30 s visualised as a flow-vector clip
Nothing is ever fully loaded into RAM; checkpoints allow resume.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import numpy as np
import cv2

OUT = Path(__file__).resolve().parents[2]
CACHE = OUT / "cache" / "p3"
STATS = CACHE / "frames_stats.npz"
META = CACHE / "frames_meta.jsonl"
PROGRESS = CACHE / "stream_progress.json"
FLOW_CLIP = CACHE / "motion_flow_clip.mp4"
FLOW_EVERY = 2          # Farneback on every 2nd frame (effective 5 fps)
FLOW_WH = (320, 180)    # low-res geometry for speed
SHEET_COLS = 5


def main():
    prep = json.loads((CACHE / "prep.json").read_text())
    ana = Path(prep["analysis"])
    cap = cv2.VideoCapture(str(ana))
    assert cap.isOpened(), f"cannot open {ana}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # resume support
    resume = PROGRESS.exists() and json.loads(PROGRESS.read_text()).get("complete", False)
    if resume:
        print("[p3-stream] already complete; reusing cache")
        return
    k0 = 0
    hist_l, sharp_l, motion_l = [], [], []
    flow_big = np.zeros((FLOW_WH[1], FLOW_WH[0]), np.float32)
    if PROGRESS.exists():
        ck = json.loads(PROGRESS.read_text())
        if STATS.exists() and ck.get("done", 0) > 0:
            k0 = ck["done"]
            with np.load(STATS) as z:
                hist_l = list(z["hist"])
                sharp_l = list(z["sharp"])
                motion_l = list(z["motion"])
                flow_big = z["flow_big"].astype(np.float32)
    if META.exists() and k0 == 0:
        META.unlink()
    if FLOW_CLIP.exists() and k0 == 0:
        FLOW_CLIP.unlink()
    cap.set(cv2.CAP_PROP_POS_FRAMES, k0)
    print(f"[p3-stream] {int(cap.get(3))}x{int(cap.get(4))}@{fps:.1f}; "
          f"{n_total - k0} frames to go (resume at {k0})")

    mfw = None
    if k0 > 0:
        pass  # flow clip is only produced on a full pass from zero
    mfw = cv2.VideoWriter(str(FLOW_CLIP), cv2.VideoWriter_fourcc(*"mp4v"),
                          12, (640, 360)) if k0 == 0 else None

    t0, tf, done = time.time(), k0, k0
    prev = None
    fw, fh = FLOW_WH
    step = fh // 18

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp_l.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                         [0, 180, 0, 256, 0, 256]).flatten()
        hist_l.append((h / (h.sum() + 1e-9)).astype(np.float16))
        mv = 0.0
        if tf % FLOW_EVERY == 0:
            small = cv2.resize(gray, FLOW_WH).astype(np.float32)
            if prev is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev, small, None, 0.4, 3, 15, 3, 5, 1.1, 0)
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                mv = float(mag.mean())
                flow_big += mag
                if mfw is not None and tf < int(30 * fps):
                    vis = cv2.cvtColor((small / 255 * 120).astype(np.uint8),
                                       cv2.COLOR_GRAY2BGR)
                    ys, xs = np.mgrid[step:fh:step, step:fw:step]
                    af = flow[ys, xs]
                    for y0, x0 in ZIP_STEP(ys, xs, step):  # thin out arrow grid
                        d = af[(y0 - step) // step, (x0 - step) // step]
                        cv2.line(vis, (x0, y0),
                                 (int(x0 + d[0] * 2.5), int(y0 + d[1] * 2.5)),
                                 (0, 255, 0), 1, cv2.LINE_AA)
                    mfw.write(cv2.resize(vis, (640, 360)))
            prev = small
        motion_l.append(mv * FLOW_EVERY)
        done = tf + 1
        tf = done
        if done % 500 == 0:
            np.savez_compressed(STATS, hist=np.asarray(hist_l, np.float16),
                                sharp=np.asarray(sharp_l, np.float32),
                                motion=np.asarray(motion_l, np.float32),
                                flow_big=flow_big)
            PROGRESS.write_text(json.dumps({"done": done, "fps": fps,
                                            "n_total": n_total}))
            with META.open("a", encoding="utf-8") as f:
                for i in range(done - 500, done):
                    f.write(json.dumps({"frame": i, "sec": i / fps,
                                        "sharp": sharp_l[i],
                                        "motion": motion_l[i]}) + "\n")
            sps = done / max(time.time() - t0, 1)
            print(f"[p3-stream] {done}/{n_total} ({sps:.0f} f/s)", flush=True)
    cap.release()
    if mfw is not None:
        mfw.release()

    np.savez_compressed(STATS, hist=np.asarray(hist_l, np.float16),
                        sharp=np.asarray(sharp_l, np.float32),
                        motion=np.asarray(motion_l, np.float32),
                        flow_big=flow_big)
    META.write_text("", encoding="utf-8")  # rewind
    with META.open("a", encoding="utf-8") as f:
        for i, (s, m) in enumerate(zip(sharp_l, motion_l)):
            f.write(json.dumps({"frame": i, "sec": i / fps,
                                "sharp": s, "motion": m}) + "\n")
    PROGRESS.write_text(json.dumps({"done": done, "fps": fps,
                                    "n_total": n_total, "complete": True}))
    print(f"[p3-stream] DONE {done} frames in {time.time() - t0:.0f}s")


def ZIP_STEP(ys, xs, step):
    for y0, x0 in zip(ys.ravel(), xs.ravel()):
        if (y0 / step) % 2 == 1 and (x0 / step) % 2 == 0:
            yield y0, x0


if __name__ == "__main__":
    main()
