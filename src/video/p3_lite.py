"""p3_lite: the SAME streaming pipeline, applied live to a short user clip
(<= 60 s at 640p/10 fps) in a few seconds. Produces the user_*.png outputs."""
from __future__ import annotations

import subprocess
from pathlib import Path
import numpy as np
import cv2

CACHE_P3 = Path(__file__).resolve().parents[1] / "cache" / "p3"


def lite_analyse(clip: Path, out_dir: Path, fps_hint: float = 10.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(clip))
    assert cap.isOpened(), "cannot open clip"
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint
    hist, sharp, motion = [], [], []
    prev = None
    fw, fh = 320, 180
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                         [0, 180, 0, 256, 0, 256]).flatten()
        hist.append((h / (h.sum() + 1e-9)).astype(np.float32))
        mv = 0.0
        if frame_i % 2 == 0:
            small = cv2.resize(gray, (fw, fh)).astype(np.float32)
            if prev is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev, small, None, 0.4, 3, 15, 3, 5, 1.1, 0)
                mv = float(np.sqrt((flow ** 2).sum(axis=2)).mean()) * 2
            prev = small
        motion.append(mv)
        frame_i += 1
    cap.release()
    hist = np.asarray(hist); sharp = np.asarray(sharp); motion = np.asarray(motion)

    # scenes: histogram cuts (fast ContentDetector equivalent)
    d = np.array([np.abs(hist[i] - hist[i - 1]).sum() for i in range(1, len(hist))])
    thr = d.mean() + 2.5 * d.std()
    cuts = [0] + list(np.where(d > thr)[0] + 1) + [len(hist)]
    cuts = sorted(set(int(c) for c in cuts))
    scenes = [(cuts[i] / fps, cuts[i + 1] / fps) for i in range(len(cuts) - 1)]

    # keyframes: best-scoring frame per scene
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gs
    nS = min(len(scenes), 8)
    fig = plt.figure(figsize=(18, 7), dpi=110)
    G = gs.GridSpec(2, 4, figure=fig)
    cap = cv2.VideoCapture(str(clip))
    for si, (s0, s1) in enumerate(scenes[:nS]):
        i0, i1 = int(s0 * fps), int(s1 * fps)
        if i1 - i0 < 2:
            i1 = i0 + 2
        seg = motion[i0:i1] + 2 * sharp[i0:i1] / (np.nanpercentile(sharp[i0:i1], 75) + 1e-6)
        f = int(i0 + np.argmax(seg))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if not ok:
            continue
        # annotate important frame with a tracked face/person hint: contours
        ax = fig.add_subplot(G[si // 4 if nS > 4 else 0,
                            si % 4 if nS > 4 else si])
        gsm = cv2.cvtColor(cv2.resize(im, (320, 180)), cv2.COLOR_BGR2RGB)
        ax.imshow(gsm)
        ax.set_title(f"scene {si}: {s0:.1f}-{s1:.1f}s  frame {f}", fontsize=8)
        ax.axis("off")
    cap.release()
    fig.suptitle("Important frames (per-scene KMeans-free keyframes)" if nS else "frames")
    fig.tight_layout()
    fig.savefig(out_dir / "user_keyframes.png"); plt.close(fig)

    # scene timeline
    fig, ax = plt.subplots(figsize=(16, 2), dpi=110)
    for si, (s0, s1) in enumerate(scenes):
        ax.axvspan(s0, s1, color="skyblue" if si % 2 else "steelblue")
    ax.set_xlim(0, max(s1 for _, s1 in scenes))
    ax.set_yticks([]); ax.set_xlabel("time (s)")
    ax.set_title(f"Scene boundaries: {len(scenes)} scenes (histogram detector)")
    fig.savefig(out_dir / "user_scene_timeline.png"); plt.close(fig)

    # motion heatmap from accumulated flow
    cap = cv2.VideoCapture(str(clip))
    acc = np.zeros((fh, fw), np.float32)
    prev = None; i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i / fps > 60:
            break
        if i % 2 == 0:
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                               (fw, fh)).astype(np.float32)
            if prev is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev, small, None, 0.4, 3, 15, 3, 5, 1.1, 0)
                acc += np.sqrt((flow ** 2).sum(axis=2))
            prev = small
        i += 1
    cap.release()
    if acc.max() > 0:
        hm = (acc / acc.max() * 255).astype(np.uint8)
        heat = cv2.applyColorMap(cv2.resize(hm, (640, 360)), cv2.COLORMAP_JET)
        cv2.putText(heat, "motion map (accumulated |Farneback flow|)",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(out_dir / "user_motion.png"), heat)
    return scenes
