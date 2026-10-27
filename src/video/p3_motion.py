"""P3c: motion-map visuals from the cached streaming pass.
Outputs: cumulative heatmap PNG, per-second motion-energy curve, motion-map file
(embedded in timeline), reusing motion_flow_clip.mp4 from the streaming pass."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[2]
CACHE = OUT / "cache" / "p3"
R = OUT / "results" / "P3"
R.mkdir(parents=True, exist_ok=True)


def main():
    with np.load(CACHE / "frames_stats.npz") as z:
        motion = z["motion"]
        flow_big = z["flow_big"]
    fps = json.loads((CACHE / "stream_progress.json").read_text())["fps"]

    import cv2
    # 1) cumulative heatmap
    hm = flow_big / (flow_big.max() + 1e-9)
    hm8 = (hm * 255).astype(np.uint8)
    heat = cv2.applyColorMap(cv2.resize(hm8, (1280, 720)),
                             cv2.COLORMAP_JET)
    heat = cv2.addWeighted(
        cv2.resize(heat, (1280, 720)), 1.0,
        np.full((720, 1280, 3), 30, np.uint8), 0.0, 0)
    cv2.imwrite(str(R / "P3_motion_heatmap.png"), heat)

    # 2) per-second motion energy curve
    per_sec = np.add.reduceat(motion, np.arange(0, len(motion), int(fps))) \
        if len(motion) >= int(fps) else None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16, 4), dpi=110)
    sec_axis = np.arange(len(per_sec)) if per_sec is not None else []
    ax.plot(sec_axis, per_sec if per_sec is not None else [], color="darkorange")
    ax.set_xlabel("time (s)"); ax.set_ylabel("motion energy (sum |flow|)")
    ax.set_title("Motion energy per second - Farneback dense optical flow")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(R / "P3_motion_energy_curve.png"); plt.close(fig)
    np.save(CACHE.parent / "p3" / "motion_per_sec.npy",
            np.asarray(per_sec) if per_sec is not None else np.array([0.0]))
    print(f"[p3-motion] heatmap + curve for {int(len(motion) / fps / 60)} min")


if __name__ == "__main__":
    main()
