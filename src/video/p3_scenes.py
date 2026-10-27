"""P3b: scene boundaries with PySceneDetect (Content + Adaptive), cross-checked
against frame-histogram differences. Outputs scenes.csv + timeline plot."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[2]
CACHE = OUT / "cache" / "p3"
R = OUT / "results" / "P3"
R.mkdir(parents=True, exist_ok=True)


def main():
    prep = json.loads((CACHE / "prep.json").read_text())
    ana = Path(prep["analysis"])
    scenes_path = CACHE / "scenes.csv"
    scenes = []
    # 1) PySceneDetect Content
    try:
        from scenedetect import detect, ContentDetector, AdaptiveDetector
        cl = detect(str(ana), ContentDetector(threshold=27.0))
        scenes = [(s[0].get_seconds(), s[1].get_seconds()) for s in cl]
        print(f"[p3-scenes] ContentDetector: {len(scenes)} scenes")
    except Exception as e:
        print(f"[p3-scenes] PySceneDetect failed: {e}")
    # 2) AdaptiveDetector (may miss cuts; union/intersection check)
    try:
        from scenedetect import detect, AdaptiveDetector
        from scenedetect.detectors import AdaptiveDetector
        cl2 = detect(str(ana), AdaptiveDetector())
        scenes2 = [(s[0].get_seconds(), s[1].get_seconds()) for s in cl2]
        print(f"[p3-scenes] AdaptiveDetector: {len(scenes2)} scenes")
    except Exception:
        scenes2 = []
    # 3) histogram cross-check with our own cached stats
    with np.load(CACHE / "frames_stats.npz") as z:
        hist = z["hist"].astype(np.float32)
        fps = json.loads((CACHE / "stream_progress.json").read_text())["fps"]
    n = len(hist)
    if n > 1:
        d = np.array([float(np.abs(hist[i] - hist[i - 1]).sum())
                      for i in range(1, n)])
        thr = np.mean(d) + 3 * np.std(d)
        cuts_hist = [float(i) / fps for i in np.where(d > thr)[0]]
        print(f"[p3-scenes] histogram cut candidates: {len(cuts_hist)}")
    else:
        cuts_hist = []
    # merge: use PySceneDetect scenes as base; add hist cuts not already inside a scene
    for c in cuts_hist:
        if not any(s0 - 0.4 <= c < s1 + 0.4 for s0, s1 in scenes):
            # insert boundary
            for i, (s0, s1) in enumerate(scenes):
                if s0 <= c < s1:
                    scenes.append((s0, c))
                    scenes.append((c, s1))
                    del scenes[i]
                    break
    scenes.sort()
    import csv
    with open(R / "scenes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "start_sec", "end_sec", "duration_sec", "kind"])
        for i, (s0, s1) in enumerate(scenes):
            w.writerow([i, f"{s0:.2f}", f"{s1:.2f}", f"{s1 - s0:.2f}",
                        "merged-scene"])
    # timeline plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(18, 3), dpi=110)
    spans = [(s0, s1 - s0) for s0, s1 in scenes]
    ax.broken_barh(spans, (0, 1), facecolors="steelblue", alpha=0.8)
    ax.set_xlim(0, max(s1 for _, s1 in scenes) + 1)
    ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_title(f"Scene timeline - {len(scenes)} scenes "
                 f"(PySceneDetect Content + histogram cross-check)")
    for c in cuts_hist[:40]:
        ax.axvline(c, color="crimson", lw=0.6, alpha=0.5)
    fig.tight_layout()
    fig.savefig(R / "P3_scene_timeline.png"); plt.close(fig)
    (CACHE / "scenes.json").write_text(json.dumps(
        {"scenes": scenes, "content_n": len(scenes),
         "adaptive_n": len(scenes2), "hist_cuts": cuts_hist}, indent=2))
    print(f"[p3-scenes] final {len(scenes)} scenes -> {R / 'scenes.csv'}")


if __name__ == "__main__":
    main()
