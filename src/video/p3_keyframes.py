"""P3a: per-shot keyframes via KMeans over HSV-histogram features (cached from
the streaming pass), scored by distance-to-centroid + sharpness + motion peak."""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[2]
CACHE = OUT / "cache" / "p3"
R = OUT / "results" / "P3"
R.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(OUT / "src"))
from lib_data import to_wav16  # noqa: F401  (keeps ffmpeg import path warm)

TOP_N = 40


def main():
    from sklearn.cluster import KMeans
    prep = json.loads((CACHE / "prep.json").read_text())
    ana = Path(prep["analysis"])
    fps = json.loads((CACHE / "stream_progress.json").read_text())["fps"]
    scenes = json.loads((CACHE / "scenes.json").read_text())["scenes"] or [(0, 0)]
    with np.load(CACHE / "frames_stats.npz") as z:
        hist = z["hist"].astype(np.float32)
        sharp = z["sharp"]
        motion = z["motion"]

    # extract one keyframe per scene KMeans cluster
    key_rows = []
    n = len(hist)
    for (s0, s1) in scenes:
        i0, i1 = int(s0 * fps), min(int(s1 * fps), n)
        if i1 - i0 < 4:
            f = (i0 + i1) // 2
            key_rows.append({"frame": f, "sec": f / fps, "score": 0.5,
                             "sharp": float(sharp[min(f, n - 1)]),
                             "motion": float(motion[min(f, n - 1)])})
            continue
        k = max(1, min(int(np.ceil((i1 - i0) / fps / 2.0)), 12))
        X = hist[i0:i1]
        if i1 - i0 > 400:
            Xs = X[:: max(1, (i1 - i0) // 400)]
            idx_map = np.arange(i0, i1)[:: max(1, (i1 - i0) // 400)]
        else:
            Xs, idx_map = X, np.arange(i0, i1)
        km = KMeans(n_clusters=min(k, len(Xs)), n_init=4,
                    random_state=17).fit(Xs)
        cents = km.cluster_centers_
        lab = km.labels_
        for ci in range(len(cents)):
            rows = np.where(lab == ci)[0]
            if len(rows) == 0:
                continue
            dd = np.linalg.norm(Xs[rows] - cents[ci], axis=1)
            dd = dd / (dd.max() + 1e-9)
            sh = sharp[idx_map[rows]] / (np.percentile(sharp, 75) + 1e-6)
            mo = motion[idx_map[rows]] / (np.percentile(motion, 90) + 1e-6)
            # best frame = close to the shot's visual theme, sharp, high motion
            obj = dd - 0.4 * np.log1p(sh) - 0.3 * np.log1p(mo)
            pick = rows[np.argmin(obj)]
            j = int(np.argmin(obj))
            f = int(idx_map[pick])
            score = float(1.0 / (0.4 + dd[j]) + 0.4 * np.log1p(sh[j])
                          + 0.3 * np.log1p(mo[j]))
            key_rows.append({"frame": f, "sec": f / fps, "score": score,
                             "sharp": float(sharp[f]), "motion": float(motion[f])})
    key_rows.sort(key=lambda r: r["sec"])
    # top-N "important frames" across the whole video by score
    top = sorted(key_rows, key=lambda r: -r["score"])[:TOP_N]
    top.sort(key=lambda r: r["sec"])

    import cv2
    kf_dir = CACHE / "keyframes"
    kf_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(ana))
    want = {r["frame"]: r for r in key_rows}
    for f, r in want.items():
        r["path"] = str(kf_dir / f"kf_{f:06d}.jpg")
    top_frames = {r["frame"] for r in top}
    for f, r in want.items():
        p = Path(r["path"])
        if p.exists():
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if not ok:
            continue
        if f in top_frames:
            cv2.imwrite(str(p), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        else:
            cv2.imwrite(str(p), cv2.resize(im, (320, 180)),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
    cap.release()

    import csv
    with open(R / "keyframes.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "sec", "score", "sharpness", "motion", "is_top",
                    "file"])
        for r in key_rows:
            w.writerow([r["frame"], f"{r['sec']:.2f}", f"{r['score']:.3f}",
                        f"{r['sharp']:.1f}", f"{r['motion']:.3f}",
                        r["frame"] in top_frames, r["path"]])
    (CACHE / "keyframes.json").write_text(json.dumps(
        {"key_rows": key_rows,
         "top": [{**t, "path": want[t["frame"]]["path"]} for t in top]},
        indent=2))

    # contact sheet of the top-N (8 x 5 grid, ffmpeg drawtext-free)
    import cv2
    ims = []
    for r in top:
        p = Path(r["path"])
        if p.exists():
            im = cv2.imread(str(p))
            if im is not None:
                ims.append(im)
    if ims:
        cols, rows_i = 8, (len(ims) + 7) // 8
        h = w = 160
        cell = 14
        sheet = np.full((rows_i * (h + cell), cols * w, 3), 245, np.uint8)
        for k, im in enumerate(ims):
            im = cv2.resize(im, (w, h))
            x, y = (k % cols) * w, (k // cols) * (h + cell)
            sheet[y:y + h, x:x + w] = im
            import time
            stamp = time.strftime("%H:%M:%S", time.gmtime(top[k]["sec"]))
            cv2.putText(sheet, stamp, (x + 3, y + h + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 30, 30), 1)
        cv2.imwrite(str(R / "P3_contact_sheet.png"), sheet)
    print(f"[p3-keyframes] {len(key_rows)} per-scene keyframes, "
          f"top-{len(ims)} -> contact sheet")


if __name__ == "__main__":
    main()
