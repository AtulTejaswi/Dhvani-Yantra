"""P3d: detection + multi-object tracking with YOLOv8 + ByteTrack (ultralytics
model.track), producing an annotated video, tracks.csv, trajectory plot and
per-ID stats. Fallback (HOG person + centroid tracker) if ultralytics fails."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import numpy as np
import cv2

OUT = Path(__file__).resolve().parents[2]
CACHE = OUT / "cache" / "p3"
R = OUT / "results" / "P3"
R.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(OUT / "src"))

PROC_HEAD_SECONDS = 240.0  # track 4 minutes head (people-dense part of ToS)
SUBSAMPLE = 2              # feed every 2nd frame (effective 5 fps tracking)


def run_yolo(ana: Path) -> None:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")   # downloads from GitHub releases (allowed)
    cap = cv2.VideoCapture(str(ana))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    W, H = int(cap.get(3)), int(cap.get(4))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    limit = min(int(PROC_HEAD_SECONDS * fps), n_total)

    ann_path = CACHE / "tracked_annotated.mp4"
    vw = cv2.VideoWriter(str(ann_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps / SUBSAMPLE, (W, H))
    rows, trails = [], {}
    t0, frame_i = time.time(), 0
    results_iter = model.track(source=str(ana), stream=True, tracker="bytetrack.yaml",
                               conf=0.35, iou=0.5, persist=True,
                               classes=[0, 1, 2, 3, 7],   # person/bike/car/motor/bus
                               vid_stride=SUBSAMPLE)
    for res in results_iter:
        frame = res.orig_img
        sec = frame_i * SUBSAMPLE / fps
        if res.boxes is not None:
            for b in res.boxes:
                tid = int(b.id) if b.id is not None else -1
                cls = int(b.cls) if b.cls is not None else -1
                x0, y0, x1, y1 = [float(v) for v in b.xyxy[0]]
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                rows.append({"id": tid, "class": model.names[cls],
                             "frame": frame_i, "sec": sec,
                             "bbox_x1": x0, "bbox_y1": y0,
                             "bbox_x2": x1, "bbox_y2": y1,
                             "conf": float(b.conf)})
                trails.setdefault(tid, []).append((cx, cy))
                cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)),
                              (0, 255, 0), 2)
                cv2.putText(frame, f"id{tid} {model.names[cls]} {float(b.conf):.2f}",
                            (int(x0), max(15, int(y0 - 6))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        # trail rendering
        for tid, tr in trails.items():
            pts = [(int(x), int(y)) for x, y in tr[-60:]]
            for j in range(1, len(pts)):
                cv2.line(frame, pts[j - 1], pts[j], (255, 0, 255), 1)
        vw.write(frame)
        frame_i += 1
        if frame_i % 100 == 0:
            print(f"[p3-track] processed block {frame_i} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if frame_i * SUBSAMPLE >= limit:
            break
    vw.release()
    import csv
    with open(R / "tracks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (CACHE / "tracks_done.json").write_text(json.dumps(
        {"rows": len(rows), "frames": frame_i, "fps": fps,
         "ids": sorted(trails), "source_of_truth": "yolov8n+bytetrack"}))
    # trajectory plot of first <=8 IDs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 7), dpi=110)
    for tid, tr in list(sorted(trails.items()))[:8]:
        if len(tr) < 8:
            continue
        xs, ys = zip(*tr)
        ax.plot(xs, ys, alpha=0.85, lw=1.6, label=f"id {tid}")
        ax.annotate(str(tid), (xs[-1], ys[-1]), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    ax.set_title("Persistent-ID trajectories (YOLOv8n + ByteTrack)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(R / "P3_trajectories.png"); plt.close(fig)
    print(f"[p3-track] {len(rows)} boxes, {len(trails)} IDs -> tracks.csv")


def fallback_hog(ana: Path) -> None:
    print("[p3-track] FALLBACK: OpenCV HOG person detector + centroid tracker")
    cap = cv2.VideoCapture(str(ana))
    hog = cv2.HOGDescriptor(); hog.setSVMDetector(
        cv2.HOGDescriptor_getDefaultPeopleDetector())
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    W, H = int(cap.get(3)), int(cap.get(4))
    ann = CACHE / "tracked_annotated.mp4"
    vw = cv2.VideoWriter(str(ann), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps / SUBSAMPLE, (W, H))
    rows, next_id, centres = [], 0, {}
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        sec = frame_index / fps
        frame_index += 1
        if sec > PROC_HEAD_SECONDS:
            break
        if (frame_index - 1) % SUBSAMPLE:
            continue
        small = cv2.resize(frame, (640, 480))
        rects, wts = hog.detectMultiScale(small, winStride=(8, 8),
                                          padding=(4, 4), scale=1.05)
        newcent = {}
        for (x0, y0, w0, h0), conf in zip(rects, wts):
            sx, sy = W / 640, H / 480
            x0, y0 = int(x0 * sx), int(y0 * sy)
            w0, h0 = int(w0 * sx), int(h0 * sy)
            cx, cy = x0 + w0 // 2, y0 + h0 // 2
            tid = min(centres, key=lambda t: np.hypot(centres[t][0] - cx,
                                                      centres[t][1] - cy),
                      default=None)
            if tid is None or np.hypot(centres[tid][0] - cx,
                                       centres[tid][1] - cy) > 100:
                tid = next_id; next_id += 1
            newcent[tid] = (cx, cy)
            rows.append({"id": tid, "class": "person", "frame": frame_index - 1,
                         "sec": sec, "bbox_x1": x0, "bbox_y1": y0,
                         "bbox_x2": x0 + w0, "bbox_y2": y0 + h0,
                         "conf": round(float(conf), 3)})
            cv2.rectangle(frame, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 0), 2)
            cv2.putText(frame, f"id{tid}", (x0, max(15, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        for tid in list(centres):
            if tid not in newcent:
                centres.pop(tid)
        centres.update(newcent)
        vw.write(frame)
    vw.release(); cap.release()
    import csv
    if rows:
        with open(R / "tracks.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    (CACHE / "tracks_done.json").write_text(json.dumps(
        {"rows": len(rows), "source_of_truth": "opencv-hog-fallback"}))


def main():
    prep = json.loads((CACHE / "prep.json").read_text())
    ana = Path(prep["analysis"])
    try:
        run_yolo(ana)
        ok = (R / "tracks.csv").exists() and (R / "tracks.csv").stat().st_size > 200
        if not ok:
            raise RuntimeError("empty tracks.csv")
    except Exception as e:
        print(f"[p3-track] YOLO path failed: {e}")
        try:
            fallback_hog(ana)
        except Exception as ee:
            print(f"[p3-track] fallback also failed: {ee}")


if __name__ == "__main__":
    main()
