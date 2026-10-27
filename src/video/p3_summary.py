"""P3e: short summary video (60-90 s) of the most important shots from the
original full-res copy, chosen by motion + keyframe score + scene diversity."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[2]
CACHE = OUT / "cache" / "p3"
R = OUT / "results" / "P3"
R.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(OUT / "src"))
from lib_data import ffmpeg_exe, media_info

SYMBOL_SHOTS = 20
SHOT_LENS = [3.5, 4.0, 4.0, 3.5, 4.0, 4.0, 4.0, 4.0, 3.5, 4.0, 4.0, 4.0,
             3.5, 4.0, 4.0, 4.0, 3.5, 4.0, 4.0, 4.0]


def main():
    prep = json.loads((CACHE / "prep.json").read_text())
    orig = Path(prep["merged"])
    total_dur = float(prep["merged_seconds"])
    if total_dur <= 0:      # refresh if prep recorded it before media_info fix
        total_dur, _ = media_info(orig)
        prep["merged_seconds"] = total_dur
        (CACHE / "prep.json").write_text(json.dumps(prep, indent=2))
    fps = json.loads((CACHE / "stream_progress.json").read_text())["fps"]
    scenes = [s for s in json.loads((CACHE / "scenes.json").read_text())["scenes"]]
    kf = json.loads((CACHE / "keyframes.json").read_text())
    key_rows = {r["frame"]: r for r in kf["key_rows"]}
    with np.load(CACHE / "frames_stats.npz") as z:
        motion = z["motion"]

    # importance per shot: mean motion + max keyframe score + rank-derived
    # diversity bonus (spread across the timeline)
    scored = []
    for i, (s0, s1) in enumerate(scenes):
        i0, i1 = int(s0 * fps), min(int(s1 * fps), len(motion))
        if i1 <= i0:
            continue
        m = float(np.mean(motion[i0:i1]) + 2.0 * float(np.max(motion[i0:i1])))
        kfs = [r for f, r in key_rows.items() if i0 <= f < i1]
        ks = max((r["score"] for r in kfs), default=0.0) if kfs else 0.0
        dur = min(s1 - s0, 30.0)
        scored.append({"scene": i, "t0": s0, "t1": s1, "dur": dur,
                       "motion": m, "kf": ks,
                       "imp": m / 3 + ks + 0.15 * dur / 10})
    # diversity: greedy pick maximising importance - overlap penalty
    picks = []
    for _ in range(SYMBOL_SHOTS):
        best = None
        for c in scored:
            pen = sum(1 if not (c["t1"] < p["t0"] - 6 or c["t0"] > p["t1"] + 6)
                      else 0 for p in picks)
            cost = -c["imp"] + 2.0 * pen
            if best is None or cost < best[0]:
                best = (cost, c)
        if best is None:
            break
        c = best[1]
        c = dict(c)
        c["len"] = min(SHOT_LENS[len(picks) % len(SHOT_LENS)], c["dur"],
                       max(1.0, total_dur - c["t0"] - 0.5))
        if c["len"] > 1.2:
            picks.append(c)

    picks.sort(key=lambda c: c["t0"])
    # build the summary with ffmpeg (cut from the ORIGINAL high-res copy)
    parts = []
    total_sum = 0.0
    for i, c in enumerate(picks):
        p = CACHE / f"_sum_{i}.mp4"
        len_c = max(1.0, c["len"])
        subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                        "-ss", f"{c['t0']:.2f}", "-i", str(orig),
                        "-t", f"{len_c:.2f}",
                        "-vf", "scale=1280:-2,setsar=1",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                        "-an", str(p)], check=True)
        d, _ = media_info(p)
        total_sum += d if d > 0 else len_c
        parts.append(p)
    lst = CACHE / "_sum_concat.txt"
    lst.write_text("\n".join(f"file '{p.as_posix()}'" for p in parts),
                   encoding="utf-8")
    out = R / "P3_summary.mp4"
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    str(out)], check=True)
    for p in parts:
        p.unlink(missing_ok=True)
    lst.unlink(missing_ok=True)
    ratio = total_sum / (total_dur or 1)
    (R / "P3_summary_manifest.json").write_text(json.dumps(
        {"picks": picks, "summary_seconds": total_sum,
         "source_seconds": total_dur, "compression_ratio": ratio}, indent=2))
    print(f"[p3-summary] {len(picks)} shots, {total_sum:.0f}s of {total_dur:.0f}s "
          f"(x{ratio:.3f}) -> {out}")
    return out


if __name__ == "__main__":
    main()
