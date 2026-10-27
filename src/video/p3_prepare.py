"""P3 data prep: download 3 CC-BY Blender/previous 'open movie' clips, concatenate
to ~30 min, and build a small 640x360@10fps analysis copy for streaming pipelines."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OUT / "src"))
from lib_data import ffmpeg_exe, media_info, logged, SOURCES_MD

VID = OUT / "data" / "video"
CACHE = OUT / "cache" / "p3"

# all CC-BY, youtube-free, Blender open movies on archive.org
FILES = [
    # (item, name, dst, licence, source, role)
    ("Tears-of-Steel", "tears_of_steel_720p.mp4", VID / "raw_tears_of_steel_720p.mp4",
     "CC-BY-3.0 (Blender Foundation)", "https://archive.org/details/Tears-of-Steel",
     "live action with human actors - primary people video"),
    ("Sintel", "sintel-2048-stereo_512kb.mp4", VID / "raw_sintel_512kb.mp4",
     "CC-BY-3.0 (Blender Foundation)", "https://archive.org/details/Sintel",
     "animation with humanoid characters - filler to reach ~30 min"),
    ("ElephantsDream", "ed_hd_512kb.mp4", VID / "raw_elephants_dream_512kb.mp4",
     "CC-BY-3.0-US (Blender Foundation)", "https://archive.org/details/ElephantsDream",
     "animation - filler to reach ~30 min"),
]
TARGET_LEN = 1800.0  # ~30 minutes


def dl(item: str, fname: str, dst: Path) -> bool:
    url = f"https://archive.org/download/{item}/{fname}"
    if dst.exists() and dst.stat().st_size > 100_000:
        print(f"  [p3] cached {dst.name}")
        return True
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (SVP-assignment; educational)"})
        with urllib.request.urlopen(req, timeout=1800) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
        ok = dst.stat().st_size > 100_000
        print(f"  [p3] downloaded {fname} -> {dst.name} ({dst.stat().st_size // 2 ** 20} MB)")
        return ok
    except Exception as e:
        print(f"  [p3] FAILED {url}: {e}")
        return False


def concat_clips(paths: list[Path], dst: Path) -> bool:
    ff = ffmpeg_exe()
    # 1) normalise every clip to identical codec/size/fps (needed by concat demuxer)
    parts = []
    for i, src in enumerate(paths):
        p = dst.parent / f"_part{i}_720p.mp4"
        parts.append(p)
        if not p.exists():
            subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y",
                            "-i", str(src), "-t", "700",  # cap each at ~11:40 => 3 parts = ~35 min
                            "-vf", "scale=1280:-2:flags=lanczos,fps=24",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                            "-an", str(p)], check=True)
    # 2) concat until TARGET_LEN seconds, trimmed
    lst = dst.parent / "_concat.txt"
    lst.write_text("\n".join(f"file '{p.as_posix()}'" for p in parts), encoding="utf-8")
    if not dst.exists():
        subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-t", str(TARGET_LEN),
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
                        "-an", str(dst)], check=True)
    for p in parts:
        p.unlink(missing_ok=True)
    lst.unlink(missing_ok=True)
    return dst.exists()


def make_analysis_copy(src: Path, dst: Path, fps: float = 10.0, w: int = 640) -> bool:
    if dst.exists():
        return True
    ff = ffmpeg_exe()
    subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(src), "-vf", f"fps={fps},scale={w}:-2:flags=lanczos",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-an", str(dst)], check=True)
    return True


def main() -> None:
    VID.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    # 1) downloads, with licence logging
    got = []
    for item, fname, dst, lic, src, role in FILES:
        if dl(item, fname, dst):
            logged("web", src, dst, lic, f"[{role}]")
            got.append(dst)
        else:
            print(f"[p3] skipping {item}")
    if not got:
        raise SystemExit("[p3] no video source available at all")
    # 2) build the ~30 min movie
    merged = VID / "movie_30min.mp4"
    print("[p3] concatenating clips ...")
    if concat_clips(got, merged):
        dur, _ = media_info(merged)
        print(f"[p3] merged length = {dur / 60:.1f} min")
    # 3) analysis copy
    ana = CACHE / "analysis_640p_10fps.mp4"
    print("[p3] making analysis copy ...")
    make_analysis_copy(merged, ana)
    dur, n = media_info(ana)
    meta = {"merged": str(merged), "merged_seconds": dur,
            "analysis": str(ana), "sources": [
                {item: str(dst), "licence": lic, "source": src}
                for (item, fname, dst, lic, src, role) in FILES if dst in got],
            }
    (CACHE / "prep.json").write_text(json.dumps(meta, indent=2))
    print(f"[p3] analysis copy: {dur / 60:.1f} min, est frames {n}")


if __name__ == "__main__":
    main()
