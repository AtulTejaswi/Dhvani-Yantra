"""TTS helpers + free-media fetchers with fallbacks.

Order of preference per the project rules:
1. try a real, freely licensed recording (Wikimedia Commons / archive.org),
2. fall back to open TTS (edge-tts -> gTTS), and record that the item is synthetic.

Every fetch result is appended to data/SOURCES.md with URL + licence.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OUT / "src"))

DATA = OUT / "data"
SOURCES_MD = DATA / "SOURCES.md"
FFMPEG = None


def ffmpeg_exe() -> str:
    global FFMPEG
    if FFMPEG is None:
        try:
            import imageio_ffmpeg
            FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            FFMPEG = "ffmpeg"
    return FFMPEG


def logged(kind: str, url: str, path: str, licence: str, note: str = "") -> None:
    SOURCES_MD.parent.mkdir(exist_ok=True, parents=True)
    flag = " (SYNTHETIC)" if kind == "tts" else ""
    with open(SOURCES_MD, "a", encoding="utf-8") as fh:
        fh.write(f"- `{Path(path).name}`{flag} | kind: {kind} | source: {url} "
                 f"| licence: {licence} {note}\n")


# ---------------------------------------------------------------------------
# network fetch with timeout
# ---------------------------------------------------------------------------
import urllib.request
import urllib.error

def fetch(url: str, dst: str | Path, timeout: int = 60,
          headers: dict | None = None) -> Path | None:
    try:
        req = urllib.request.Request(url, headers=headers or {
            "User-Agent": "Mozilla/5.0 (SVP-assignment-fetch; educational use)"})
        with urllib.request.urlopen(req, timeout=timeout) as r, \
                open(dst, "wb") as f:
            f.write(r.read())
        return Path(dst)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
def tts(text: str, out_path: str | Path, voice: str = "hi-IN-MadhurNeural",
        rate: str = "+0%", pitch: str = "+0Hz", retries: int = 2) -> Path | None:
    """Synthesize with edge-tts (open service, no key). Returns path or None."""
    import asyncio

    async def run():
        import edge_tts
        for attempt in range(retries + 1):
            try:
                com = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
                await com.save(out_path)
                if os.path.getsize(out_path) > 500:
                    return Path(out_path)
            except Exception:
                if attempt == retries:
                    raise
                await asyncio.sleep(2)
        return None

    try:
        return asyncio.run(run())
    except Exception:
        return None


def tts_gtts(text: str, out_path: str | Path, lang: str = "hi") -> Path | None:
    """Fallback TTS (gTTS)."""
    try:
        from gtts import gTTS
        gTTS(text, lang=lang).save(out_path)
        return Path(out_path) if os.path.getsize(out_path) > 500 else None
    except Exception:
        return None


def to_wav16(src: str | Path, dst: str | Path, sr: int = 16000) -> Path | None:
    """Convert/transcode any audio to 16 kHz mono wav; also used to splice video."""
    import subprocess
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(src), "-ac", "1", "-ar", str(sr), str(dst)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return Path(dst)
    except Exception:
        return None


def media_info(path: str | Path) -> tuple[float, float]:
    """(duration_sec, n_frames) via ffmpeg stderr parse; (-1,-1) on failure."""
    import subprocess
    import re
    try:
        r = subprocess.run([ffmpeg_exe(), "-i", str(path), "-f", "null", "-"],
                           capture_output=True, text=True, errors="replace",
                           timeout=300)
        m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", r.stderr)
        f_m = re.search(r"(?:V:|Video:).+?([\d.]+) fps", r.stderr)
        if m:
            dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
            fps = float(f_m.group(1)) if f_m else -1.0
            return dur, int(dur * fps) if fps > 0 else -1
        return -1, -1
    except Exception:
        return -1, -1
