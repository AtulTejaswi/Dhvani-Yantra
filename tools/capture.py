"""Automated screenshots + walkthrough video of the live Streamlit app.

Launches the app headless, drives it with Playwright (system Edge/Chrome,
NO `playwright install` needed), and captures 1920x1080 PNGs of every output
state, plus a captioned walkthrough recording converted to mp4.

Usage:  python tools/capture.py [--shots-only] [--video-only]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1]
for _sub in ("", "speech_a", "speech_b", "video"):
    sys.path.insert(0, str(OUT / "src" / _sub))
SHOTS = OUT / "screenshots"
SHOTS.mkdir(exist_ok=True)
(OUT / "logs").mkdir(exist_ok=True)
PORT = 8599
APP_URL = f"http://localhost:{PORT}"


def start_app():
    py = sys.executable
    env = os.environ | {"PYTHONIOENCODING": "utf-8",
                        "PYTHONPATH": str(OUT / "src")}
    proc = subprocess.Popen(
        [py, "-m", "streamlit", "run", str(OUT / "app.py"),
         "--server.port", str(PORT), "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=str(OUT), env=env,
        stdout=open(OUT / "logs" / "streamlit.log", "w", encoding="utf-8"),
        stderr=subprocess.STDOUT)
    import urllib.request
    for _ in range(90):
        try:
            urllib.request.urlopen(APP_URL, timeout=2)
            return proc
        except Exception:
            time.sleep(1)
    raise RuntimeError("streamlit did not come up; see logs/streamlit.log")


def find_pw_browser(p):
    """playwright channel preference: msedge -> chrome -> bundled chromium."""
    for channel in ("msedge", "chrome", None):
        try:
            ex = p.chromium.launch(headless=True,
                                   **({"channel": channel} if channel else {}))
            ex.close()
            return channel
        except Exception:
            continue
    raise RuntimeError("no chromium-based browser found for playwright")


def launch(p, channel, headless=True):
    return p.chromium.launch(headless=headless,
                             **({"channel": channel} if channel else {}))


def click_button(page, label: str, timeout: int = 4000) -> bool:
    name = re.sub(r"^[^A-Za-z0-9]+", "", label)  # drop leading emoji/icon
    cands = [page.get_by_role("button", name=label),
             page.get_by_role("button", name=name),
             page.locator("button", has_text=name).last]
    for loc in cands:
        try:
            loc.click(timeout=timeout)
            return True
        except Exception:
            continue
    return False


def click_and_wait(page, label: str, marker: str, timeout: int = 40000,
                   attempts: int = 3):
    """Click a Streamlit button; retried because clicks during a rerun are
    dropped, and verified via a result marker becoming visible."""
    for attempt in range(attempts):
        click_button(page, label, timeout=timeout // 3)
        try:
            page.get_by_text(marker).first.wait_for(
                state="visible", timeout=timeout)
            if attempt:
                print(f"    (button '{label}' needed {attempt + 1} tries)")
            return
        except Exception:
            page.wait_for_timeout(2000)
    raise RuntimeError(f"button '{label}' had no effect "
                       f"(marker '{marker}' never visible)")


def click_tab(page, name: str, marker: str, timeout: int = 20000):
    """Switch to a Streamlit tab button and wait until its content is visible.

    Retries: a click issued while Streamlit is mid-rerun is silently dropped.
    """
    tab = page.locator('[role="tab"]', has_text=name).first
    for attempt in range(4):
        tab.click(timeout=timeout)
        try:
            page.get_by_text(marker).first.wait_for(
                state="visible", timeout=6000)
            return
        except Exception:
            page.wait_for_timeout(1500)
    raise RuntimeError(f"tab '{name}' did not switch (marker '{marker}' "
                       "never visible)")


def scroll(page, frac: float):
    """Scroll the Streamlit main container; frac=1.0 -> bottom, 0 -> top."""
    page.evaluate(
        """(frac) => { const el = document.querySelector('[data-testid="stMain"]')
              || document.scrollingElement;
            if (el) { el.scrollTop = frac * (el.scrollHeight - el.clientHeight); }
            window.scrollTo(0, frac * document.body.scrollHeight); }""",
        frac)
    page.wait_for_timeout(900)


def shot(page, name: str, caption: str):
    page.wait_for_timeout(2600)
    path = SHOTS / f"{name}.png"
    page.screenshot(path=str(path), full_page=False)
    print(f"  [shot] {name}.png")
    return name, caption


def banner(page, text: str, ms: int = 2500):
    try:
        page.evaluate(
            """(t) => { const d = document.getElementById("__demo_banner");
                  if (!d) { d = document.createElement('div'); d.id='__demo_banner';
                    d.style.cssText = 'position:fixed;bottom:24px;left:24px;z-index:' +
                      '99999;background:#101418;color:#8dff96;padding:14px 22px;' +
                      'font-size:20px;border-radius:10px;border:2px solid #8dff96;' +
                      'font-family:sans-serif';
                    document.body.appendChild(d); }
                  d.textContent = t; }""", text)
    except Exception:
        pass
    page.wait_for_timeout(ms)


def shots_pass(p, channel, index_rows):
    browser = launch(p, channel)
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    page.goto(APP_URL)
    page.wait_for_selector("text=Pronunciation Assessment", timeout=30000)
    page.wait_for_timeout(5000)
    index_rows.append(shot(page, "01_pronunciation_home",
                           "Pronunciation tab: user picks a reference word and a "
                           "student take (or uploads/records)."))
    click_and_wait(page, "Compare & score", "Pronunciation similarity")
    index_rows.append(shot(page, "02_pronunciation_result",
                           "Live result: similarity %, verdict, DTW metrics, "
                           "segment-level diagnosis."))
    page.wait_for_timeout(1200)
    scroll(page, 0.55)
    shot(page, "03_pronunciation_plots",
         "MFCC / spectral / DTW-path analysis chart of the pair.")

    # tab 2
    click_tab(page, "Gita Verse Recognition", "Identify verse")
    page.wait_for_timeout(2500)
    scroll(page, 0)
    index_rows.append(shot(page, "04_gita_home",
                           "Gita tab: choose a sample clip or upload/record."))
    click_and_wait(page, "Identify verse", "Held-out-voice test accuracy")
    index_rows.append(shot(page, "05_gita_result",
                           "Verse ID + top-3 softmax ranking + per-verse "
                           "log-likelihood chart."))
    scroll(page, 1)
    index_rows.append(shot(page, "06_gita_confusion",
                           "Confusion matrix + state-selection grid over the "
                           "held-out-voice test set."))
    # tab 3
    click_tab(page, "Video Processing", "Important frames")
    page.wait_for_timeout(5000)
    scroll(page, 0)
    index_rows.append(shot(page, "07_video_keyframes",
                           "Gallery of important frames + scene timeline."))
    scroll(page, 0.45)
    index_rows.append(shot(page, "08_video_motion",
                           "Cumulative motion heatmap and per-second energy."))
    scroll(page, 0.75)
    index_rows.append(shot(page, "09_video_tracking",
                           "Tracking video (YOLO + ByteTrack) and trajectories."))
    scroll(page, 1)
    index_rows.append(shot(page, "10_video_summary",
                           "60-90 s summary video player + compression ratio."))
    browser.close()


def video_pass(p, channel):
    browser = launch(p, channel, headless=False)
    vctx = browser.new_context(viewport={"width": 1600, "height": 1000},
                               record_video_dir=str(OUT / "cache" / "walkthrough"),
                               record_video_size={"width": 1600, "height": 1000})
    vpage = vctx.new_page()
    vpage.goto(APP_URL)
    vpage.wait_for_timeout(4000)
    steps = [
        ("Welcome — Sanskrit Speech + Video project live demo", 3500),
        ("Tab 1: Pronunciation assessment (MFCC + DTW)", 2500),
        ("Compare a student take against the reference कृष्ण", 2500),
    ]
    for cap, ms in steps:
        banner(vpage, cap, ms)
    click_and_wait(vpage, "Compare & score", "Pronunciation similarity")
    vpage.wait_for_timeout(5000)
    banner(vpage, "Tab 2: Gita verse recognition (MFCC + HMM)")
    click_tab(vpage, "Gita Verse Recognition", "Identify verse")
    vpage.wait_for_timeout(2500)
    banner(vpage, "Identify the verse by HMM log-likelihood")
    click_and_wait(vpage, "Identify verse", "Held-out-voice test accuracy")
    vpage.wait_for_timeout(1500)
    banner(vpage, "Confusion matrix over the unseen-voice test set", 1500)
    banner(vpage, "Tab 3: Video processing")
    click_tab(vpage, "Video Processing", "Important frames")
    vpage.wait_for_timeout(4000)
    banner(vpage, "Important frames + scene timeline", 4000)
    scroll(vpage, 0.5)
    banner(vpage, "Motion map, per-second energy, tracked objects and summary", 5000)
    scroll(vpage, 1)
    banner(vpage, "Demo complete — Thank you!", 3000)
    video = vpage.video
    vctx.close()
    vp = Path(video.path())
    browser.close()
    from lib_data import ffmpeg_exe  # src/ is on sys.path via app.py fallback
    out_mp4 = OUT / "demo_walkthrough.mp4"
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(vp),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "26",
                    str(out_mp4)], check=True)
    print("  [video] demo_walkthrough.mp4 written")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--shots-only", action="store_true")
    a.add_argument("--video-only", action="store_true")
    args = a.parse_args()

    proc = start_app()
    try:
        import playwright.sync_api as pw
        do_shots = not args.video_only
        do_video = not args.shots_only
        index_rows = []
        with pw.sync_playwright() as p:
            channel = find_pw_browser(p)
            print(f"[capture] playwright channel = {channel!r}")
            if do_shots:
                shots_pass(p, channel, index_rows)
            if do_video:
                video_pass(p, channel)
        if do_shots:
            with open(SHOTS / "INDEX.md", "w", encoding="utf-8") as f:
                f.write("# Screenshot index\n\n")
                for i, (name, cap) in enumerate(index_rows, 1):
                    rows = cap.split("\n")
                    for r in rows:
                        pass
                    f.write(f"{i}. `{name}.png` — {rows[0]}\n")
            print("[capture] INDEX.md written")
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
