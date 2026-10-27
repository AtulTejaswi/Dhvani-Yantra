"""P1 dataset builder: reference + student pronunciations of 5 Sanskrit words.

Words: कृष्ण krishna, धर्म dharma, योग yoga, शान्ति shanti, ॐ om.

Reference side: first tries Wikimedia Commons pronunciation files (real speech,
freely licensed). Whatever is not found falls back to Edge TTS (hi-IN voices,
synthesized Devanagari) and is labelled SYNTHETIC in SOURCES.md and the app.
Student side: (a) en-IN TTS saying romanised variants (krishna / kishna /
krisna ...), and (b) controlled perturbations of the reference at several
severities (time-stretch, noise, consonant deletion, vowel-shift via pitch).
"""
from __future__ import annotations

import json
import sys
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OUT / "src"))

from lib_data import (OUT as OUTDIR, DATA, SOURCES_MD, logged, tts, tts_gtts,
                      to_wav16, fetch, ffmpeg_exe)
import lib_audio as LA
import soundfile as sf

WORDS = {
    "krishna": {"deva": "कृष्ण", "roman": "krishna", "variants": ["krisna", "kishna"]},
    "dharma": {"deva": "धर्म", "roman": "dharma", "variants": ["dhram", "dharm"]},
    "yoga":   {"deva": "योग", "roman": "yoga",   "variants": ["yog", "yuga"]},
    "shanti": {"deva": "शान्ति", "roman": "shanti", "variants": ["shant", "shanti"]},
    "om":     {"deva": "ॐ", "roman": "om",       "variants": ["aum", "om"]},
}

AUD = DATA / "audio"
META_PATH = AUD / "p1_dataset.json"

# ---------------------------------------------------------------------------
# 1. reference acquisition
# ---------------------------------------------------------------------------
WIKI_CANDIDATES = {
    # hand-listed Commons filenames that plausibly exist; verified by HTTP probe
    "krishna": ["https://upload.wikimedia.org/wikipedia/commons/0/04/Krishna.ogg"],
    "yoga":    ["https://upload.wikimedia.org/wikipedia/commons/9/9b/Yoga.ogg"],
    "dharma":  ["https://upload.wikimedia.org/wikipedia/commons/b/b9/Dharma.ogg"],
    "shanti":  ["https://upload.wikimedia.org/wikipedia/commons/5/5f/Shanti.ogg"],
    "om":      ["https://upload.wikimedia.org/wikipedia/commons/b/b2/OM.ogg"],
}
REF_VOICES = ["hi-IN-MadhurNeural", "hi-IN-SwaraNeural"]


def build_reference() -> dict[str, dict]:
    AUD.mkdir(parents=True, exist_ok=True)
    refs = {}
    for key, w in WORDS.items():
        out = AUD / f"ref_{key}.wav"
        if out.exists():
            refs[key] = {"path": str(out), "kind": "ok-known"}
            continue
        got = False
        # (1) probe Wikimedia candidates
        for url in WIKI_CANDIDATES[key]:
            raw = AUD / f"_wiki_{key}_{Path(url).name}"
            if fetch(url, raw, timeout=25):
                wav = to_wav16(raw, out)
                raw.unlink(missing_ok=True)
                if wav:
                    logged("web", url, out, "see Commons file page (CC-BY/-SA/PD)",
                           note="[wilhelm: verify licence on Commons]")
                    got = True
                    refs[key] = {"path": str(out), "kind": "wikimedia"}
                    break
        # (2) fallback: edge-tts of the Devanagari word
        if not got:
            for voice in REF_VOICES:
                mp3 = AUD / f"_ttsref_{key}.mp3"
                p = tts(w["deva"], mp3, voice=voice)
                if p:
                    wav = to_wav16(p, out)
                    p.unlink(missing_ok=True)
                    if wav:
                        logged("tts", f"edge-tts voice:{voice} text:'{w['deva']}'",
                               out, "edge-tts output (Microsoft Neural TTS)",
                               note="[SYNTHETIC REFERENCE]")
                        # keep one alternate reference voice too
                        alt = AUD / f"ref_{key}_alt.wav"
                        mp3b = AUD / f"_ttsref_{key}_alt.mp3"
                        vb = REF_VOICES[1] if voice == REF_VOICES[0] else REF_VOICES[0]
                        pb = tts(w["deva"], mp3b, voice=vb)
                        if pb:
                            to_wav16(pb, alt)
                            pb.unlink(missing_ok=True)
                        got = True
                        refs[key] = {"path": str(out), "kind": "tts", "voice": voice}
                        break
    return refs


# ---------------------------------------------------------------------------
# 2. student acquisitions
# ---------------------------------------------------------------------------
STUDENT_VOICES = ["en-IN-PrabhatNeural", "en-IN-NeerjaNeural"]


def tts_student(ref_key: str, roman: str, fname: str) -> Path | None:
    out = AUD / fname
    if out.exists():
        return out
    for voice in STUDENT_VOICES:
        mp3 = AUD / "_st.mp3"
        p = tts(roman, mp3, voice=voice)
        if p:
            wav = to_wav16(p, out)
            mp3.unlink(missing_ok=True)
            if wav:
                logged("tts", f"edge-tts voice:{voice} text:'{roman}'", out,
                       "edge-tts output", note="[SYNTHETIC STUDENT TAKE]")
                return out
    return None


def perturb(ref_key: str, y: np.ndarray, jitter: str, tag: str) -> Path:
    """Deterministic (seeded) controlled perturbations.

    The word core is trimmed FIRST so every family (incl. additive noise)
    has the same base duration, then the family transform is applied.
    """
    out = AUD / f"pert_{ref_key}_{tag}.wav"
    if out.exists():
        return out
    rs = np.random.default_rng(hash((ref_key, tag)) % (2 ** 32))
    y = LA.trim_silence(y)
    if "stretch" in tag:
        r = float(tag.split("_")[1]) / 100.0  # 5 -> 5 % slower take
        y2 = LA.sola_stretch(y, 1.0 / (1.0 + r))
    elif "pitch" in tag:
        r = int(tag.split("_")[1])
        y2 = librosa_pitch_shift(y, r)
    elif "noise" in tag:
        snr = float(tag.split("_")[1])
        y2 = add_noise(y, snr, rs)
    elif "cut" in tag:
        frac = float(tag.split("_")[1]) / 100.0
        cut_early = "e" == tag[-1]
        y2 = delete_onset(y, frac, cut_early)
    else:
        y2 = y
    y2 = LA.trim_silence(y2)
    y2 = LA.peak_norm(y2)
    sf.write(out, y2, LA.SR)
    return out


def librosa_time_stretch(y, rate):
    import librosa
    return librosa.effects.time_stretch(y, rate=rate)


def librosa_pitch_shift(y, semitones):
    import librosa
    return librosa.effects.pitch_shift(y, sr=LA.SR, n_steps=semitones)


def add_noise(y, snr_db, rs):
    px = y ** 2
    snr_lin = 10 ** (snr_db / 10.0)
    pn = px.mean() / snr_lin
    return (y + rs.normal(0, pn ** 0.5, y.shape)).astype(np.float32)


def delete_onset(y, frac, cut_early=True):
    # restrict to a single leading consonant window: first 40% duration
    n0 = int(len(y) * 0.40)
    if cut_early:
        y2 = np.concatenate([y[:int(n0 * (1.0 - frac))], y[n0:]])
    else:
        y2 = np.concatenate([y[:n0], y[int(n0 + (len(y) - n0) * frac * 0.5):]])
    return y2


def build_students(refs: dict) -> dict[str, list[dict]]:
    students = {k: [] for k in WORDS}
    for key, w in WORDS.items():
        # (a) natural-ish TTS variants of the word ("student" takes)
        roman0 = w["roman"]
        s0 = tts_student(key, roman0, f"stud_{key}.wav")
        if s0:
            students[key].append({"path": str(s0), "label": roman0, "kind": "tts-student"})
        for i, var in enumerate(w["variants"]):
            si = tts_student(key, var, f"stud_{key}_{i}_{var}.wav")
            if si:
                students[key].append({"path": str(si), "label": var, "kind": "tts-student"})
        # (b) controlled perturbations of the reference itself
        import librosa
        y_ref, _ = librosa.load(refs[key]["path"], sr=LA.SR)
        # 15 unique severity-ordered perturbations
        tags = ([f"stretch_{r}" for r in (15, 30, 50, 80)]
                + [f"noise_{s}" for s in (35, 30, 22, 10)]
                + [f"cut_{f}_e" for f in (11, 13, 16, 20)]
                + [f"pitch_{p}" for p in (1, 2, 3)])
        for t in tags:
            p = perturb(key, y_ref, t, t)
            students[key].append({"path": str(p), "label": t, "kind": "perturbation"})
    return students


# ---------------------------------------------------------------------------
# 3. main
# ---------------------------------------------------------------------------
def main() -> None:
    print("[p1] building references ...")
    refs = build_reference()
    print(f"[p1] references ok: {list(refs)}")
    print("[p1] building students ...")
    students = build_students(refs)
    n = sum(len(v) for v in students.values())
    meta = {"refs": refs, "students": students,
            "words": WORDS, "n_students": n}
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"[p1] {n} student recordings for {len(WORDS)} words -> {META_PATH}")
    # sanity: everything must exist
    missing = [p for v in students.values() for p in [f["path"] for f in v]
               if not Path(p).exists()]
    if missing:
        print("[p1] WARNING missing:", missing)


if __name__ == "__main__":
    main()
