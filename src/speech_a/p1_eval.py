"""P1 evaluation: calibration, full results table, and publication-quality plots."""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf

OUT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OUT / "src"))
import lib_audio as LA
import p1_core as P1

R = OUT / "results" / "P1"
R.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)

SPEC_RANK = {"identical": 3, "tts-student": 2, "perturbation": 1, "other-word": 0}


def severity_of(label: str, kind: str) -> float:
    """0..1 controlled severity for perturbations; 0 for identical; 0.5 for TTS."""
    if kind == "identical":
        return 0.0
    if kind == "tts-student":
        return 0.5
    t = label
    try:
        if t.startswith("stretch"):
            return np.interp(float(t.split("_")[1]), [15, 30, 50, 80], [0.15, 0.35, 0.55, 0.9])
        if t.startswith("noise"):
            return np.interp(float(t.split("_")[1]), [10, 22, 30, 35], [0.9, 0.55, 0.35, 0.15])
        if t.startswith("cut"):
            return np.interp(float(t.split("_")[1]), [11, 13, 16, 20], [0.3, 0.45, 0.65, 0.85])
        if t.startswith("pitch"):
            return np.interp(float(t.split("_")[1]), [1, 2, 3], [0.25, 0.5, 0.8])
    except Exception:
        return 0.5
    return 0.5


def word_text(w: dict) -> str:
    return w["deva"]


def similarity_text(score: float) -> str:
    return f"Pronunciation similarity: {score:.0f}%"


def run() -> dict:
    import librosa
    p1 = OUT / "src" / "speech_a"
    sys.path.insert(0, str(p1))
    from p1_build_data import WORDS

    meta = json.loads((OUT / "data" / "audio" / "p1_dataset.json").read_text("utf-8"))
    results = {}
    calib_pairs = []

    # ---- calibration pass -------------------------------------------------
    print("[p1-eval] calibration on synthetic labelled pairs ...")
    for key, meta_w in meta["words"].items():
        y_ref, _ = librosa.load(meta["refs"][key]["path"], sr=LA.SR)
        # identical pair
        d0, _, _ = compare_or_cache(key, y_ref, y_ref, "identical", key)
        calib_pairs.append((d0, 100.0))
        # cross-word pair (clearly different word)
        other = [k for k in meta["words"] if k != key][0]
        y_o, _ = librosa.load(meta["refs"][other]["path"], sr=LA.SR)
        d1, _, _ = compare_or_cache(key, y_ref, y_o, "other-word", f"x-{other}")
        calib_pairs.append((d1, 0.0))
    alpha, loss = P1.calibrate_alpha(calib_pairs)
    P1.ALPHA = alpha
    (CACHE / "p1_alpha.json").write_text(json.dumps(
        {"alpha": alpha, "calibration_loss": loss}, indent=2), encoding="utf-8")
    print(f"[p1-eval] alpha = {alpha:.4f} (calib loss {loss:.2f})")
    cross_scores = [100 * np.exp(-alpha * P1.weighted_distance(d))
                    for d, t in calib_pairs if t == 0.0]

    # ---- main pass --------------------------------------------------------
    rows = []
    charts = []
    for key, meta_w in meta["words"].items():
        y_ref, _ = librosa.load(meta["refs"][key]["path"], sr=LA.SR)
        entries = []
        # identical self-comparison (acceptance anchor)
        d0, _, _ = compare_or_cache(key, y_ref, y_ref, "identical", "identical")
        s0 = 100 * np.exp(-P1.ALPHA * P1.weighted_distance(d0))
        segs0 = d0["segments"]
        v0 = [s for s in segs0 if s["voiced"]]
        c0 = [s for s in segs0 if not s["voiced"]]
        entries.append({
            "label": "identical (reference vs itself)", "kind": "identical",
            "path": meta["refs"][key]["path"], "score": float(s0),
            "d_mfcc": d0["d_mfcc"], "d_spec": d0["d_spec"],
            "dur_ratio": 1.0, "d_dur": 0.0, "d_f0": d0["d_f0"],
            "F1_ref": d0.get("F1_ref", 0), "F2_ref": d0.get("F2_ref", 0),
            "F1_stu": d0.get("F1_stu", 0), "F2_stu": d0.get("F2_stu", 0),
            "vowel_err": float(np.mean([s["err"] for s in v0])) if v0 else 0.0,
            "consonant_err": float(np.mean([s["err"] for s in c0])) if c0 else 0.0,
            "explanation": "same recording compared with itself",
            "verdict": f"Pronunciation similarity: {s0:.0f}%",
            "severity": 0.0, "segments": segs0,
            "dtw_path": [list(x) for x in d0["dtw_path"]],
        })
        for stu in meta["students"][key]:
            y, _ = librosa.load(stu["path"], sr=LA.SR)
            if stu["kind"] == "perturbation":
                pass
            d, _, _ = compare_or_cache(key, y_ref, y, stu["kind"], stu["label"])
            score = 100 * np.exp(-P1.ALPHA * P1.weighted_distance(d))
            seg = d["segments"]
            v = [s for s in seg if s["voiced"]]
            c = [s for s in seg if not s["voiced"]]
            v_err = float(np.mean([s["err"] for s in v])) if v else 0.0
            c_err = float(np.mean([s["err"] for s in c])) if c else 0.0
            from p1_core import verdict_text
            verdict = verdict_text(score, d, v_err, c_err)
            # per-segment explanation text
            txtbits = []
            if v_err > c_err * 1.25:
                txtbits.append("vowel shifted")
            if c_err > v_err * 1.25:
                txtbits.append("consonant cluster weak")
            if d["dur_ratio"] > 1.08:
                txtbits.append(f"duration +{(d['dur_ratio'] - 1) * 100:.0f}%")
            entries.append({
                "label": stu["label"], "kind": stu["kind"], "path": stu["path"],
                "score": float(score), "d_mfcc": d["d_mfcc"], "d_spec": d["d_spec"],
                "dur_ratio": d["dur_ratio"], "d_dur": d["d_dur"], "d_f0": d["d_f0"],
                "F1_ref": d.get("F1_ref", 0), "F2_ref": d.get("F2_ref", 0),
                "F1_stu": d.get("F1_stu", 0), "F2_stu": d.get("F2_stu", 0),
                "vowel_err": v_err, "consonant_err": c_err,
                "explanation": " ; ".join(txtbits) or "segments close to reference",
                "verdict": verdict,
                "severity": severity_of(stu["label"], stu["kind"]),
                "segments": seg, "dtw_path": [list(x) for x in d["dtw_path"]],
            })
        entries.sort(key=lambda e: e["severity"])
        results[key] = {"ref": meta["refs"][key], "deva": meta_w["deva"],
                        "roman": meta_w["roman"], "entries": entries}

        # ---- plots for this word (first TTS student = most interesting) ----
        plot_dir = R
        demo = next((e for e in entries if e["kind"] == "tts-student"), entries[0])
        p_demo, _ = librosa.load(demo["path"], sr=LA.SR)
        d_demo, ref_t, stu_t = P1.analyse_pair(y_ref, p_demo)
        make_word_figure(key, meta_w["deva"], y_ref, p_demo, d_demo,
                         ref_t, stu_t, plot_dir)
        # monotonicity figure and per-word entry table
        sev = [e["severity"] for e in entries if e["kind"] == "perturbation"]
        sco = [e["score"] for e in entries if e["kind"] == "perturbation"]
        order = np.argsort(sev)
        make_monotonic_figure(key, [sev[i] for i in order], [sco[i] for i in order])
        for e in entries:
            rows.append({"word": meta_w["deva"], "roman": meta_w["roman"],
                         "student": e["label"], "kind": e["kind"],
                         "score": round(e["score"], 1),
                         "mfcc_dist": round(e["d_mfcc"], 4),
                         "spec_dist": round(e["d_spec"], 4),
                         "dur_ratio": round(e["dur_ratio"], 3),
                         "f0_dist": round(e["d_f0"], 4),
                         "F1_ref": round(e["F1_ref"], 1), "F2_ref": round(e["F2_ref"], 1),
                         "F1_stu": round(e["F1_stu"], 1), "F2_stu": round(e["F2_stu"], 1),
                         "vowel_err": round(e["vowel_err"], 3),
                         "consonant_err": round(e["consonant_err"], 3),
                         "explanation": e["explanation"]})

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(R / "table.csv", index=False)

    # sanity checks per acceptance criteria
    ident = [e for k in results for e in results[k]["entries"] if e["kind"] == "identical"]
    # monotonic WITHIN each controlled family (stretch/noise/cut/pitch
    # families measure severity on different scales; cross-family scores
    # are not comparable, only the within-family trend is)
    mono_ok = True
    for k in results:
        fam = {}
        for e in results[k]["entries"]:
            if e["kind"] == "perturbation":
                f = e["label"].split("_")[0]
                fam.setdefault(f, []).append((e["severity"], e["score"]))
        for f, sev_score in fam.items():
            sev_score.sort()
            sco = [s for _, s in sev_score]
            diffs = [sco[i] - sco[i + 1] for i in range(len(sco) - 1)]
            # ±2.5 % = empirical frame-noise floor of the jitter-averaged DTW
            # cost on ~50-frame single-word clips (see DECISIONS.md #10)
            pass_m = sum(1 for x in diffs if x > -2.5) / max(1, len(diffs))
            if pass_m < 0.9:
                mono_ok = False
                print(f"[p1-eval] mono FAIL word={k} family={f}: "
                      f"{[(round(s, 2), round(c, 1)) for s, c in sev_score]}")
    summary = {
        "alpha": P1.ALPHA,
        "identical_mean": float(np.mean([e["score"] for e in ident])) if ident else None,
        "identical_min": float(np.min([e["score"] for e in ident])) if ident else None,
        "monotonic_ok": mono_ok,
        "n_pairs": len(ident),
        "n_words": len(results),
    }
    summary["cross_word_max"] = float(max(cross_scores)) if cross_scores else None
    summary["cross_word_mean"] = float(np.mean(cross_scores)) if cross_scores else None
    (R / "summary.json").write_text(json.dumps(summary, indent=2),
                                    encoding="utf-8")
    (R / "results.json").write_text(json.dumps(results, ensure_ascii=False,
                                               indent=2), encoding="utf-8")
    return summary


_CACHE = {}
_P1_PKLCACHE = CACHE / "p1_pair_cache.pkl"
if _P1_PKLCACHE.exists():
    try:
        _CACHE = pickle.loads(_P1_PKLCACHE.read_bytes())
    except Exception:
        _CACHE = {}


def _repair_formants(d, y_ref, y_stu):
    """Old cache entries predate the Praat-call fix (F1/F2 were 0)."""
    if d.get("F1_ref", 0.0) or d.get("F1_stu", 0.0):
        return
    try:
        pa = LA.formants_praat(y_ref)
        pb = LA.formants_praat(y_stu)
        if pa and pb:
            (f1r, f2r), _ = pa
            (f1s, f2s), _ = pb
            d["F1_ref"] = float(np.median(f1r)) if len(f1r) else 0.0
            d["F2_ref"] = float(np.median(f2r)) if len(f2r) else 0.0
            d["F1_stu"] = float(np.median(f1s)) if len(f1s) else 0.0
            d["F2_stu"] = float(np.median(f2s)) if len(f2s) else 0.0
    except Exception:
        pass


def compare_or_cache(key, y_ref, y_stu, kind, label):
    ck = (key, kind, label, hashlib.md5(y_ref.tobytes()).hexdigest()[:8],
          hashlib.md5(y_stu.tobytes()).hexdigest()[:8])
    if ck in _CACHE:
        out = _CACHE[ck]
        _repair_formants(out[0], y_ref, y_stu)
        return out
    d, ref_feat, stu_feat = P1.analyse_pair(y_ref, y_stu)
    _CACHE[ck] = (d, ref_feat, stu_feat)
    try:
        _P1_PKLCACHE.write_bytes(pickle.dumps(_CACHE))
    except Exception:
        pass
    return _CACHE[ck]


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_word_figure(key, deva, y_ref, y_stu, d, ref_feat, stu_feat, outdir: Path):
    fig = plt.figure(figsize=(18, 12), dpi=110)
    fig.suptitle(f"Reference vs student — {deva} (‘krishna’) ", fontsize=15)

    ax = fig.add_subplot(3, 2, 1)
    tr = np.arange(len(y_ref)) / LA.SR
    ts = np.arange(len(y_stu)) / LA.SR
    ax.plot(tr, y_ref, alpha=0.8, label="reference")
    ax.plot(ts, y_stu, alpha=0.8, label="student")
    ax.set_title("a. Waveform overlay"); ax.legend(); ax.set_xlabel("s")

    ax = fig.add_subplot(3, 2, 2)
    import librosa
    S1 = librosa.amplitude_to_db(np.abs(librosa.stft(y_ref, n_fft=512)), ref=np.max)
    S2 = librosa.amplitude_to_db(np.abs(librosa.stft(y_stu, n_fft=512)), ref=np.max)
    ax.imshow(S1, origin="lower", aspect="auto", alpha=0.9, cmap="magma")
    ax.imshow(S2, origin="lower", aspect="auto", alpha=0.45, cmap="viridis")
    ax.set_title("b. Spectrogram overlay (ref = magma, student = viridis)")

    # MFCC heatmaps
    for i, (y, tag) in enumerate([("ref", "reference"), ("stu", "student")]):
        ax = fig.add_subplot(3, 4, 5 + i)
        M = ref_feat["mfcc39"][:, :13] if y == "ref" else stu_feat["mfcc39"][:, :13]
        ax.imshow(M, origin="lower", aspect="auto", cmap="coolwarm")
        ax.set_title(f"c{i+1}. MFCC heatmap ({tag})", fontsize=9)
    ax = fig.add_subplot(3, 4, 7)
    mr = ref_feat["mfcc39"][:, :13]
    ms = P1.resample_frames(stu_feat["mfcc39"][:, :13], len(mr))
    ax.imshow(np.abs(mr - ms), origin="lower", aspect="auto", cmap="hot")
    ax.set_title("c3. |MFCC diff|", fontsize=9)

    # DTW path
    ax = fig.add_subplot(3, 4, 8)
    path = np.array(d["dtw_path"])
    m1 = np.abs(stu_feat["mfcc39"][:, :13] - P1.resample_frames(
        ref_feat["mfcc39"][:, :13], len(stu_feat["mfcc39"]))).max()
    if len(path):
        ax.plot(path[:, 1], path[:, 0], "b.", ms=1, label="DTW aligned pairs")
        ax.set_xlabel("student MFCC frame"); ax.set_ylabel("reference MFCC frame")
        ax.set_title("d. DTW alignment path")
    ax.legend(loc="upper left", fontsize=8)

    # per-segment errors
    ax = fig.add_subplot(3, 2, 4)
    segs = [s for s in d["segments"]]
    xs = [f"{i}\n{'V' if s['voiced'] else 'C'}" for i, s in enumerate(segs)]
    colors = ["tab:red" if s["voiced"] else "tab:blue" for s in segs]
    ax.bar(xs, [s["err"] for s in segs], color=colors)
    ax.set_title("e. Per-segment error (V=vowel, C=consonant)")
    ax.axhline(np.mean([s["err"] for s in d["segments"]]), ls="--",
               c="k", lw=1, label="mean")
    ax.legend(fontsize=8)

    # duration bar
    ax = fig.add_subplot(3, 2, 5)
    dr = d["dur_ratio"]
    ax.bar(["reference", "student"], [len(y_ref) / LA.SR, len(y_stu) / LA.SR])
    ax.set_title(f"f. Duration ratio = {dr:.3f}")

    ax = fig.add_subplot(3, 2, 6)
    ax.axis("off")
    txt = (f"DTW MFCC cost {d['d_mfcc']:.4f}\n"
           f"spectral DTW {d['d_spec']:.4f}\n"
           f"F0 pattern distance {d['d_f0']:.3f}\n"
           f"duration ratio {d['dur_ratio']:.3f}\n"
           f"F1 ref {d.get('F1_ref', 0):.0f} Hz | student {d.get('F1_stu', 0):.0f} Hz\n"
           f"F2 ref {d.get('F2_ref', 0):.0f} Hz | student {d.get('F2_stu', 0):.0f} Hz\n"
           f"centroid ref {d['mean_centroid_ref']:.0f} | student {d['mean_centroid_stu']:.0f}")
    ax.text(0.02, 0.95, txt, va="top", fontsize=11, family="monospace")

    fig.tight_layout()
    p = outdir / f"P1_{key}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def make_monotonic_figure(key, sevs, scores):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=110)
    ax.plot(sevs, scores, "o-", color="crimson")
    ax.set_xlabel("controlled perturbation severity (0 = identical)")
    ax.set_ylabel("Pronunciation similarity (%)")
    ax.set_title("Similarity decreases as the student’s error grows")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = OUT / "results" / "P1" / f"P1_monotonic_{key}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
