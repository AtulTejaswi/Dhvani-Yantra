"""Unified LIVE demo app: [Pronunciation] [Gita Verse] [Video].

Run:  streamlit run app.py   (from project_out/)
Everything works offline from cached models/data; uploaded clips are handled
live with the very same code paths as the batch pipelines.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

OUT_DIR = Path(__file__).resolve().parent
for _sub in ("src", r"src\speech_a", r"src\speech_b", r"src\video"):
    sys.path.insert(0, str(OUT_DIR / _sub))
sys.path.insert(0, str(OUT_DIR))

import lib_audio as LA            # noqa: E402
OK = lambda p: Path(p).exists()   # noqa: E731

st.set_page_config(page_title="Sanskrit Speech + Video Project",
                   page_icon="🎧", layout="wide")

CACHE_DIR = OUT_DIR / "cache" / "app"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
P1 = OUT_DIR / "cache" / "p1_alpha.json"

# ---------------------------------------------------------------------------
@st.cache_resource
def load_p1_meta():
    p = OUT_DIR / "data" / "audio" / "p1_dataset.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else None


@st.cache_resource
def load_p1_eval():
    p = OUT_DIR / "results" / "P1" / "results.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else None


@st.cache_resource
def load_p2():
    meta = OUT_DIR / "data" / "gita" / "p2_dataset.json"
    models = OUT_DIR / "cache" / "p2" / "hmm_models.pkl"
    smeta = OUT_DIR / "cache" / "p2" / "hmm_meta.json"
    if not (meta.exists() and models.exists() and smeta.exists()):
        return None
    from p2_train_eval import feats, clean
    import pickle as pk
    models = pk.loads(models.read_bytes())
    sm = json.loads(smeta.read_text("utf-8"))
    # test-time metrics live in the eval summary; merge them over the meta grid
    p2sum = OUT_DIR / "results" / "P2" / "p2_summary.json"
    if p2sum.exists():
        for k in ("test_acc", "dtw_acc", "n_test", "test_voices"):
            try:
                sm[k] = json.loads(p2sum.read_text("utf-8"))[k]
            except Exception:
                pass
    return {"meta": json.loads(meta.read_text("utf-8")),
            "models": {k: v for k, v in models.items()},
            "smeta": sm}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# helpers for live P1 analysis
@st.cache_data
def p1_analyse(ref_path: str, stu_path: str, r_m: float, s_m: float):
    import p1_core as P1CORE
    y_ref = LA.load_audio(ref_path)
    y_stu = LA.load_audio(stu_path)
    d, ref_feat, stu_feat = P1CORE.analyse_pair(y_ref, y_stu)
    # ALPHA comes from the last calibration run (cache/p1_alpha.json)
    try:
        P1CORE.ALPHA = float(json.loads(P1.read_text("utf-8"))["alpha"])
    except Exception:
        pass
    score = 100 * np.exp(-P1CORE.ALPHA * P1CORE.weighted_distance(d))
    v = [s for s in d["segments"] if s["voiced"]]
    c = [s for s in d["segments"] if not s["voiced"]]
    v_err = float(np.mean([s["err"] for s in v])) if v else 0
    c_err = float(np.mean([s["err"] for s in c])) if c else 0
    from p1_core import verdict_text  # type: ignore
    return (d, ref_feat, stu_feat, score, v_err, c_err,
            verdict_text(score, d, v_err, c_err))


def p1_audio_bytes(bts: bytes | None, suffix=".wav") -> Path | None:
    if not bts:
        return None
    h = hashlib.md5(bts).hexdigest()
    p = CACHE_DIR / f"upl_{h}{suffix}"
    if not p.exists():
        p.write_bytes(bts)
    return p


# ===========================================================================
# SIDEBAR
# ===========================================================================
st.sidebar.title("Sanskrit Speech + Video Projects")
st.sidebar.caption("Pronunciation • Gita Verse ID • Video Processing")
synthetic = False
if P1.exists():
    st.sidebar.info("P1 α calibrated ✔ | models cached ✔")
st.sidebar.caption("Synthetic-data status:")
try:
    kinds = sorted({r.get("kind", "synthetic") for r in load_p1_meta()["refs"].values()})
    kinds = [k.replace("-", " ") for k in kinds]
    s_p1 = "SYNTHETIC (edge-tts)" if any("tts" in k for k in kinds) else "mixed"
    st.sidebar.caption(f"P1 references: {', '.join(kinds)}")
    st.sidebar.caption("P2 recitations: SYNTHETIC (edge-tts)")
    st.sidebar.caption("P3 videos: CC-BY Blender open movies (real)")
except Exception:
    pass

tab1, tab2, tab3 = st.tabs(
    ["🗣️ Pronunciation Assessment", "🕉️ Gita Verse Recognition", "🎬 Video Processing"])

# ===========================================================================
# TAB 1 — P1
# ===========================================================================
with tab1:
    st.header("Speech A — Pronunciation Assessment (MFCC + spectral features, DTW)")
    st.caption("Reference vs student take of a Sanskrit word; similarity %, segment-"
               "level diagnosis and full plots. References are **synthetic** "
               "(Edge TTS Devanagari) where marked.")
    meta1 = load_p1_meta()
    if not meta1:
        st.error("P1 dataset missing - run project_out/src/speech_a/p1_build_data.py")
    else:
        c0, c1, c2 = st.columns([1, 2, 2])
        word = c0.selectbox("Reference word",
                             list(meta1["words"]),
                             format_func=lambda k: f"{k}  ({meta1['words'][k]['deva']})")
        entries = meta1["students"][word]
        labels = [f"{i+1}. {e['label']} ({e['kind']})" for i, e in enumerate(entries)]
        pick = c1.selectbox("Student take (or upload/record below)", labels)
        up = c2.file_uploader("...or upload your own pronunciation (wav/mp3)",
                              type=["wav", "mp3", "ogg", "flac", "m4a"])
        rec = c2.audio_input("...or record now (microphone)")
        stu_btn = st.button("🔎 Compare & score", type="primary")
        res_box = st.container()

        def run_compare(stu_path: Path, stu_label: str):
            with res_box:
                with st.spinner("aligning (DTW) + scoring..."):
                    ref_path = meta1["refs"][word]["path"]
                    d, ref_feat, stu_feat, score, v_err, c_err, verdict = \
                        p1_analyse(ref_path, str(stu_path), 0, 0)
                st.markdown(
                    f"## Pronunciation similarity: **{score:.0f}%**")
                st.markdown(f"**Verdict**: {verdict}")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("MFCC DTW cost", f"{d['d_mfcc']:.4f}")
                m2.metric("Duration ratio",
                          f"{d['dur_ratio']:.2f}x",
                          f"{(d['dur_ratio']-1)*100:+.0f}%")
                m3.metric("Vowel-segment error", f"{v_err:.3f}")
                m4.metric("Consonant-segment error", f"{c_err:.3f}")
                tbl = st.columns(3)
                tbl[0].markdown(
                    f"**F1/F2 (ref)**  \n{d.get('F1_ref',0):.0f} Hz / {d.get('F2_ref',0):.0f} Hz")
                tbl[1].markdown(
                    f"**F1/F2 (student)**  \n{d.get('F1_stu',0):.0f} Hz / {d.get('F2_stu',0):.0f} Hz")
                tbl[2].markdown(
                    f"**Spectral centroid**  \nref {d['mean_centroid_ref']:.0f} Hz | "
                    f"student {d['mean_centroid_stu']:.0f} Hz")
                explain = []
                if v_err > c_err * 1.25:
                    explain.append("🟠 vowel segments drift the most "
                                   "(formant/energy shift)")
                if c_err > v_err * 1.25:
                    explain.append("🔵 consonant clusters are the weaker part")
                if d["dur_ratio"] > 1.08:
                    explain.append("⏱️ spoken slower than the reference")
                elif d["dur_ratio"] < 0.92:
                    explain.append("⏱️ spoken faster than the reference")
                st.markdown("**Plain-language explanation**: " +
                            ("; ".join(explain) or "all segments close to the reference."))
                st.markdown(
                    "#### Segment-by-segment error bar (V=vowel, C=consonant)")
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(9, 2.4))
                segs = d["segments"]
                ax.bar(range(len(segs)), [s["err"] for s in segs],
                       color=["tab:red" if s["voiced"] else "tab:blue"
                              for s in segs])
                ax.set_xticks(range(len(segs)))
                ax.set_xticklabels(["V" if s["voiced"] else "C" for s in segs])
                ax.set_ylabel("frame err")
                st.pyplot(fig, clear_figure=True)
                # reference waveform overlays + MFCC heatmap from cached chart
                png = OUT_DIR / "results" / "P1" / f"P1_{word}.png"
                if png.exists():
                    st.image(str(png), caption="Reference-vs-student analysis chart")
                c1d, c2d = st.columns(2)
                c1d.audio(str(ref_path))
                if str(stu_path) != ref_path:
                    c2d.audio(str(stu_path))

        if stu_btn:
            src = None
            if up is not None:
                src = p1_audio_bytes(up.getvalue(), suffix=Path(up.name).suffix or ".wav")
            elif rec is not None:
                src = p1_audio_bytes(rec.getvalue())
            if src is None:
                src = Path(entries[int(pick.split(".")[0]) - 1]["path"])
            run_compare(src, pick)
        else:
            res_box.info("Choose a word + student take, then press **Compare & score**.")

        st.divider()
        st.markdown("### All results (validation table)")
        ev = load_p1_eval()
        if ev:
            import pandas as pd
            rows = []
            for k in ev:
                for e in ev[k]["entries"][:4]:
                    rows.append({"word": ev[k]["roman"],
                                 "student": e["label"],
                                 "kind": e["kind"],
                                 "similarity_%": round(e["score"], 1)})
            st.dataframe(pd.DataFrame(rows), width="stretch")
            ch1, ch2 = st.columns(2)
            mono = OUT_DIR / "results" / "P1" / "P1_monotonic_krishna.png"
            if mono.exists():
                ch1.image(str(mono), caption="Score vs perturbation severity")
            tw = OUT_DIR / "results" / "P1" / "P1_dharma.png"
            if tw.exists():
                ch2.image(str(tw), caption="Full analysis chart (another word)")

# ===========================================================================
# TAB 2 — P2
# ===========================================================================
with tab2:
    st.header("Speech B — HMM recognition of Bhagavad Gita verses (MFCC)")
    st.caption("One left-to-right Gaussian HMM per verse; argmax log-likelihood → "
               "verse ID. All recitations are **synthetic** (Edge-TTS, "
               "4 voices × 3 rates; 2 train voices / 2 fully-unseen test voices). "
               "Unseen-voice test accuracy is shown at the bottom.")
    p2 = load_p2()
    if not p2:
        st.error("P2 models missing - run project_out/src/speech_b/p2_train_eval.py")
    else:
        meta2 = p2["meta"]
        clips = [c for c in meta2["clips"] if c["split"] in ("test", "train")]
        c0, c1 = st.columns([2, 2])
        opts = [f"{c['verse']} | {'test' if c['split']=='test' else 'train'} voice | {c['rate']}"
                for c in clips]
        pick2 = c0.selectbox("Sample clip", opts,
                             index=next((i for i, o in enumerate(opts)
                                         if "9.26" in o), 0))
        up2 = c1.file_uploader("...or upload a verse recording (wav/mp3)",
                               type=["wav", "mp3", "ogg", "flac", "m4a"])
        rec2 = c1.audio_input("...or record now")
        btn2 = st.button("🕉️ Identify verse", type="primary")
        box2 = st.container()
        if btn2:
            src = None
            if up2 is not None:
                src = p1_audio_bytes(up2.getvalue(),
                                     suffix=Path(up2.name).suffix or ".wav")
            elif rec2 is not None:
                src = p1_audio_bytes(rec2.getvalue())
            if src is None:
                src = Path(clips[opts.index(pick2)]["path"])
            with box2:
                from p2_train_eval import feats, clean  # type: ignore
                m = clean(feats(str(src)))
                verses = meta2["verses"]
                with st.spinner("scoring against every verse HMM..."):
                    scores = {v: float(p2["models"][v].score(m)) for v in verses}
                raw = np.array([scores[v] for v in verses])
                probs = np.exp(raw - raw.max()); probs /= probs.sum()
                order = np.argsort(-probs)
                top3 = [verses[i] for i in order[:3]]
                gold_gold = None
                for c in clips:
                    try:
                        if Path(c["path"]).resolve() == Path(src).resolve():
                            gold_gold = c["verse"]
                            break
                    except Exception:
                        pass
                corr = "✅" if gold_gold == top3[0] else \
                    ("☑️" if gold_gold == top3[1] else "")
                st.markdown(f"## Verse **{top3[0]}** {corr}")
                st.markdown("**Top-3:** " + " › ".join(
                    f"{v} ({probs[list(verses).index(v)]*100:.1f}%)" for v in top3))
                sm = p2["smeta"]
                st.metric("Held-out-voice test accuracy",
                          f"{sm['test_acc']*100:.1f}% (n={sm['n_test']})")
                st.metric("DTW nearest-neighbour baseline",
                          f"{sm['dtw_acc']*100:.1f}%")
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(9, 3))
                xs = np.arange(len(verses))
                ax.bar(xs, raw - raw.min(), color="seagreen")
                ax.set_xticks(xs); ax.set_xticklabels(verses, rotation=45,
                                                      fontsize=8)
                ax.set_ylabel("log-likelihood (Δ)")
                ax.set_title("HMM log-likelihood per verse")
                st.pyplot(fig, clear_figure=True)
                st.audio(str(src))

        st.divider()
        st.markdown("### Model evidence")
        cm = OUT_DIR / "results" / "P2" / "p2_confusion.png"
        if cm.exists():
            st.image(str(cm), caption="Confusion matrix (held-out voice)")
        g = OUT_DIR / "results" / "P2" / "p2_state_cv.csv"
        if g.exists():
            import pandas as pd
            st.dataframe(pd.read_csv(g), width="stretch")

# ===========================================================================
# TAB 3 — P3
# ===========================================================================
with tab3:
    st.header("Video — important frames • scenes • motion • tracking • summary")
    p = json.loads((OUT_DIR / "cache" / "p3" / "prep.json").read_text()) \
        if (OUT_DIR / "cache" / "p3" / "prep.json").exists() else None
    if not p:
        st.error("P3 outputs missing - run project_out/src/video/p3_run_all.py")
    else:
        st.markdown(f"Source: *movie_30min.mp4* ({p['merged_seconds']/60:.0f} min) — "
                    "concatenated **CC-BY** Blender open movies "
                    "(Tears of Steel + Sintel + Elephants Dream).")
        k = OUT_DIR / "cache" / "p3" / "keyframes.json"
        a1, a2 = st.columns(2)
        if k.exists():
            kf = json.loads(k.read_text())
            top = kf.get("top", [])[:12]
            a1.markdown("### 🖼️ Important frames (top-12 of 40)")
            cols = a1.columns(4)
            for i, t in enumerate(top):
                cols[i % 4].image(t["path"],
                                  caption=f"{t['sec']:.0f}s (score {t['score']:.2f})")
        tl = OUT_DIR / "results" / "P3" / "P3_scene_timeline.png"
        hm = OUT_DIR / "results" / "P3" / "P3_motion_heatmap.png"
        mo = OUT_DIR / "results" / "P3" / "P3_motion_energy_curve.png"
        if tl.exists():
            st.image(str(tl), caption="Scene boundaries timeline (PySceneDetect + histogram)")
        c3, c4 = st.columns(2)
        if hm.exists():
            c3.image(str(hm), caption="Cumulative Farneback optical-flow map")
        if mo.exists():
            c4.image(str(mo), caption="Motion energy per second")
        fb = OUT_DIR / "cache" / "p3" / "motion_flow_clip.mp4"
        if fb.exists():
            c3.video(str(fb))
        tr = OUT_DIR / "cache" / "p3" / "tracked_annotated.mp4"
        tj = OUT_DIR / "results" / "P3" / "P3_trajectories.png"
        c5, c6 = st.columns(2)
        if tr.exists():
            c5.header("🧍 Tracking (YOLOv8n + ByteTrack)")
            c5.video(str(tr))
        if tj.exists():
            c6.header("🧭 ID trajectories")
            c6.image(str(tj))
        su = OUT_DIR / "results" / "P3" / "P3_summary.mp4"
        man = OUT_DIR / "results" / "P3" / "P3_summary_manifest.json"
        if su.exists():
            st.header("🎞️ 60-90 s summary")
            st.video(str(su))
            if man.exists():
                mf = json.loads(man.read_text())
                st.caption(f"{len(mf['picks'])} shots, {mf['summary_seconds']:.0f}s "
                           f"from {mf['source_seconds']:.0f}s → compression x"
                           f"{mf['compression_ratio']:.3f}")
        st.divider()
        st.markdown("### ⚡ Process my own clip (≤60 s)")
        upc = st.file_uploader("Short clip (mp4/avi/mov)", type=["mp4", "avi", "mov", "webm"],
                               key="clipup")
        if upc is not None and st.button("Process my clip", type="primary"):
            tmp = OUT_DIR / "cache" / "p3" / "user_clip.mp4"
            tmp.write_bytes(upc.getvalue())
            from lib_data import ffmpeg_exe
            import subprocess
            subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-t", "60",
                            "-i", str(tmp), "-vf", "fps=10,scale=640:-2",
                            "-an", str(OUT_DIR / "cache" / "p3" / "user_analysis.mp4")],
                           check=True)
            from p3_lite import lite_analyse  # your own clip goes to the same code
            lite_analyse(OUT_DIR / "cache" / "p3" / "user_analysis.mp4",
                         OUT_DIR / "results" / "P3")
            st.success("Done — outputs below")
            for f2 in ("user_keyframes.png", "user_motion.png"):
                ffp = OUT_DIR / "results" / "P3" / f2
                if ffp.exists():
                    st.image(str(ffp))
            if (OUT_DIR / "results" / "P3" / "user_scene_timeline.png").exists():
                st.image(str(OUT_DIR / "results" / "P3" / "user_scene_timeline.png"))

st.caption("Synthetic audio = Edge-TTS voices; all embedding/reference media "
           " licences in data/SOURCES.md.")
