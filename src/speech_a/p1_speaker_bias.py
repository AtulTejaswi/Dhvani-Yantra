"""P1 speaker-bias probe: how much does the similarity score punish the
SPEAKER rather than the WORD?

The references are synthetic (Edge-TTS hi-IN Neural voices), and every
"student" take in the calibrated corpus is synthetic too, so no number here
can validate real-voice accuracy. What CAN be measured with the existing
assets is the metric's sensitivity to speaker identity:

  identical      ref_W  vs ref_W        same speaker, same word  (upper bound)
  alt-voice      ref_W  vs ref_W_alt    DIFFERENT TTS voice, SAME word/text
  tts-student    ref_W  vs stud_W       different (en-IN) voice + romanised text
  cross-word     ref_W  vs ref_W2       different word (the mismatch baseline)

If the alt-voice score (word identical, speaker changed) sits near or below
the cross-word baseline (word changed), the metric is dominated by speaker
identity, and a real human voice — far further from a Neural TTS voice than
one Neural voice is from another — should be expected to score below the
identical-pair numbers regardless of pronunciation quality. The alt-voice
drop is therefore a LOWER BOUND of the speaker penalty for real voices.

Output: results/P1/speaker_bias.json + a printed table.
Run:    python src/speech_a/p1_speaker_bias.py        (from project_out/)
No network access and no audio generation: only files already in data/audio.
Formants (Praat) are skipped — they are diagnostic-only and do not enter the
score — which keeps the run to ~1-2 minutes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OUT / "src"))
sys.path.insert(0, str(OUT / "src" / "speech_a"))
import lib_audio as LA      # noqa: E402
import p1_core as P1        # noqa: E402

R = OUT / "results" / "P1"


def _alpha() -> float:
    try:
        return float(json.loads((OUT / "cache" / "p1_alpha.json")
                                .read_text("utf-8"))["alpha"])
    except Exception:
        return P1.ALPHA


def score_pair(y_a, y_b) -> float:
    d, _, _ = P1.analyse_pair(y_a, y_b, do_praat=False)
    return float(100 * np.exp(-P1.ALPHA * P1.weighted_distance(d)))


def run() -> dict:
    meta = json.loads((OUT / "data" / "audio" / "p1_dataset.json")
                      .read_text("utf-8"))
    P1.ALPHA = _alpha()
    words = list(meta["words"])
    rows = []
    for w in words:
        ref_p = meta["refs"][w]["path"]
        y_ref = LA.load_audio(ref_p)
        alt_p = OUT / "data" / "audio" / f"ref_{w}_alt.wav"
        tts = next((e for e in meta["students"][w]
                    if e["kind"] == "tts-student"), None)
        cross = [meta["refs"][k]["path"] for k in words if k != w]
        rows.append({"word": w, "kind": "identical", "vs": w,
                     "score": score_pair(y_ref, y_ref)})
        if alt_p.exists():
            rows.append({"word": w, "kind": "alt-voice-same-word", "vs": w,
                         "score": score_pair(y_ref, LA.load_audio(str(alt_p)))})
        if tts is not None:
            rows.append({"word": w, "kind": "tts-student-same-word",
                         "vs": tts["label"],
                         "score": score_pair(y_ref, LA.load_audio(tts["path"]))})
        for cp in cross:
            rows.append({"word": w, "kind": "cross-word",
                         "vs": Path(cp).stem.replace("ref_", ""),
                         "score": score_pair(y_ref, LA.load_audio(cp))})
        print(f"[p1-speaker-bias] {w}: done")

    def mean(kind):
        v = [r["score"] for r in rows if r["kind"] == kind]
        return float(np.mean(v)) if v else None

    summary = {
        "alpha": P1.ALPHA,
        "n_pairs": len(rows),
        "identical_mean": mean("identical"),
        "alt_voice_same_word_mean": mean("alt-voice-same-word"),
        "tts_student_same_word_mean": mean("tts-student-same-word"),
        "cross_word_mean": mean("cross-word"),
        "note": ("All audio is synthetic (Edge-TTS). alt-voice pairs change "
                 "only the speaker (same word, same text, same engine) so "
                 "their drop below 'identical' isolates speaker cost; the "
                 "tts-student drop adds a different TTS voice + romanised "
                 "text. Human voices are far further from the synthetic "
                 "reference than two Neural voices are from each other, so "
                 "these numbers are a lower bound of the speaker penalty a "
                 "real recording faces. They do NOT validate real-voice "
                 "accuracy."),
    }
    R.mkdir(parents=True, exist_ok=True)
    (R / "speaker_bias.json").write_text(
        json.dumps({"summary": summary, "pairs": rows}, indent=2), "utf-8")

    print(f"\n{'word':8s} {'kind':26s} {'vs':10s} {'score %':>8s}")
    for r in rows:
        print(f"{r['word']:8s} {r['kind']:26s} {str(r['vs'])[:10]:10s} "
              f"{r['score']:8.1f}")
    print("\n== summary ==")
    for k in ("identical_mean", "alt_voice_same_word_mean",
              "tts_student_same_word_mean", "cross_word_mean"):
        v = summary[k]
        print(f"{k:30s} {v:6.1f}" if v is not None else f"{k:30s}   n/a")
    bias = None
    if summary["alt_voice_same_word_mean"] is not None:
        bias = summary["cross_word_mean"] - summary["alt_voice_same_word_mean"]
        print(f"\nspeaker-vs-word bias margin: {bias:+.1f} pts "
              "(negative => speaker change costs MORE than a wrong word)")
    summary["bias_margin_cross_minus_alt"] = bias
    (R / "speaker_bias.json").write_text(
        json.dumps({"summary": summary, "pairs": rows}, indent=2), "utf-8")
    return summary


if __name__ == "__main__":
    run()
