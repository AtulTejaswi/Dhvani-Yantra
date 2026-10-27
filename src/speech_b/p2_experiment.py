"""One-off feature-recipe experiment for P2 (documented in DECISIONS.md).

Compares feature definitions + state counts using leave-one-voice-out CV, then
checks the winner on the held-out test voices. Runs quickly (~4 configs).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OUT / "src"))
sys.path.insert(0, str(OUT / "src" / "speech_b"))
import lib_audio as LA
verses = None


def feats_A(path):          # current: 39-dim, CMVN applied to everything
    return LA.cmvn(LA.mfcc_full(LA.load_audio(path)))


def feats_B(path):          # static CMVN, then deltas of the CMVN'd statics
    m = LA.mfcc_full(LA.load_audio(path), with_deltas=False)
    m = LA.cmvn(m)
    d1 = np.gradient(m, axis=0)
    d2 = np.gradient(d1, axis=0)
    return np.hstack([m, d1, d2])


def feats_C(path):          # static only
    return LA.cmvn(LA.mfcc_full(LA.load_audio(path), with_deltas=False))


def run():
    from p2_train_eval import load_meta, fit_for, clean
    meta = load_meta()
    vs = meta["verses"]
    train_v = [v.rsplit("-", 1)[1].replace("Neural", "")
               for v in meta["voices"]["train"]]
    cache = {}
    def cached(fn, key, path):
        k = (key, path)
        if k not in cache:
            cache[k] = clean(fn(path))
        return cache[k]

    results = {}
    for name, fn in (("B_cmvn_then_delta", feats_B),
                     ("C_static_only", feats_C),
                     ("A_current", feats_A)):
        for ns in (6, 8):
            accs = []
            for hold in train_v:
                tr = {v: [] for v in vs}
                for c in meta["clips"]:
                    ct = c["voice"].rsplit("-", 1)[1].replace("Neural", "")
                    if c["split"] == "train" and ct != hold:
                        tr[c["verse"]].append(cached(fn, name, c["path"]))
                models = fit_for(vs, tr, ns, 7)
                preds, gold = [], []
                for c in meta["clips"]:
                    ct = c["voice"].rsplit("-", 1)[1].replace("Neural", "")
                    if c["split"] == "train" and ct == hold:
                        m = cached(fn, name, c["path"])
                        s = {v: models[v].score(m) for v in vs}
                        preds.append(max(s, key=s.get))
                        gold.append(c["verse"])
                acc = float(np.mean([p == g for p, g in zip(preds, gold)]))
                accs.append(acc)
            cv = float(np.mean(accs))
            results[f"{name}_state{ns}"] = {"cv": cv, "folds": accs}
            print(f"[exp] {name} states={ns}: CV {cv:.3f} {accs}")
    (OUT / "cache" / "p2" / "feature_experiment.json").write_text(
        json.dumps(results, indent=2))


if __name__ == "__main__":
    run()
