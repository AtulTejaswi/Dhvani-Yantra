"""P2: MFCC -> CMVN -> left-to-right GaussianHMM per verse; DTW baseline.

State count is tuned by 3-fold cross-validation *within the training voice*
(hold out one speech-rate), then final models are retrained on all training
clips and evaluated once on the held-out voice. No test leakage.
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OUT / "src"))
import lib_audio as LA

R = OUT / "results" / "P2"
R.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "cache" / "p2"
CACHE.mkdir(parents=True, exist_ok=True)
SEED = 7
SR = LA.SR


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def feats(path: str) -> np.ndarray:
    return LA.cmvn(LA.mfcc_full(LA.load_audio(path)))


def clean(m: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.nan_to_num(m, nan=0.0,
                                             posinf=0.0, neginf=0.0),
                                dtype=np.float64)


def load_meta():
    return json.loads((OUT / "data" / "gita" / "p2_dataset.json")
                      .read_text("utf-8"))


def augment_y(rng: np.random.Generator, y: np.ndarray) -> np.ndarray:
    """Sep. noise / tempo / pitch shifts, +-10% or +-1 semi-tone. Train only."""
    import librosa
    kind = rng.integers(0, 3)
    if kind == 0:      # additive noise
        snr = rng.uniform(10, 30)
        n = y.std() / (10 ** (snr / 20)) * rng.normal(size=y.shape)
        return (y + n).astype(np.float32)
    if kind == 1:      # tempo
        return librosa.effects.time_stretch(y, rate=rng.uniform(0.9, 1.1))
    return librosa.effects.pitch_shift(y, sr=SR, n_steps=rng.uniform(-1.0, 1.0))


def all_features(meta, cache=True) -> dict[str, np.ndarray]:
    fp = CACHE / "feats.pkl"
    if cache and fp.exists():
        return pickle.loads(fp.read_bytes())
    X = {}
    for c in meta["clips"]:
        if c["split"] == "ignore":
            continue
        X[c["path"]] = clean(feats(c["path"]))
    if cache:
        fp.write_bytes(pickle.dumps(X))
    return X


def new_hmm(n_states: int, seed: int):
    from hmmlearn.hmm import GaussianHMM
    k = n_states
    h = GaussianHMM(n_components=k, covariance_type="diag", n_iter=80,
                    tol=1e-3, min_covar=0.02, init_params="stmc",
                    params="stmc", random_state=seed, algorithm="viterbi")
    return h


def fit_for(verses: list[str], seqs_of: dict[str, list[np.ndarray]],
            n_states: int, seed: int) -> dict:
    models = {}
    for v in verses:
        seqs = [np.asarray(s, dtype=np.float64) for s in seqs_of[v]]
        lengths = [len(s) for s in seqs]
        Xcat = np.vstack(seqs)
        h = new_hmm(n_states, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            h.fit(Xcat, lengths=lengths)
        models[v] = h
    return models


def loader_y(path: str) -> np.ndarray:
    return LA.load_audio(path)


def make_augmented_train(meta) -> dict[str, list[np.ndarray]]:
    """verse -> MFCC sequences for all train clips + 2 seeded augmentations."""
    out = {v: [] for v in meta["verses"]}
    rng = np.random.default_rng(SEED + 1)
    for c in meta["clips"]:
        if c["split"] != "train":
            continue
        y = loader_y(c["path"])
        base = clean(feats(c["path"]))
        out[c["verse"]].append(base)
        for _ in range(2):
            ya = augment_y(rng, y)
            mf = LA.mfcc_full(ya)
            mf = LA.cmvn(mf)
            out[c["verse"]].append(clean(mf))
    return out


def train_models(force: bool = False):
    meta = load_meta()
    verses = meta["verses"]
    mpk = CACHE / "hmm_models.pkl"
    if mpk.exists() and not force:
        cm = json.loads((CACHE / "hmm_meta.json").read_text())
        models = pickle.loads(mpk.read_bytes())
        print(f"[p2] reusing cached models (n_states={cm['n_states']})")
        return models, meta, cm["n_states"], cm["grid"]

    train_voices = [v.rsplit("-", 1)[1].replace("Neural", "")
                    for v in meta["voices"]["train"]]
    grid_rows = []
    best = None
    for n_states in (4, 6, 8, 10, 12):
        # Leave-one-voice-out CV: genuinely cross-SPEAKER generalisation.
        accs = []
        for hold in train_voices:
            training = {v: [] for v in verses}
            for c in meta["clips"]:
                ct = c["voice"].rsplit("-", 1)[1].replace("Neural", "")
                if c["split"] == "train" and ct != hold:
                    training[c["verse"]].append(clean(feats(c["path"])))
            models = fit_for(verses, training, n_states, SEED)
            gold, preds = [], []
            for c in meta["clips"]:
                ct = c["voice"].rsplit("-", 1)[1].replace("Neural", "")
                if c["split"] == "train" and ct == hold:
                    m = clean(feats(c["path"]))
                    scores = {v: models[v].score(m) for v in verses}
                    preds.append(max(scores, key=scores.get))
                    gold.append(c["verse"])
            accs.append(float(np.mean([p == g for p, g in zip(preds, gold)])))
        mean_cv = float(np.mean(accs))
        grid_rows.append({"n_states": n_states, "cv_acc": mean_cv,
                          "folds": accs})
        print(f"[p2] n_states={n_states}, leave-one-voice-out CV acc = {mean_cv:.3f}")
        if best is None or mean_cv > best[0]:
            best = (mean_cv, n_states)
    best_cv, best_n_states = best

    # final models: full training set (with seeded augmentations) at best states
    train_data = make_augmented_train(meta)
    models = fit_for(verses, train_data, best_n_states, SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pass
    (CACHE / "hmm_models.pkl").write_bytes(pickle.dumps(models))
    (CACHE / "hmm_meta.json").write_text(json.dumps(
        {"n_states": best_n_states, "cv_acc": best_cv, "grid": grid_rows},
        indent=2))
    print(f"[p2] best n_states = {best_n_states} (cv {best_cv:.3f})")
    return models, meta, best_n_states, grid_rows


def run_tests(models, meta, n_states, grid_rows):
    test = [c for c in meta["clips"] if c["split"] == "test"]
    train = [c for c in meta["clips"] if c["split"] == "train"]
    X = {c["path"]: clean(load_feats(c["path"])) for c in meta["clips"]}
    verses = meta["verses"]
    idx = {v: i for i, v in enumerate(verses)}
    conf = np.zeros((len(verses), len(verses)), dtype=int)
    rows = []
    for c in test:
        m = X[c["path"]]
        scores = {v: float(models[v].score(m)) for v in verses}
        raw = np.array([scores[v] for v in verses])
        probs = np.exp(raw - raw.max()); probs /= probs.sum()
        order = np.argsort(-probs)
        pred = verses[int(order[0])]
        gold = c["verse"]
        rows.append({"clip": Path(c["path"]).name, "voice": c["voice"],
                     "gold": gold, "pred": pred, "correct": pred == gold,
                     "top3": ";".join(verses[i] for i in order[:3]),
                     "conf": float(probs[int(order[0])])})
        conf[idx[gold], idx[pred]] += 1
    acc = float(np.mean([r["correct"] for r in rows]))

    dtw_acc = dtw_baseline(test, train, X)

    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7), dpi=110)
    im = ax.imshow(conf, cmap="Blues")
    ax.set_xticks(range(len(verses))); ax.set_yticks(range(len(verses)))
    ax.set_xticklabels(verses, rotation=90, fontsize=8)
    ax.set_yticklabels(verses, fontsize=8)
    for i in range(len(verses)):
        for j in range(len(verses)):
            if conf[i, j]:
                ax.text(j, i, str(conf[i, j]), ha="center", va="center",
                        color="red", fontsize=9)
    ax.set_xlabel("Predicted verse"); ax.set_ylabel("True verse")
    ax.set_title(f"MFCC HMM ({n_states} states), held-out-voice test\n"
                 f"accuracy {acc * 100:.1f}% | DTW baseline {dtw_acc * 100:.1f}%")
    fig.tight_layout()
    fig.savefig(R / "p2_confusion.png"); plt.close(fig)

    pd.DataFrame(rows).to_csv(R / "p2_test_results.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(R / "p2_state_cv.csv", index=False)
    summary = {"test_acc": acc, "n_test": len(test), "n_train": len(train),
               "n_states": n_states, "dtw_acc": dtw_acc,
               "n_verses": len(verses), "confusion": conf.tolist(),
               "synthetic_data": True, "n_aug_per_train_clip": 2,
               "test_voices": meta["voices"]["test"],
               "train_voices": meta["voices"]["train"],
               "augment": "gaussian noise / tempo 0.9-1.1 / pitch +-1 st"}
    (R / "p2_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_feats(path: str) -> np.ndarray:
    return feats(path)


def dtw_baseline(test, train, X):
    import librosa
    from collections import Counter
    preds = []
    for c in test:
        q = X[c["path"]].T  # (39, T)
        scores = []
        for t in train:
            D = librosa.sequence.dtw(
                q, X[t["path"]].T, metric="cosine", backtrack=False)
            scores.append((float(D[-1, -1] / (D.shape[0] + D.shape[1])),
                           t["verse"]))
        scores.sort()
        preds.append(Counter([v for _, v in scores[:5]]).most_common(1)[0][0])
    return float(np.mean([p == c["verse"] for p, c in zip(preds, test)]))


def main():
    models, meta, n_states, grid = train_models()
    summary = run_tests(models, meta, n_states, grid)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "confusion"}, indent=2))


if __name__ == "__main__":
    main()

