"""P1 core: feature extraction, DTW, similarity scoring, segment-level errors."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(OUT / "src"))
import lib_audio as LA

ALPHA = 0.115          # calibrated in calibrate_alpha(); overwritten at load time
WEIGHTS = {"mfcc": 0.35, "spec": 0.12, "dur": 0.35, "f0": 0.08, "warp": 0.10}
CALIB_PATH = OUT / "cache" / "p1_alpha.json"


# ---------------------------------------------------------------------------
# distances
# ---------------------------------------------------------------------------
def dtw_cost(a: np.ndarray, b: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    """Classic DP-DTW implementation, cosine distance, returns (cost, path).

    More transparent than calling into librosa and cheaper (one pass,
    full path backtrace), and normalized by path length afterwards.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 2.0, []
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    # cosine distance per (i, j): computed lazily
    aa = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    bb = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    SIM = aa @ bb.T
    DIST = np.clip(1.0 - SIM, 0.0, 2.0)
    for i in range(1, n + 1):
        Drow = D[i]
        prev = D[i - 1]
        for j in range(1, m + 1):
            Drow[j] = DIST[i - 1, j - 1] + min(prev[j], prev[j - 1], Drow[j - 1])
    # backtrace
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        if i == 1 and j == 1:
            break
        cands = ((D[i - 1, j - 1], -1, -1),
                 (D[i - 1, j], -1, 0),
                 (D[i, j - 1], 0, -1))
        di, dj = min(cands)[1:]
        i, j = i + di, j + dj
    while j > 1:            # pad the remnant along the student axis
        path.append((0, j - 2)); j -= 1
    while i > 1:            # pad the remnant along the reference axis
        path.append((i - 2, 0)); i -= 1
    path.reverse()
    # mean cosine cost over the (diagonal-heavy) warp path — a true per-frame
    # average, not shrunk by the long (n+m) denominator
    cost = D[n, m] / max(1, len(path))
    return float(cost), path


def _fill_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    ok = np.isfinite(x)
    if ok.sum() < 2:
        x[np.isnan(x)] = 0.0
        return x
    idx = np.arange(len(x))
    x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    return x


def _med5(x: np.ndarray) -> np.ndarray:
    if len(x) < 5:
        return x
    pad = np.pad(x, 2, mode="edge")
    return np.median(np.vstack([pad[i:i + len(x)] for i in range(5)]), axis=0)


def f0_pattern_distance(f_ref: np.ndarray, f_stu: np.ndarray,
                        path: list[tuple[int, int]], n_ref: int, n_stu: int,
                        n_avg: int = 60) -> float:
    """F0-contour distance AFTER warping the student contour onto the
    reference time axis with the MFCC-DTW path (much lower variance than
    resampling both contours independently). Both are median-smoothed,
    z-normalised, resampled to 60 points and compared with mean abs diff.
    """
    f_r = _med5(_fill_nan(f_ref))
    f_s = _med5(_fill_nan(f_stu))
    # map: student frames aligned to each reference frame
    warp = np.full(n_ref, np.nan)
    sums = np.zeros(n_ref)
    cnts = np.zeros(n_ref)
    for a, b in path:
        if 0 <= a < n_ref and 0 <= b < n_stu and np.isfinite(f_s[b]):
            sums[a] += f_s[b]
            cnts[a] += 1
    with np.errstate(invalid="ignore"):
        warp = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    warp = _fill_nan(warp)
    f_rv = np.interp(np.linspace(0, len(f_r) - 1, n_avg),
                     np.arange(len(f_r)), f_r)
    f_sw = np.interp(np.linspace(0, len(warp) - 1, n_avg),
                     np.arange(len(warp)), warp)
    f_rz = (f_rv - f_rv.mean()) / (f_rv.std() + 1e-6)
    f_sz = (f_sw - f_sw.mean()) / (f_sw.std() + 1e-6)
    return float(np.abs(f_rz - f_sz).mean())


def dtw_cost_jitter(a: np.ndarray, b: np.ndarray, shifts=(-4, -2, 0, 2, 4)):
    """Ensemble DTW: average the mean-path cost over small start/end offsets.

    Reduces endpoint-trim noise (the ±0.02 cost wiggle that made severity
    ordering unstable on short single-word clips). The (0,0) alignment is the
    canonical one and its path is what the callers use for plotting.
    """
    vals = []
    for sa in shifts:
        for sb in shifts:
            aa = a[max(0, sa):] if sa else a
            bb = b[max(0, sb):] if sb else b
            if len(aa) > 3 and len(bb) > 3:
                vals.append(dtw_cost(aa, bb)[0])
    if not vals:
        return dtw_cost(a, b)[0]
    return float(np.mean(vals))


def spec_vec(feat_dict: dict, n_reducer: int = 40) -> np.ndarray:
    """Reduce frame-level spectral features to a compact per-utterance sequence
    (z-normalized each, resampled to n_reducer frames) for DTW over spectra."""
    keys = ["centroid", "bandwidth", "rolloff", "flatness", "flux", "zcr"]
    arrs = []
    for k in keys:
        x = feat_dict[k]
        x = (x - x.mean()) / (x.std() + 1e-9)
        idx = np.linspace(0, len(x) - 1, n_reducer)
        arrs.append(np.interp(idx, np.arange(len(x)), x))
    return np.vstack(arrs)  # (6, n)


def resample_frames(x: np.ndarray, n: int) -> np.ndarray:
    if x.size == 0:
        return np.zeros((n, x.shape[1] if x.ndim > 1 else 0))
    idx = np.linspace(0, len(x) - 1, n)
    out = np.zeros((n, x.shape[1] if x.ndim > 1 else 1))
    for j in range(x.shape[1] if x.ndim > 1 else 1):
        xx = x[:, j] if x.ndim > 1 else x
        out[:, j] = np.interp(idx, np.arange(len(xx)), xx)
    return out


# ---------------------------------------------------------------------------
# full comparison
# ---------------------------------------------------------------------------
def analyse_pair(y_ref: np.ndarray, y_stu: np.ndarray, do_praat: bool = True):
    """Full feature-level comparison of a reference and a student take.

    Returns a dict with the DTW paths, normalized costs and everything the
    plots need. `y_ref` and `y_stu` are 16-kHz mono float arrays.
    Both sides are silence-trimmed FIRST so duration/F0 statistics refer to
    the spoken word only, and MFCC side is utterance-CMVN'd so the speaker
    channel does not swamp word identity.
    """
    import librosa
    y_ref = LA.trim_silence(y_ref)
    y_stu = LA.trim_silence(y_stu)
    ref = {}
    stu = {}
    ref["mfcc39"] = LA.cmvn(LA.mfcc_full(y_ref))
    stu["mfcc39"] = LA.cmvn(LA.mfcc_full(y_stu))
    ref["spec"] = LA.spectral_features(y_ref)
    stu["spec"] = LA.spectral_features(y_stu)
    ref["f0"] = LA.f0_track(y_ref)
    stu["f0"] = LA.f0_track(y_stu)

    d = {}
    c_mfcc = dtw_cost_jitter(ref["mfcc39"], stu["mfcc39"])
    path = dtw_cost(ref["mfcc39"], stu["mfcc39"])[1]
    d["dtw_path"] = path
    d["d_mfcc"] = c_mfcc
    # spectral DTW (6-channel compact representations), jitter-averaged
    sv_r, sv_s = spec_vec(ref["spec"]), spec_vec(stu["spec"])
    d["d_spec"] = dtw_cost_jitter(sv_r, sv_s)
    # mean static MFCC delta distance (frame-wise cepstral difference)
    m_ref = ref["mfcc39"][:, :13]
    m_stu = resample_frames(np.atleast_2d(stu["mfcc39"][:, :13]), len(m_ref))
    with np.errstate(divide="ignore", invalid="ignore"):
        d["d_mfcc_frame_mean"] = float(np.nanmean(
            np.abs(m_ref - m_stu) / (np.abs(m_ref).mean(0, keepdims=True) + 1e-6)))
    # duration ratio
    dur_r = len(y_ref) / LA.SR
    dur_s = len(y_stu) / LA.SR
    d["dur_ratio"] = dur_s / (dur_r + 1e-9)
    d["d_dur"] = float(abs(np.log(max(d["dur_ratio"], 1e-3))))
    # F0 pattern distance — DTW-path-aligned student contour vs reference
    # (n_r / n_s also feed the structural warp-deviation measure below)
    n_r = ref["mfcc39"].shape[0]
    n_s = stu["mfcc39"].shape[0]
    d["d_f0"] = f0_pattern_distance(ref["f0"], stu["f0"], path, n_r, n_s)
    d["d_warp"] = path_deviation(path, n_r, n_s)

    # mean spectral scalar distances
    for k in ("centroid", "bandwidth", "rolloff", "flatness", "zcr"):
        mr = float(resample_frames(ref["spec"][k].reshape(-1, 1), 60).mean())
        ms = float(resample_frames(stu["spec"][k].reshape(-1, 1), 60).mean())
        d[f"mean_{k}_ref"] = mr
        d[f"mean_{k}_stu"] = ms

    # Praat formants: mean F1/F2 per voiced region
    if do_praat:
        pa = LA.formants_praat(y_ref)
        pb = LA.formants_praat(y_stu)
        if pa and pb:
            (f1r, f2r), _ = pa
            (f1s, f2s), _ = pb
            d["F1_ref"] = float(np.median(f1r)) if len(f1r) else 0.0
            d["F2_ref"] = float(np.median(f2r)) if len(f2r) else 0.0
            d["F1_stu"] = float(np.median(f1s)) if len(f1s) else 0.0
            d["F2_stu"] = float(np.median(f2s)) if len(f2s) else 0.0
        else:
            d["F1_ref"] = d["F2_ref"] = d["F1_stu"] = d["F2_stu"] = 0.0

    # segment structure: vowel vs consonant distances along the DTW path
    vr = LA.voicing_by_frame(y_ref)
    vs = LA.voicing_by_frame(y_stu)
    d["ref_segments"] = LA.segment_speech(vr)
    d["stu_segments"] = LA.segment_speech(vs)
    d["video_voiced_ref"] = vr
    d["video_voiced_stu"] = vs
    # per-segment cost: average cost of the part of the DTW path inside it
    seg_errs = []
    m = stu["mfcc39"].shape[0]
    n = ref["mfcc39"].shape[0]
    error_frame = np.zeros(len(path))
    for k, (a, b) in enumerate(path):
        # frame error proxy: local static-MFCC euclidean, order 1.
        # m_stu was resampled to len(m_ref): map student frame b onto it
        vv = None
        if 0 <= a < n and 0 <= b < m:
            bb = min(len(m_stu) - 1, int(round(b * (len(m_stu) - 1) / max(1, m - 1))))
            vv = d_local_error(m_ref, m_stu, a, bb)
        error_frame[k] = 0 if vv is None else vv
    for (i0, i1, voiced) in d["ref_segments"]:
        kidx = [k for k, (a, b) in enumerate(path) if i0 <= a < i1]
        err = float(np.mean(error_frame[kidx])) if len(kidx) else 0.0
        seg_errs.append({"i0": int(i0), "i1": int(i1), "voiced": bool(voiced),
                         "type": "vowel" if voiced else "consonant", "err": err})
    # combine two consecutive unvoiced-frame runs into consonant clusters
    d["segments"] = seg_errs
    return d, ref, stu


def d_local_error(m_ref, m_stu, a, b):
    x = m_ref[a]
    yb = m_stu[b]
    return float(np.abs(x - yb).mean() / (np.abs(x).mean() + 1e-6))


# ---------------------------------------------------------------------------
# combined similarity %, with alpha calibrated on labelled pairs
# ---------------------------------------------------------------------------
def weighted_distance(d: dict) -> float:
    return (WEIGHTS["mfcc"] * d["d_mfcc"]
            + WEIGHTS["spec"] * d["d_spec"]
            + WEIGHTS["dur"] * d["d_dur"]
            + WEIGHTS["f0"] * d["d_f0"]
            + WEIGHTS["warp"] * d["d_warp"])


def path_deviation(path: list[tuple[int, int]], n_ref: int, n_stu: int) -> float:
    """Structural timing-discrepancy: how far the DTW warp path departs from
    the uniform time mapping. Strictly grows with time-stretch rate and
    region deletion, near zero for matched pairs — the deterministic monotone
    signal for the stretch/cut families (frame cost alone is warp-invariant)."""
    if not path or n_ref < 2 or n_stu < 2:
        return 0.0
    slope = (n_stu - 1) / (n_ref - 1)
    dev = [abs(b - a * slope) for a, b in path]
    return float(np.mean(dev) / n_stu)


def calibrate_alpha(pairs: list[tuple[dict, float]], low=0.02, high=60.0, steps=200):
    """pairs: [(analysis_dict, target_similarity)].  target in {very high, very low}.

    Minimise hinge loss against the acceptance criteria:
      identical pairs        -> similarity >= 95
      clearly-different word -> similarity <= 30
    Scans alpha on a log grid; among loss-0 candidates prefers the central one
    (max margin to both criteria), otherwise the loss minimiser, else largest
    separation between low and high pair scores.
    """
    dvals = np.array([weighted_distance(d) for d, _ in pairs])
    targets = np.array([t for _, t in pairs], dtype=float)
    best = None
    for a in np.geomspace(low, high, steps):
        scores = 100 * np.exp(-a * dvals)
        hi = targets > 90
        lo = targets < 40
        loss_hi = np.maximum(0, 95 - scores[hi]).sum()
        loss_lo = np.maximum(0, scores[lo] - 30).sum()
        loss = loss_hi + loss_lo
        # margin: distance of the worst pair from its criterion
        margin = max(0.0, min((scores[hi].min() - 95 if hi.any() else np.inf),
                              (30 - scores[lo].max() if lo.any() else np.inf)))
        key = (loss, -margin)
        if best is None or key < best[0]:
            best = (key, float(a), loss)
    return best[1], best[2]


# ---------------------------------------------------------------------------
# plain-language report
# ---------------------------------------------------------------------------
def verdict_text(score: float, d: dict, vowel_err: float, cons_err: float) -> str:
    bits = []
    if vowel_err > cons_err * 1.25:
        bits.append("the vowel parts match the reference the least")
    elif cons_err > vowel_err * 1.25:
        bits.append("the consonant clusters are noticeably weaker than the vowels")
    else:
        bits.append("vowels and consonants drift from the reference equally")
    if d["dur_ratio"] > 1.08:
        bits.append(f"duration is +{(d['dur_ratio'] - 1) * 100:.0f}%")
    elif d["dur_ratio"] < 0.92:
        bits.append(f"duration is -{(1 - d['dur_ratio']) * 100:.0f}%")
    if d["mean_centroid_stu"] and d["mean_centroid_ref"]:
        df = 100 * (d["mean_centroid_stu"] - d["mean_centroid_ref"]) / \
            max(d["mean_centroid_ref"], 1)
        bit = ("brightness" if df > 0 else "darkness")
        if abs(df) > 12:
            bits.append(f"spectral {bit} is {abs(df):.0f}% off")
    if abs(np.log(d["dur_ratio"])) < 0.053:
        bits.append("duration matches well")
    if score >= 85:  return "Excellent match — " + "; ".join(bits)
    if score >= 70:  return "Good — " + "; ".join(bits)
    if score >= 45:  return "Needs practice — " + "; ".join(bits)
    return "Quite different — " + "; ".join(bits)
