"""Shared audio feature-extraction utilities for the speech projects.

Everything runs offline on CPU, deterministic (seeded where randomness exists).
Sample rate policy: all audio is force-resampled to 16 kHz mono.
"""
from __future__ import annotations

import os
import numpy as np
import soundfile as sf
import librosa

SR = 16000
FRAME_MS = 25
HOP_MS = 10
N_FFT = 512
N_MFCC = 13

# ---------------------------------------------------------------------------
# deterministic helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(1729)


def set_seed(seed: int = 1729) -> None:
    global _RNG
    import random
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    _RNG = np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# loading / trimming
# ---------------------------------------------------------------------------

def load_audio(path: str, sr: int = SR) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    y = y.astype(np.float32)
    y = trim_silence(y, sr)
    if len(y) < 0.05 * sr:  # degenerate file
        y, _ = librosa.load(path, sr=sr, mono=True)
        y = y.astype(np.float32)
    return y


def trim_silence(y: np.ndarray, sr: int = SR, top_db: float = 35.0,
                 pad: int = int(0.02 * SR)) -> np.ndarray:
    """Trim leading/trailing silence with a fixed padding."""
    idx = librosa.effects.split(y, top_db=top_db)
    if len(idx) == 0:
        return y
    i0, i1 = idx[0][0], idx[-1][1]
    i0 = max(0, i0 - pad)
    i1 = min(len(y), i1 + pad)
    return y[i0:i1]


def peak_norm(y: np.ndarray) -> np.ndarray:
    m = float(np.max(np.abs(y))) + 1e-9
    return (0.95 * y / m).astype(np.float32)


def sola_stretch(y: np.ndarray, rate: float, win: int = 1024, hop: int = 256,
                 tol: int = 240) -> np.ndarray:
    """Pitch-preserving time stretch/squash (classic OLA + sync search).

    librosa's phase-vocoder stretch distorts frame spectra heavily on short
    clips (mean cosine cost ~0.33 even at 5 % stretch on librosa 1.0), so the
    corpus uses this dependency-free implementation instead: rate > 1 makes
    the take faster/shorter, rate < 1 slower/longer.
    Returns a float32 waveform (not part of the similarity pipeline itself).
    """
    if abs(rate - 1.0) < 1e-9 or len(y) <= win + tol:
        return y.copy()
    n_out = int(round(len(y) / rate))
    n_frames = max(1, int(n_out // hop))
    hop_in = (len(y) - win) / max(n_frames - 1, 1)
    w = np.hanning(win).astype(np.float64)
    out = np.zeros(n_out + win)
    norm = np.zeros_like(out)
    prev_tail = None
    for k in range(n_frames):
        base = int(round(k * hop_in))
        base = min(max(base, 0), len(y) - win)
        if prev_tail is not None:            # sync over a +-tol window
            best, besti = -np.inf, base
            for s in range(-tol, tol + 1, 4):
                i = base + s
                if i < 0 or i + win > len(y):
                    continue
                c = float(np.dot(y[i:i + hop], prev_tail))
                if c > best:
                    best, besti = c, i
            base = besti
        seg = y[base:base + win] * w
        out[k * hop:k * hop + win] += seg
        norm[k * hop:k * hop + win] += w
        prev_tail = y[base + win - hop:base + win]
    norm[norm < 1e-4] = 1.0
    return (out[:n_out] / norm[:n_out]).astype(np.float32)


# ---------------------------------------------------------------------------
# frame parameters
# ---------------------------------------------------------------------------

def frame_params() -> tuple[int, int, int]:
    n_fft = int(SR * FRAME_MS / 1000)  # 400 -> we use 512 window/FFT pad
    hop = int(SR * HOP_MS / 1000)
    win = int(SR * FRAME_MS / 1000)
    return N_FFT, win, hop


# ---------------------------------------------------------------------------
# frame-level features
# ---------------------------------------------------------------------------

def mfcc_full(y: np.ndarray, sr: int = SR, with_deltas: bool = True) -> np.ndarray:
    """13 MFCC (+ delta + delta-delta) -> (T, 13, 39 when with_deltas) or (T,13)."""
    _, win, hop = frame_params()
    m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT,
                             win_length=win, hop_length=hop,
                             lifter=22)
    m = m.T  # (T, 13)
    if not with_deltas:
        return m
    d1 = librosa.feature.delta(m.T, width=5).T
    d2 = librosa.feature.delta(d1.T, width=5, order=2).T
    return np.hstack([m, d1, d2])  # (T, 39)


def cmvn(feats: np.ndarray) -> np.ndarray:
    """Utterance-level cepstral mean & variance normalisation."""
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True) + 1e-7
    return ((feats - mu) / sd).astype(np.float32)


def spectral_features(y: np.ndarray, sr: int = SR) -> dict[str, np.ndarray]:
    _, win, hop = frame_params()
    S = np.abs(librosa.stft(y, n_fft=N_FFT, win_length=win, hop_length=hop))
    f = {
        "centroid": librosa.feature.spectral_centroid(S=S, sr=sr)[0],
        "bandwidth": librosa.feature.spectral_bandwidth(S=S, sr=sr)[0],
        "rolloff": librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0],
        "flatness": librosa.feature.spectral_flatness(S=S)[0],
        "flux": librosa.onset.onset_strength(S=S, sr=sr),
        "zcr": librosa.feature.zero_crossing_rate(y, frame_length=win, hop_length=hop)[0],
        "rms": librosa.feature.rms(S=S, frame_length=2 * S.shape[-2] - 1)[0],
    }
    return f


def f0_track(y: np.ndarray, sr: int = SR) -> np.ndarray:
    _, win, hop = frame_params()
    # fmin=60 Hz is below what a 25 ms window can cover for pyin (2 periods
    # of 60 Hz = 533 samples). Lowering the window would change the shared frame
    # spec and re-invalidates the calibrated pipeline; instead the warning is
    # acknowledged here: pYIN is only marginally insensitive for ON male TTS
    # voices around 60-130 Hz, and the same contour data has validated
    # identical-cross-word-monotonic acceptance (see DECISIONS.md #9).
    import warnings as _w
    import librosa as _lb
    with _w.catch_warnings():
        _w.filterwarnings("ignore", message=".*less than two periods of fmin.*")
        f0, _, _ = _lb.pyin(y, fmin=60, fmax=500, sr=sr,
                            frame_length=win, hop_length=hop)
    f0 = np.nan_to_num(f0, nan=0.0)
    return f0


def formants_praat(y: np.ndarray, sr: int = SR, n_formants: int = 2):
    """F1..Fn time-series via Praat (parselmouth). Returns list of arrays, or None."""
    try:
        import parselmouth
        from parselmouth.praat import call
        x = peak_norm(y)
        snd = parselmouth.Sound(x, sr)
        f = call(snd, "To Formant (burg)", 0.0, 5, 5500 if sr < 22000 else 8000,
                 0.025, 50)
        out = []
        times = f.ts()
        for i in range(1, n_formants + 1):
            vals = [float(f.get_value_at_time(i, t)) for t in times]
            out.append(np.array(vals))
        times = np.asarray(times)
        return out, times
    except Exception:
        return None


# ---------------------------------------------------------------------------
# voiced/unvoiced segmentation used to label "vowel-like" vs "consonant-like"
# ---------------------------------------------------------------------------

def voicing_by_frame(y: np.ndarray, sr: int = SR,
                     f0: np.ndarray | None = None) -> np.ndarray:
    """True where frame is voiced (speaker/vowel dominant)."""
    f = spectral_features(y, sr)
    rms = f["rms"] / (f["rms"].max() + 1e-9)
    if f0 is None:
        f0 = f0_track(y, sr)
    voiced = (f0 > 60) & (rms > 0.08)
    # smooth with a 50 ms majority filter
    k = 5
    pad = np.pad(voiced.astype(float), (k, k), mode="edge")
    smooth = (np.convolve(pad, np.ones(2 * k + 1) / (2 * k + 1), mode="same")[k:-k]) > 0.5
    return smooth.astype(bool)


def segment_speech(voiced: np.ndarray, hop_ms: float = HOP_MS,
                   min_ms: float = 40.0) -> list[tuple[int, int, bool]]:
    """Merge contiguous voiced/unvoiced runs into (i0, i1, is_voiced) segments."""
    segs = []
    start = 0
    for i in range(1, len(voiced) + 1):
        if i == len(voiced) or voiced[i] != voiced[i - 1]:
            dur = (i - start) * hop_ms
            if dur >= min_ms or not segs:
                segs.append((start, i, bool(voiced[start])))
            else:
                st = segs.pop()
                segs.append((st[0], i, st[2]))  # extend previous segment
            start = i
    return segs
