"""Unit tests for the project's core libraries.
Run:  pytest tests -q   (from project_out/)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "speech_a"))
sys.path.insert(0, str(ROOT / "src" / "speech_b"))

import lib_audio as LA  # noqa: E402


def test_frame_params_match_spec():
    n_fft, win, hop = LA.frame_params()
    assert win == int(LA.SR * 0.025) == 400      # 25 ms window
    assert hop == int(LA.SR * 0.010) == 160      # 10 ms hop
    assert n_fft == 512


def test_mfcc_dimensions():
    y = np.zeros(LA.SR, np.float32)
    y[8000:12000] = 1.0
    m = LA.mfcc_full(y)
    assert m.shape[1] == 39                      # 13 + delta + delta-delta


def test_dtw_cost_identical_equals_half_norm():
    import p1_core as P1
    x = np.random.default_rng(0).normal(size=(40, 13))
    c, path = P1.dtw_cost(x, x)
    assert c < 1e-9
    assert len(path) >= 40 and path[0] == (0, 0)


def test_dtw_cost_monotone_with_distortion():
    import p1_core as P1
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 13))
    c0, _ = P1.dtw_cost(x, x)
    c1, _ = P1.dtw_cost(x, x + rng.normal(0, 0.3, x.shape))
    c2, _ = P1.dtw_cost(x, x + rng.normal(0, 1.5, x.shape))
    assert c0 < c1 < c2


def test_second_powerlaw_voicing():
    y = np.zeros(LA.SR // 2, np.float32)
    t = np.arange(len(y)) / LA.SR
    y += (0.5 * np.sin(2 * np.pi * 120 * t)          # voiced component
          + 0.2 * np.sin(2 * np.pi * 240 * t)).astype(np.float32)
    vr = LA.voicing_by_frame(y)
    assert vr[len(vr) // 2]          # middle is voiced
    v = LA.segment_speech(vr)
    assert v and any(s[2] for s in v)


def test_cmvn_zero_mean_unit_std():
    x = np.random.default_rng(3).normal(3, 5, size=(100, 39))
    z = LA.cmvn(x)
    assert abs(z.mean()) < 1e-6
    assert abs(z.std() - 1) < 0.05


def test_similarity_transform_bounds():
    import p1_core as P1
    d = {"d_mfcc": 0.0, "d_spec": 0.0, "d_dur": 0.0, "d_f0": 0.0, "d_warp": 0.0}
    P1.ALPHA = 0.2
    assert P1.weighted_distance(d) == 0.0
    # similarity must be strictly decreasing in the weighted distance
    d2 = {"d_mfcc": 2.0, "d_spec": 1.0, "d_dur": 1.0, "d_f0": 1.0, "d_warp": 1.0}
    d3 = {"d_mfcc": 4.0, "d_spec": 2.0, "d_dur": 2.0, "d_f0": 2.0, "d_warp": 2.0}
    s1 = 100 * np.exp(-P1.ALPHA * P1.weighted_distance(d2))
    s2 = 100 * np.exp(-P1.ALPHA * P1.weighted_distance(d3))
    assert 0 < s1 < 100 and 0 < s2 < s1 and s2 < 60   # very-different pair < 60


def test_p2_features_pipeline():
    try:
        from p2_train_eval import clean, load_meta, feats
    except Exception:
        pytest.skip("p2 dataset not present")
    meta = load_meta()
    assert len(meta["verses"]) >= 10
    m = clean(feats(meta["clips"][0]["path"]))
    assert m.shape[1] == 39 and np.isfinite(m).all()


def test_cmvn_speech_tiny_segment():
    # a large mean offset must hurt far more than a tiny perturbation
    import p1_core as P1
    x = np.random.default_rng(2).normal(size=(50, 13))
    y = np.random.default_rng(2).normal(size=(50, 13))
    c_small, _ = P1.dtw_cost(x, x + 0.01 * y)
    c_big, _ = P1.dtw_cost(x, x + 10.0)
    assert c_big > 0.1           # mean offset must produce a substantial cost
    assert c_big > 5 * c_small   # and far more than a tiny perturbation

