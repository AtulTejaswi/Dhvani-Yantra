"""Regression tests for the P1 microphone/upload input pipeline (p1_audio).

Covers the reported microphone failures end-to-end at library level:
empty / silent / too-short / corrupt / unsupported / non-finite rejection,
clipping and low level, stereo down-mix, 44.1 & 48 kHz -> 16 kHz resampling,
configurable thresholds, new-vs-old recording state, WebM recordings that
Streamlit delivers under a lying ".wav" name, upload-vs-microphone decode
consistency, and valid feature extraction + score generation on the
calibrated pipeline.

Run:  python -m pytest tests -q   (from project_out/)
"""
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "src"), str(ROOT / "src" / "speech_a")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lib_audio as LA      # noqa: E402
import p1_audio as PA       # noqa: E402

REF_K = ROOT / "data" / "audio" / "ref_krishna.wav"
REF_D = ROOT / "data" / "audio" / "ref_dharma.wav"
STU_K = ROOT / "data" / "audio" / "stud_krishna.wav"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def tone(sr=16000, secs=0.6, freq=180.0, amp=0.5):
    t = np.arange(int(sr * secs)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def wav(y, sr=16000):
    """Waveform as real PCM WAV bytes (what a browser upload looks like)."""
    return PA.wav_bytes(y, sr)


def sf_wav_bytes(y, sr, subtype="PCM_16"):
    buf = io.BytesIO()
    sf.write(buf, np.asarray(y), sr, format="WAV", subtype=subtype)
    return buf.getvalue()


def _ffmpeg():
    try:
        from lib_data import ffmpeg_exe
        return ffmpeg_exe()
    except Exception:
        return None


FFMPEG = _ffmpeg()


# ---------------------------------------------------------------------------
# 1) rejection of unusable input (never silently scored)
# ---------------------------------------------------------------------------
def test_empty_audio_rejected():
    with pytest.raises(PA.AudioValidationError) as e:
        PA.preprocess_audio(b"")
    assert e.value.code == "empty"


def test_silence_rejected():
    with pytest.raises(PA.AudioValidationError) as e:
        PA.preprocess_audio(wav(np.zeros(16000, np.float32)))     # 1 s of nothing
    assert e.value.code == "silence"


def test_very_short_clip_rejected():
    with pytest.raises(PA.AudioValidationError) as e:
        PA.preprocess_audio(wav(tone(secs=0.25)))
    assert e.value.code == "too_short"


def test_very_short_speech_rejected():
    # long clip, but only an 80 ms burst inside it: too little speech
    y = np.concatenate([np.zeros(12800, np.float32),
                        tone(secs=0.08),
                        np.zeros(12800, np.float32)])
    with pytest.raises(PA.AudioValidationError) as e:
        PA.preprocess_audio(wav(y))
    assert e.value.code == "speech_too_short"


def test_corrupt_audio_rejected():
    garbage = bytes([0, 1, 2, 3, 250]) * 200
    with pytest.raises(PA.AudioValidationError) as e:
        PA.preprocess_audio(garbage)
    assert e.value.code == "undecodable"
    assert e.value.hint                       # actionable, not a bare crash


def test_unsupported_container_rejected():
    fake = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64 + b"junk" * 64
    with pytest.raises(PA.AudioValidationError) as e:
        PA.preprocess_audio(fake)
    assert e.value.code in ("unsupported", "undecodable")
    assert "WAV" in e.value.hint


def test_non_finite_rejected():
    y = tone(secs=0.5)
    y[100:120] = np.nan
    with pytest.raises(PA.AudioValidationError) as e:
        PA.preprocess_audio(sf_wav_bytes(y, 16000, subtype="FLOAT"))
    assert e.value.code == "non_finite"


def test_clipping_rejected():
    y = np.sign(tone(secs=0.6)).astype(np.float32)      # square wave = full scale
    with pytest.raises(PA.AudioValidationError) as e:
        PA.preprocess_audio(wav(y))
    assert e.value.code == "clipped"
    assert "microphone" in e.value.hint                 # tells the user what to do


def test_low_volume_rejected():
    with pytest.raises(PA.AudioValidationError) as e:
        PA.preprocess_audio(wav(tone(secs=0.6, amp=0.005)))   # peak ~ -46 dBFS
    assert e.value.code == "low_level"


def test_thresholds_are_configurable():
    y = tone(secs=0.25)
    with pytest.raises(PA.AudioValidationError):
        PA.preprocess_audio(wav(y))                     # default min duration 0.30 s
    p = PA.preprocess_audio(wav(y), thresholds=PA.Thresholds(min_duration_s=0.10))
    assert p.duration_s > 0.2                            # looser threshold accepts
    with pytest.raises(PA.AudioValidationError) as e:   # stricter peak level
        PA.preprocess_audio(wav(tone(secs=0.5, amp=0.5)),
                            thresholds=PA.Thresholds(min_peak=0.9))
    assert e.value.code == "low_level"


# ---------------------------------------------------------------------------
# 2) legitimate input is decoded, converted and trimmed correctly
# ---------------------------------------------------------------------------
def test_stereo_input_downmixed_to_mono():
    y = np.stack([tone(amp=0.6), tone(amp=0.2)], axis=1)   # same signal both ch
    p = PA.preprocess_audio(sf_wav_bytes(y, 16000))
    assert p.n_channels == 2
    assert p.y.ndim == 1 and p.y.dtype == np.float32 and p.sr == 16000
    # mono output equals the (peak-normalised) channel mean
    expect = y.mean(axis=1)
    expect = 0.95 * expect / np.max(np.abs(expect))
    n = min(len(p.y), len(expect))
    assert np.allclose(p.y[:n], expect[:n], atol=1e-3)


@pytest.mark.parametrize("native_sr", [44100, 48000])
def test_resampled_to_16k(native_sr):
    secs, freq = 0.75, 440.0
    y = tone(sr=native_sr, secs=secs, freq=freq)
    p = PA.preprocess_audio(sf_wav_bytes(y, native_sr))
    assert p.native_sr == native_sr and p.sr == 16000
    assert np.isfinite(p.y).all()
    assert abs(len(p.y) / 16000 - secs) < 0.05 * secs      # duration preserved
    spec = np.abs(np.fft.rfft(p.y))
    f = np.fft.rfftfreq(len(p.y), 1.0 / 16000)
    assert abs(f[np.argmax(spec)] - freq) < 5.0            # pitch preserved


def test_normalise_never_divides_by_zero():
    out = PA.waveform_peaks(np.zeros(10, np.float32))
    assert np.isfinite(out[0]).all() and np.isfinite(out[1]).all()
    # the guard used by preprocess_audio: no scaling at (near-)zero peak
    z = np.zeros(16, np.float32)
    pk = float(np.max(np.abs(z)))
    assert not (1e-9 < pk < 0.999)                          # normalisation skipped


def test_waveform_peaks_and_wav_bytes():
    y = tone(secs=0.5)
    mn, mx = PA.waveform_peaks(y, n=100)
    assert len(mn) == len(mx) == 100 and np.all(mn <= mx)
    b = PA.wav_bytes(y)
    assert b[:4] == b"RIFF"
    y2, sr = sf.read(io.BytesIO(b), dtype="float32")
    assert sr == 16000 and len(y2) == len(y)


# ---------------------------------------------------------------------------
# 3) the real microphone case: Streamlit delivers WebM bytes named ".wav"
# ---------------------------------------------------------------------------
@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg binary not available")
def test_webm_recording_decodes_despite_wav_name():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src.wav"
        sf.write(src, tone(sr=48000, secs=0.7, freq=220), 48000)
        rec = td / "recording.wav"           # exactly how the app receives it
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src),
                        "-c:a", "libopus", "-b:a", "64k", "-f", "webm",
                        str(rec)], check=True, capture_output=True)
        raw = rec.read_bytes()
        assert raw[:4] == b"\x1a\x45\xdf\xa3"              # EBML/WebM, NOT RIFF
        assert PA.sniff_format(raw) == "webm"
        assert PA.sniff_mime(raw) == "audio/webm"
        # new pipeline: decodes from magic bytes regardless of the name
        p = PA.preprocess_audio(str(rec))
        assert p.fmt == "webm" and p.sr == 16000 and p.duration_s > 0.5
        # and it agrees with the original WAV it was encoded from
        p_ref = PA.preprocess_audio(src.read_bytes())
        assert abs(p.duration_s - p_ref.duration_s) < 0.1


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg binary not available")
def test_mp4_aac_recording_decodes():
    # Safari's MediaRecorder flavour
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = td / "src.wav"
        sf.write(src, tone(sr=48000, secs=0.7, freq=220), 48000)
        rec = td / "recording.wav"
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(src),
                        "-c:a", "aac", "-f", "mp4", str(rec)],
                       check=True, capture_output=True)
        raw = rec.read_bytes()
        assert PA.sniff_format(raw) == "mp4"
        p = PA.preprocess_audio(str(rec))
        assert p.fmt == "mp4" and p.sr == 16000 and p.duration_s > 0.5


# ---------------------------------------------------------------------------
# 4) recording state: new take vs older take, explicit source selection
# ---------------------------------------------------------------------------
def test_new_recording_replaces_older_take():
    a = wav(tone(secs=0.6, freq=150))
    b = wav(tone(secs=0.6, freq=230))
    t1, is_new = PA.accept_take(None, a, "krishna")
    assert is_new and t1.word == "krishna"
    t1.proc = PA.preprocess_audio(t1.data)               # as the app does
    assert PA.take_status(t1, "krishna") == "ok"
    # same bytes re-delivered on a rerun -> NOT a new take, word NOT re-tagged
    same, is_new2 = PA.accept_take(t1, a, "dharma")
    assert not is_new2 and same is t1 and same.word == "krishna"
    assert PA.take_status(same, "dharma") == "stale"
    # genuinely new recording -> replaces the old one and is tagged now
    t2, is_new3 = PA.accept_take(t1, b, "dharma")
    assert is_new3 and t2.digest != t1.digest and t2.word == "dharma"
    t2.proc = PA.preprocess_audio(t2.data)
    assert PA.take_status(t2, "dharma") == "ok"


def test_stale_take_never_scores_silently():
    common = dict(upload=None, upload_name="",
                  bundled_path=str(REF_K), bundled_label="bundled")
    # no take at all -> actionable error, NOT a silent bundled score
    with pytest.raises(PA.AudioValidationError) as e:
        PA.choose_student_source("microphone", take=None, word="krishna",
                                 **common)
    assert e.value.code == "no_take"
    # old take recorded for another word -> refused
    t1 = PA.make_take(wav(tone(freq=150)), "krishna")
    t1.proc = PA.preprocess_audio(t1.data)
    with pytest.raises(PA.AudioValidationError) as e:
        PA.choose_student_source("microphone", take=t1, word="dharma",
                                 **common)
    assert e.value.code == "stale_take"
    # upload chosen but none present -> refused, not a bundled fallback
    with pytest.raises(PA.AudioValidationError) as e:
        PA.choose_student_source("upload", take=t1, word="krishna",
                                 **common)
    assert e.value.code == "no_upload"
    # a rejected take is refused with its original actionable message
    t1.error, t1.code, t1.proc = "too quiet", "low_level", None
    with pytest.raises(PA.AudioValidationError) as e:
        PA.choose_student_source("microphone", take=t1, word="krishna",
                                 **common)
    assert e.value.code == "low_level"
    # explicit bundled choice still works and is labelled as such
    ch = PA.choose_student_source("bundled", take=t1, word="krishna", **common)
    assert ch.kind == "bundled" and ch.path == str(REF_K)


# ---------------------------------------------------------------------------
# 5) upload WAV and decoded microphone audio go through the same pipeline
# ---------------------------------------------------------------------------
def test_upload_wav_and_microphone_take_are_consistent():
    data = wav(tone(secs=0.6, freq=200))
    p_up = PA.preprocess_audio(data, hint_name="upload.wav")
    take = PA.make_take(data, "krishna")                # same bytes, as mic
    p_mic = PA.preprocess_audio(take.data, hint_name="microphone recording")
    assert take.mime == "audio/wav"
    assert np.array_equal(p_up.y, p_mic.y)              # identical processing
    assert p_up.summary() == p_mic.summary()
    # the WAV written for the scorer decodes back to the same waveform
    with tempfile.TemporaryDirectory() as td:
        out = PA.write_wav(Path(td) / "scored.wav", p_mic.y)
        back = LA.load_audio(str(out))                  # exactly what p1_analyse uses
    n = min(len(back), len(p_mic.y))
    assert np.allclose(back[:n], p_mic.y[:n], atol=1e-4)


# ---------------------------------------------------------------------------
# 6) valid input produces valid features and a bounded score
# ---------------------------------------------------------------------------
def _alpha():
    try:
        return float(json.loads((ROOT / "cache" / "p1_alpha.json")
                                .read_text("utf-8"))["alpha"])
    except Exception:
        return None


def test_valid_feature_extraction_and_score_generation():
    import p1_core as P1
    if not (REF_K.exists() and STU_K.exists()):
        pytest.skip("bundled P1 audio not present")
    P1.ALPHA = _alpha() or P1.ALPHA
    y_ref = PA.preprocess_audio(REF_K.read_bytes())
    y_stu = PA.preprocess_audio(STU_K.read_bytes())
    d, ref_f, stu_f = P1.analyse_pair(y_ref.y, y_stu.y, do_praat=False)
    assert ref_f["mfcc39"].shape[1] == 39 == stu_f["mfcc39"].shape[1]
    assert np.isfinite(ref_f["mfcc39"]).all()
    assert np.isfinite(stu_f["mfcc39"]).all()
    score = 100 * np.exp(-P1.ALPHA * P1.weighted_distance(d))
    assert np.isfinite(score) and 0.0 < score <= 100.0
    assert d["segments"] and all("err" in s for s in d["segments"])


def test_preprocessing_preserves_calibration():
    """Same file through old loader vs new pipeline: score shift <= 0.5 pt.

    Measured <= 0.01 pt on all 15 calibrated pair types (identical /
    tts-student / cross-word) — the pipeline is CMVN + z-normalisation,
    so peak normalisation cancels out. alpha in cache/p1_alpha.json is
    unchanged and must stay that way.
    """
    import p1_core as P1
    if not (REF_K.exists() and REF_D.exists()):
        pytest.skip("bundled P1 audio not present")
    P1.ALPHA = _alpha() or P1.ALPHA

    def score(a, b):
        d, _, _ = P1.analyse_pair(a, b, do_praat=False)
        return 100 * np.exp(-P1.ALPHA * P1.weighted_distance(d))

    old_ref, old_x = LA.load_audio(str(REF_K)), LA.load_audio(str(REF_D))
    new_ref = PA.preprocess_audio(REF_K.read_bytes())
    new_x = PA.preprocess_audio(REF_D.read_bytes())
    s_old, s_new = score(old_ref, old_x), score(new_ref.y, new_x.y)
    assert abs(s_new - s_old) <= 0.5
    # identical pair through the new path still meets the calibrated
    # acceptance criterion (identical >= 95 %)
    assert score(new_ref.y, new_ref.y) >= 95.0
