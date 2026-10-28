"""P1 input pipeline: decode -> validate -> preprocess -> score-ready waveform.

Used by BOTH the microphone recording and the uploaded file in the
Pronunciation Assessment tab (app.py). The bundled student samples keep going
through ``lib_audio.load_audio`` unchanged so the calibrated scoring pipeline
sees exactly what it saw during calibration.

Key facts this module is built around (verified against Streamlit 1.66.1):

* ``st.audio_input`` returns ``UploadedFile | None`` (a BytesIO subclass).
  The frontend uploads it under a forced ``*_audio.wav`` file name, but the
  bytes are whatever ``MediaRecorder`` produced: the recorder prefers
  ``['audio/webm', 'audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/mp3']``,
  so Chrome/Edge/Firefox deliver **WebM/Opus** and Safari delivers
  **MP4/AAC** while claiming the name is ``.wav``.
* libsndfile (soundfile) cannot read WebM/MP4: the old code path saved the
  bytes as ``.wav`` and handed them to ``librosa.load`` which raised
  ``LibsndfileError: Format not recognised`` — the root cause of the broken
  microphone workflow.
* The bundled static ffmpeg (imageio-ffmpeg, already a project dependency)
  decodes WebM/Opus and MP4/AAC, so decode = sniff -> soundfile -> ffmpeg.

Nothing here touches feature extraction, DTW or the calibrated scoring
formula. The preprocessing was measured to shift similarity scores by
<= 0.01 points on the calibrated pairs (see tests/test_p1_audio.py), because
MFCCs are utterance-CMVN'd and the spectral channels are z-normalised, so a
gain change cancels out; alpha in cache/p1_alpha.json is untouched.
"""
from __future__ import annotations

import io
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

SR = 16000

# ---------------------------------------------------------------------------
# thresholds (all configurable; app.py exposes them in an expander)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Thresholds:
    min_duration_s: float = 0.30      # raw clip length
    max_duration_s: float = 15.0      # raw clip length (guards DTW runtime)
    min_speech_s: float = 0.15        # audible span after silence-trim
    max_speech_s: float = 6.0         # audible span after silence-trim
    min_peak: float = 0.02            # linear full-scale peak (~ -34 dBFS)
    min_rms: float = 0.004            # linear RMS (~ -48 dBFS)
    clip_level: float = 0.98          # samples at/above this count as clipped
    max_clipped_frac: float = 0.002   # 0.2 % of samples
    silence_top_db: float = 35.0      # same as lib_audio.trim_silence
    trim_pad_s: float = 0.02          # same as lib_audio.trim_silence pad


THRESHOLDS = Thresholds()


class AudioValidationError(ValueError):
    """Actionable input problem; never silently score through one."""

    def __init__(self, code: str, message: str, hint: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint

    def __str__(self) -> str:          # type: ignore[override]
        return self.message if not self.hint else f"{self.message} {self.hint}"


# ---------------------------------------------------------------------------
# container sniffing (the file name lies; magic bytes do not)
# ---------------------------------------------------------------------------
MIME = {
    "wav": "audio/wav", "webm": "audio/webm", "mp4": "audio/mp4",
    "mp3": "audio/mpeg", "ogg": "audio/ogg", "flac": "audio/flac",
    "aiff": "audio/aiff", "au": "audio/basic", "unknown": "audio/wav",
}
_SOUNDFILE_UNREADABLE = {"webm", "mp4"}      # libsndfile has no demuxer for these


def sniff_format(data: bytes) -> str:
    """Identify the real container from magic bytes (never trust the name)."""
    if len(data) < 12:
        return "unknown"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] == b"fLaC":
        return "flac"
    if data[:4] == b"OggS":
        return "ogg"
    if data[:4] == b"FORM":
        return "aiff"
    if data[:4] == b".snd":
        return "au"
    if data[:4] == b"\x1a\x45\xdf\xa3":        # EBML (webm/mkv)
        return "webm"
    if data[4:8] == b"ftyp":                    # mp4/m4a/mov
        return "mp4"
    if data[:3] == b"ID3":
        return "mp3"
    if data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:   # MPEG frame sync
        return "mp3"
    return "unknown"


def sniff_mime(data: bytes) -> str:
    return MIME.get(sniff_format(data), MIME["unknown"])


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------
@dataclass
class DecodedAudio:
    y: np.ndarray            # float32, shape (n, channels), native sample rate
    sr: int
    fmt: str                 # sniffed container


def _ffmpeg_exe() -> str:
    import sys
    from pathlib import Path as _P
    out = _P(__file__).resolve().parents[2]
    for sub in ("src",):
        p = str(out / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    from lib_data import ffmpeg_exe           # project-local, imageio-ffmpeg binary
    return ffmpeg_exe()


def _ffmpeg_decode(data: bytes, fmt: str) -> np.ndarray:
    """Transcode any container ffmpeg understands to a temp wav, read it back."""
    suffix = f".{fmt}" if fmt in MIME and fmt != "unknown" else ".bin"
    tmp = Path(tempfile.mkdtemp(prefix="p1dec_"))
    src, dst = tmp / f"in{suffix}", tmp / "out.wav"
    src.write_bytes(data)
    try:
        exe = _ffmpeg_exe()
        subprocess.run(
            [exe, "-y", "-loglevel", "error", "-i", str(src), str(dst)],
            check=True, capture_output=True, timeout=60)
    except FileNotFoundError as e:
        raise AudioValidationError(
            "unsupported",
            f"The recording is {fmt.upper()} audio, which the built-in decoder "
            "cannot read and no ffmpeg binary is available.",
            "Re-record in this browser, or upload a WAV/MP3/FLAC/OGG file.") from e
    except subprocess.TimeoutExpired as e:
        raise AudioValidationError(
            "undecodable", "Decoding the audio timed out after 60 s.",
            "Try a shorter recording (just the word).") from e
    except subprocess.CalledProcessError as e:
        if fmt == "unknown":
            raise AudioValidationError(
                "undecodable",
                "The file does not look like any supported audio format "
                "(unrecognised header), so it cannot be decoded.",
                "Supported: WAV, MP3, FLAC, OGG, AIFF, WebM/Opus, MP4/M4A. "
                "Re-export the audio as WAV and upload it again.") from e
        err = (e.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        err = err[-1][:160] if err else ""
        raise AudioValidationError(
            "unsupported",
            f"The {fmt.upper()} audio could not be decoded."
            + (f" Decoder said: {err}" if err else ""),
            "Re-export the audio as WAV and upload it again.") from e
    y, sr = sf.read(dst, dtype="float32", always_2d=True)
    return y, sr


def decode_audio(source: bytes | bytearray | memoryview | str | Path,
                 *, hint_name: str = "") -> DecodedAudio:
    """Decode the ACTUAL container to float32 (n, channels) at its native rate.

    Raises AudioValidationError for empty / undecodable / unsupported /
    non-finite content. Never trusts a file extension.
    """
    if isinstance(source, (str, Path)):
        p = Path(source)
        try:
            data = p.read_bytes()
        except OSError as e:
            raise AudioValidationError(
                "undecodable", f"Could not read the audio file ({p.name}).",
                "Check that the file still exists.") from e
        name = p.name
    else:
        data = bytes(source)
        name = hint_name

    if len(data) == 0:
        raise AudioValidationError(
            "empty",
            "The recording is empty (0 bytes) — nothing was captured."
            if not name else f"'{name}' is empty (0 bytes).",
            "Press the record button and speak the word, then stop.")

    fmt = sniff_format(data)
    y = None
    err = None
    if fmt not in _SOUNDFILE_UNREADABLE:       # fast, in-process path
        try:
            y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        except Exception as e:                 # libsndfile raises many types
            err = e
    if y is None:
        y, sr = _ffmpeg_decode(data, fmt)      # webm/mp4 or a soundfile miss

    if y is None or y.shape[0] == 0:
        raise AudioValidationError(
            "empty", "The audio contains no samples.",
            "Record again and make sure to stop after speaking.")
    if not np.isfinite(y).all():
        raise AudioValidationError(
            "non_finite",
            "The audio data contains NaN/Inf values and cannot be analysed.",
            "Record again; the capture was corrupted.")
    if y.shape[1] > 2:
        # ffmpeg could not have produced this, but guard anyway
        y = y[:, :1]
    return DecodedAudio(y=np.ascontiguousarray(y, dtype=np.float32),
                        sr=int(sr), fmt=fmt)


# ---------------------------------------------------------------------------
# preprocessing (single function for mic + upload)
# ---------------------------------------------------------------------------
@dataclass
class ProcessedAudio:
    y: np.ndarray                     # float32 mono, 16 kHz, finite, score-ready
    sr: int
    fmt: str
    native_sr: int
    n_channels: int
    raw_duration_s: float             # decoded, pre-trim
    duration_s: float                 # post-trim (what scoring sees)
    peak: float                       # post-trim, pre-normalisation
    peak_dbfs: float
    rms_dbfs: float
    clipped_frac: float
    trim_start_s: float
    trim_end_s: float
    raw_y: np.ndarray                 # mono, 16 kHz, pre-trim (for review plots)
    warnings: list = field(default_factory=list)

    @property
    def peak_pct(self) -> float:
        return 100.0 * self.clipped_frac

    def summary(self) -> dict:
        return {
            "duration_s": self.duration_s,
            "raw_duration_s": self.raw_duration_s,
            "peak_dbfs": self.peak_dbfs,
            "rms_dbfs": self.rms_dbfs,
            "clipped_frac": self.clipped_frac,
            "native_sr": self.native_sr,
            "n_channels": self.n_channels,
            "format": self.fmt,
        }


def _dbfs(x: float) -> float:
    return float(20.0 * np.log10(max(x, 1e-10)))


def _trim_bounds(y: np.ndarray, th: Thresholds) -> tuple[int, int]:
    """Same split/pad arithmetic as lib_audio.trim_silence (calibrated path)."""
    pad = int(th.trim_pad_s * SR)
    idx = librosa.effects.split(y, top_db=th.silence_top_db)
    if len(idx) == 0:
        return 0, len(y)
    i0 = max(0, int(idx[0][0]) - pad)
    i1 = min(len(y), int(idx[-1][1]) + pad)
    return i0, i1


def preprocess_audio(source: bytes | str | Path,
                     *,
                     thresholds: Thresholds = THRESHOLDS,
                     trim: bool = True,
                     normalize: bool = True,
                     hint_name: str = "") -> ProcessedAudio:
    """Decode -> mono -> 16 kHz -> validate -> trim -> normalise.

    The single entry point for microphone recordings and uploaded files.
    Raises AudioValidationError with an actionable message instead of
    returning junk for the DTW pipeline. No pitch correction, no denoising.
    """
    th = thresholds
    dec = decode_audio(source, hint_name=hint_name)
    y = dec.y
    n_channels = int(y.shape[1])
    native_sr = int(dec.sr)

    # --- mono ------------------------------------------------------------
    mono = y.mean(axis=1) if n_channels > 1 else y[:, 0]
    mono = np.asarray(mono, dtype=np.float32)

    # --- resample to 16 kHz ---------------------------------------------
    if native_sr != SR:
        mono = librosa.resample(mono, orig_sr=native_sr, target_sr=SR,
                                res_type="soxr_hq")
        mono = np.asarray(mono, dtype=np.float32)

    if mono.size == 0:
        raise AudioValidationError(
            "empty", "The audio contains no samples after decoding.",
            "Record again.")
    if not np.isfinite(mono).all():
        raise AudioValidationError(
            "non_finite",
            "The waveform contains NaN/Inf samples and cannot be scored.",
            "Record again; the capture was corrupted.")

    raw_duration = len(mono) / SR
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    clipped_frac = float(np.mean(np.abs(mono) >= th.clip_level))

    # --- validation, in a deliberate order -------------------------------
    if raw_duration < th.min_duration_s:
        raise AudioValidationError(
            "too_short",
            f"The recording is {raw_duration:.2f} s; at least "
            f"{th.min_duration_s:.2f} s is needed.",
            "Record the whole word, then stop.")
    if raw_duration > th.max_duration_s:
        raise AudioValidationError(
            "too_long",
            f"The recording is {raw_duration:.1f} s (maximum "
            f"{th.max_duration_s:.0f} s).",
            "Record just the single word — that is all that is compared.")
    if peak <= 1e-6:
        raise AudioValidationError(
            "silence",
            "The recording is silent — no audio was captured.",
            "Check that the browser has microphone permission and that the "
            "selected input device is not muted, then retake.")
    if peak < th.min_peak or rms < th.min_rms:
        raise AudioValidationError(
            "low_level",
            f"The recording is too quiet (peak {_dbfs(peak):.1f} dBFS, "
            f"RMS {_dbfs(rms):.1f} dBFS; minimum peak "
            f"{_dbfs(th.min_peak):.1f} dBFS).",
            "Move closer to the microphone, raise its gain, or speak louder, "
            "then retake. This is a level problem, not a pronunciation result.")
    if clipped_frac > th.max_clipped_frac:
        raise AudioValidationError(
            "clipped",
            f"{100 * clipped_frac:.1f}% of the samples are clipped "
            f"(full scale) — the input is overdriven.",
            "Move further from the microphone or lower its gain, then retake. "
            "Clipping distorts the spectrum and invalidates the comparison.")

    # --- how much of the clip is actually audible? -----------------------
    # Sample-level activity, NOT librosa.effects.split: the split window is
    # 2048 samples, so a 50 ms burst inside silence bleeds ~150 ms into the
    # "speech" region and a too-short take would never be rejected.
    thr_amp = peak * 10.0 ** (-th.silence_top_db / 20.0)
    active_s = float(np.count_nonzero(np.abs(mono) > thr_amp)) / SR
    if active_s < th.min_speech_s:
        raise AudioValidationError(
            "speech_too_short",
            f"Only {active_s:.2f} s of audible speech was found (minimum "
            f"{th.min_speech_s:.2f} s).",
            "You probably stopped before/after speaking the word — record "
            "again and say the whole word.")
    if active_s > th.max_speech_s:
        raise AudioValidationError(
            "too_long",
            f"{active_s:.1f} s of speech found (maximum "
            f"{th.max_speech_s:.0f} s).",
            "Only a single word is scored — record again without extra "
            "speech.")

    # --- trim leading/trailing silence (keeps onsets + final release) ----
    warnings: list = []
    if trim:
        i0, i1 = _trim_bounds(mono, th)
        speech = mono[i0:i1]
    else:
        i0, i1 = 0, len(mono)
        speech = mono
    speech = np.ascontiguousarray(speech, dtype=np.float32)

    # --- normalise only when appropriate, never divide by zero -----------
    pk2 = float(np.max(np.abs(speech))) if speech.size else 0.0
    if normalize and 1e-9 < pk2 < 0.999:
        speech = (0.95 * speech / pk2).astype(np.float32)
        if pk2 < 0.1:
            warnings.append(
                f"input was very low ({_dbfs(pk2):.1f} dBFS peak) and has been "
                "normalised — expect reduced reliability")

    # --- final gate before MFCC ------------------------------------------
    if speech.size < int(th.min_speech_s * SR):
        raise AudioValidationError(
            "speech_too_short",
            "The processed signal is too short for feature extraction.",
            "Record again.")
    if not np.isfinite(speech).all():
        raise AudioValidationError(
            "non_finite", "The processed signal is not finite.",
            "Record again; the capture was corrupted.")

    return ProcessedAudio(
        y=speech, sr=SR, fmt=dec.fmt, native_sr=native_sr,
        n_channels=n_channels, raw_duration_s=raw_duration,
        duration_s=len(speech) / SR, peak=pk2, peak_dbfs=_dbfs(pk2),
        rms_dbfs=_dbfs(rms), clipped_frac=clipped_frac,
        trim_start_s=i0 / SR, trim_end_s=i1 / SR, raw_y=mono,
        warnings=warnings)


# ---------------------------------------------------------------------------
# helpers for the review UI
# ---------------------------------------------------------------------------
def write_wav(path: str | Path, y: np.ndarray, sr: int = SR) -> Path:
    """Persist a processed waveform as a real 16-bit WAV for the scorer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y, dtype=np.float32)
    sf.write(str(path), np.clip(y, -1.0, 1.0), sr, subtype="PCM_16")
    return path


def wav_bytes(y: np.ndarray, sr: int = SR) -> bytes:
    """Processed waveform as WAV bytes (for st.audio playback)."""
    buf = io.BytesIO()
    y = np.asarray(y, dtype=np.float32)
    sf.write(buf, np.clip(y, -1.0, 1.0), sr, subtype="PCM_16",
             format="WAV")
    return buf.getvalue()


def waveform_peaks(y: np.ndarray, n: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """Down-sampled min/max envelope for a review-plot."""
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return np.zeros(1, np.float32), np.zeros(1, np.float32)
    n = max(2, min(n, y.size))
    edges = np.linspace(0, y.size, n + 1).astype(int)
    mins = np.array([y[edges[i]:edges[i + 1]].min() if edges[i + 1] > edges[i]
                     else 0.0 for i in range(n)], dtype=np.float32)
    maxs = np.array([y[edges[i]:edges[i + 1]].max() if edges[i + 1] > edges[i]
                     else 0.0 for i in range(n)], dtype=np.float32)
    return mins, maxs


# ---------------------------------------------------------------------------
# recording-state helpers (pure: testable without Streamlit)
# ---------------------------------------------------------------------------
@dataclass
class Take:
    """One microphone take, tagged with the word it was recorded for."""
    digest: str
    word: str
    data: bytes
    mime: str = "audio/wav"
    created: float = 0.0
    saved_path: str | None = None       # processed WAV the scorer reads
    proc: ProcessedAudio | None = None  # None + error => rejected take
    error: str | None = None
    code: str | None = None
    th: "Thresholds | None" = None      # thresholds used for the last check

    @property
    def ok(self) -> bool:
        return self.error is None and self.proc is not None


def make_take(data: bytes, word: str, *, mime: str | None = None,
              created: float | None = None, **kw) -> Take:
    import hashlib
    return Take(digest=hashlib.md5(data).hexdigest(), word=word, data=data,
                mime=mime or sniff_mime(data),
                created=time.time() if created is None else created, **kw)


def accept_take(current: Take | None, data: bytes, word: str) -> tuple[Take, bool]:
    """Decide which take the app should store, and whether it is a NEW one.

    Streamlit re-delivers the same ``st.audio_input`` value on every rerun,
    so the stored take must only change when the captured bytes change;
    otherwise each rerun would look like a fresh recording (and would
    silently re-tag an old take with a newly selected word).

    Returns ``(take_to_store, is_new)``.
    """
    import hashlib
    digest = hashlib.md5(data).hexdigest()
    if current is not None and current.digest == digest:
        return current, False          # same take, whatever word is selected
    return make_take(data, word), True


def take_status(take: Take | None, word: str) -> str:
    """'none' | 'ok' | 'stale' | 'rejected' — what a take means for `word`."""
    if take is None:
        return "none"
    if take.word != word:
        return "stale"
    if take.error is not None:
        return "rejected"
    if take.proc is None:
        return "pending"            # stored but not validated yet
    return "ok"


@dataclass(frozen=True)
class SourceChoice:
    kind: str            # "microphone" | "upload" | "bundled"
    label: str
    data: bytes | None   # raw bytes for microphone/upload, None for bundled
    path: str | None     # bundled sample path, None for microphone/upload
    mime: str | None
    take: Take | None = None


def choose_student_source(mode: str, *,
                          take: Take | None,
                          word: str,
                          upload: bytes | None,
                          upload_name: str,
                          bundled_path: str,
                          bundled_label: str) -> SourceChoice:
    """Explicit input-source selection — never a silent fallback.

    Raises AudioValidationError with an actionable message when the chosen
    source is missing or is a stale take, instead of quietly scoring a
    bundled sample the user did not select.
    """
    mode = (mode or "").strip().lower()
    if mode.startswith("mic"):
        if take is None:
            raise AudioValidationError(
                "no_take",
                "No microphone recording is available yet.",
                "Press **Start recording**, say the word, stop, review it — "
                "or choose a different input source.")
        status = take_status(take, word)
        if status == "stale":
            raise AudioValidationError(
                "stale_take",
                f"The stored recording is for '{take.word}', but '{word}' is "
                "selected now.",
                "Discard the old take and record the selected word — "
                "scoring it would compare the wrong word.")
        if status == "pending":
            raise AudioValidationError(
                "unvalidated",
                "The recording has not been validated yet.",
                "Retry the comparison.")
        if status == "rejected":
            raise AudioValidationError(
                take.code or "invalid",
                f"The recording was rejected: {take.error}",
                "Discard it and retake.")
        return SourceChoice("microphone", f"microphone take for '{take.word}'",
                            take.data, None, take.mime, take)
    if mode.startswith("upl") or mode.startswith("file"):
        if not upload:
            raise AudioValidationError(
                "no_upload",
                "No file is uploaded yet.",
                "Choose an audio file first, or pick another input source.")
        return SourceChoice("upload", upload_name or "uploaded file",
                            bytes(upload), None,
                            sniff_mime(bytes(upload)))
    if mode.startswith("bund") or mode.startswith("sam") or mode.startswith("stud"):
        return SourceChoice("bundled", bundled_label, None, bundled_path,
                            sniff_mime(Path(bundled_path).read_bytes())
                            if Path(bundled_path).exists() else "audio/wav")
    raise AudioValidationError(
        "unknown_source", f"Unknown input source '{mode}'.",
        "Choose microphone, upload or bundled sample.")
