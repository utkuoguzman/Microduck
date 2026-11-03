"""Cactus Whistle Speech-to-Text (S2T) Engine for Microduck / XGO-Duck + Needle 3.

Integrates **Cactus Whistle** (`Cactus-Compute/whistle`, `whistle_d4.cact` / `whistle.cact`)
from Cactus Compute (`cactus-needle==3.2.10` + `andrisgauracs/whistle-esp32`) into our
unified simulator and Web UI so you can command the duck with your voice!

Why this architecture is awesome:
1. Both **Whistle** (Speech-to-Text) and **Needle 3** (Tool Calling) are from Cactus Compute
   and share the exact same `.cact` v4 container format (`0x05E12A84`) and **Needle 3** blocks:
   - Hadamard / Monarch Kronecker MLPs (`16x32` factors + permutations)
   - mHC multi-lane Sinkhorn residual mixing
   - N-gram Engram tables (`0x9E3779B9` FNV-style hash)
2. Dual-Backend Execution:
   - **Primary (`needle.Whistle` C++ Engine):** Runs `whistle_d4.cact` (14.5 MB, 4-layer laddered
     decoder stripped via `whistle-esp32/tools/whistle_strip.py`) in **~65–80 ms** (`~300 tok/s`,
     `~30 ms` TTFT) while `NeedleZooBrain` runs `duck_4l.cact` in its `FineTuneWorker` subprocess!
   - **Fallback (`whistle-esp32/tools/whistle_ref.py`):** 100% Pure-NumPy reference forward pass
     when the C++ shared library is unavailable.
"""
from __future__ import annotations

import io
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

SUITE_DIR = Path(__file__).resolve().parent
ROOT_DIR = SUITE_DIR.parent
WHISTLE_REPO_TOOLS = ROOT_DIR / "tools" / "whistle_tools"
if not WHISTLE_REPO_TOOLS.is_dir():
    WHISTLE_REPO_TOOLS = ROOT_DIR / "vendor" / "whistle-esp32" / "tools"
MODELS_WHISTLE_DIR = ROOT_DIR / "ml_checkpoints" / "cactus_whistle"

if str(WHISTLE_REPO_TOOLS) not in sys.path:
    sys.path.insert(0, str(WHISTLE_REPO_TOOLS))

# Domain keywords for optional keyword biasing in Cactus Whistle
DUCK_BIAS_KEYWORDS = [
    "please",
    "walk forward",
    "walk backward",
    "turn left",
    "turn right",
    "backflip",
    "moonwalk",
    "roulade",
    "somersault",
    "headstand",
    "side roll",
    "long jump",
    "happy hop",
    "polite bow",
    "groove dance",
    "robot dance",
    "peck seeds",
    "stand on one leg",
    "play dead",
    "sprint",
    "waddle",
    "kick",
    "roller",
]


def ensure_whistle_models() -> Tuple[Path, Path]:
    """Ensure `whistle.cact` (16.9 MB) and `whistle_d4.cact` (14.5 MB, depth=4) exist on disk."""
    MODELS_WHISTLE_DIR.mkdir(parents=True, exist_ok=True)
    full_path = MODELS_WHISTLE_DIR / "whistle.cact"
    d4_path = MODELS_WHISTLE_DIR / "whistle_d4.cact"

    if not full_path.is_file():
        from huggingface_hub import hf_hub_download
        import shutil

        cached = hf_hub_download("Cactus-Compute/whistle", "whistle.cact")
        shutil.copyfile(cached, full_path)

    if not d4_path.is_file() and full_path.is_file():
        from whistle_strip import strip  # type: ignore

        strip(str(full_path), 4, str(d4_path))

    return full_path, d4_path


def resample_to_16khz_mono(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    """Convert mono/stereo float32 audio at `orig_sr` to 16,000 Hz mono float32."""
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if orig_sr == 16000 or len(audio) == 0:
        return audio
    duration = len(audio) / float(orig_sr)
    target_len = max(1, int(round(duration * 16000)))
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def decode_wav_bytes_to_16k_float32(wav_bytes: bytes) -> np.ndarray:
    """Parse any 16-bit/8-bit/32-bit PCM WAV (any sample rate, mono or stereo) into 16 kHz mono float32."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)

    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return resample_to_16khz_mono(samples, framerate)


class WhistleSpeechToText:
    """Cactus Whistle Speech-to-Text engine (`whistle_d4.cact` / `whistle.cact`).

    Uses the native C++ `needle.Whistle` engine (~75 ms per clip, ~300 tok/s) when available,
    and falls back to `whistle-esp32/tools/whistle_ref.py` (100% pure NumPy) otherwise.
    """

    def __init__(self, prefer_d4: bool = True, language: str = "en", use_numpy_ref: bool = False):
        self.language = language
        self.available = False
        self.backend = "none"
        self.model_path: Optional[Path] = None
        self._cpp_whistle = None
        self._np_whistle = None
        self._np_decoder = None
        self._load(prefer_d4=prefer_d4, use_numpy_ref=use_numpy_ref)

    def _load(self, prefer_d4: bool = True, use_numpy_ref: bool = False) -> None:
        full_path, d4_path = ensure_whistle_models()
        chosen = d4_path if (prefer_d4 and d4_path.is_file()) else full_path
        self.model_path = chosen

        if not use_numpy_ref:
            try:
                import needle

                self._cpp_whistle = needle.Whistle(weights=str(chosen))
                self.backend = "needle-cpp"
                self.available = True
                return
            except Exception as exc:
                print(f"[WhistleS2T] C++ needle.Whistle unavailable ({exc}); falling back to Pure-NumPy whistle_ref.")

        try:
            from whistle_ref import Decoder, Whistle  # type: ignore

            self._np_whistle = Whistle(str(chosen))
            self._np_decoder = Decoder(self._np_whistle)
            self.backend = "numpy-ref"
            self.available = True
        except Exception as exc:
            print(f"[WhistleS2T] Warning: could not load Whistle model ({exc})")
            self.available = False

    def transcribe_audio(
        self,
        audio_16k: np.ndarray,
        language: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        max_tokens: int = 32,
    ) -> dict:
        """Transcribe 16 kHz mono float32 audio array using Cactus Whistle."""
        if not self.available:
            raise RuntimeError("Whistle S2T model is not loaded")

        t0 = time.perf_counter()
        audio_16k = np.asarray(audio_16k, dtype=np.float32).reshape(-1)

        # Noise gate check (from whistle-esp32 README: avoid transcribing pure silence as 'Thank you.')
        rms = float(np.sqrt(np.mean(audio_16k**2))) if len(audio_16k) > 0 else 0.0
        peak = float(np.max(np.abs(audio_16k))) if len(audio_16k) > 0 else 0.0
        if len(audio_16k) < 1600 or peak < 0.005:
            return {
                "ok": False,
                "text": "",
                "rms": round(rms, 5),
                "peak": round(peak, 5),
                "ms": round((time.perf_counter() - t0) * 1000.0, 1),
                "backend": self.backend,
                "error": "Audio too quiet or short (silence detected — hold button and speak clearly)",
            }

        lang = language or self.language
        if self._cpp_whistle is not None:
            res = self._cpp_whistle.transcribe(
                audio_16k,
                language=lang,
                keywords=keywords,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            raw_text = str(res.get("text", "")).strip().strip('"')
            return {
                "ok": True,
                "text": raw_text,
                "language": res.get("language", lang),
                "ttft_ms": res.get("ttft_ms"),
                "decode_tps": res.get("decode_tps"),
                "duration_s": round(len(audio_16k) / 16000.0, 2),
                "rms": round(rms, 4),
                "peak": round(peak, 4),
                "ms": elapsed_ms,
                "backend": self.backend,
                "model": self.model_path.name if self.model_path else "whistle",
            }

        assert self._np_whistle is not None and self._np_decoder is not None
        enc = self._np_whistle.encode(audio_16k)
        text, token_ids = self._np_decoder.greedy(enc, language=lang, max_tokens=max_tokens)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        return {
            "ok": True,
            "text": text.strip().strip('"'),
            "tokens": token_ids,
            "language": lang,
            "duration_s": round(len(audio_16k) / 16000.0, 2),
            "rms": round(rms, 4),
            "peak": round(peak, 4),
            "ms": elapsed_ms,
            "backend": self.backend,
            "model": self.model_path.name if self.model_path else "whistle",
        }

    def transcribe_wav_bytes(self, wav_bytes: bytes, language: Optional[str] = None) -> dict:
        audio_16k = decode_wav_bytes_to_16k_float32(wav_bytes)
        return self.transcribe_audio(audio_16k, language=language)

    def transcribe_wav_file(self, wav_path: str | Path, language: Optional[str] = None) -> dict:
        raw_bytes = Path(wav_path).read_bytes()
        return self.transcribe_wav_bytes(raw_bytes, language=language)


def record_windows_mic_wav(seconds: float = 3.0) -> bytes:
    """Record 16 kHz 16-bit mono WAV from the default Windows microphone using built-in `winmm.dll`."""
    import ctypes

    winmm = ctypes.windll.winmm

    def mci(cmd: str) -> None:
        buf = ctypes.create_unicode_buffer(256)
        err = winmm.mciSendStringW(cmd, buf, 255, 0)
        if err != 0:
            raise RuntimeError(f"MCI command failed ({cmd!r}): error code {err}")

    tmp_file = Path(tempfile.mktemp(suffix=".wav"))
    try:
        mci("close mic_rec")
    except Exception:
        pass

    try:
        mci("open new type waveaudio alias mic_rec")
        mci("set mic_rec time format ms bitspersample 16 channels 1 samplespersec 16000 bytespersec 32000 alignment 2")
        print(f"  [Whistle Mic] Recording for {seconds:.1f}s... Speak now! (Say 'please ...')")
        mci("record mic_rec")
        time.sleep(seconds)
        mci("stop mic_rec")
        mci(f'save mic_rec "{tmp_file}"')
        mci("close mic_rec")
        return tmp_file.read_bytes()
    finally:
        try:
            mci("close mic_rec")
        except Exception:
            pass
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass


def synthesize_spoken_wav_windows(phrase: str) -> bytes:
    """Use Windows SAPI.SpVoice TTS to synthesize a spoken 16 kHz mono WAV for end-to-end Whistle testing!"""
    import subprocess

    tmp_file = Path(tempfile.mktemp(suffix=".wav"))
    ps_script = (
        f"Add-Type -AssemblyName System.Speech; "
        f"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
        f"[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
        f"[System.Speech.AudioFormat.AudioChannel]::Mono); "
        f"$s.SetOutputToWaveFile('{tmp_file}', $fmt); "
        f"$s.Speak(\"{phrase}\"); "
        f"$s.Dispose()"
    )
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], check=True, timeout=15)
        return tmp_file.read_bytes()
    finally:
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass
