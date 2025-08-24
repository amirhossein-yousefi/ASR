from typing import Dict, Any, Optional
from datasets import load_dataset, Audio, DatasetDict
from transformers import AutoProcessor
import numpy as np
import platform
import os
import shutil
import subprocess

from ..utils.text_normalization import default_english_normalizer, whitelist_normalizer
from ..utils.augment import WaveformAugment, AugmentConfig
from ..utils.log import get_logger

logger = get_logger("data")
IS_WINDOWS = platform.system().lower().startswith("win")


def _resolve_ffmpeg_exe() -> Optional[str]:
    """Find ffmpeg: env var -> PATH -> imageio-ffmpeg bundled binary."""
    # 1) explicit env var override
    for var in ("FFMPEG_BINARY", "IMAGEIO_FFMPEG_EXE"):
        p = os.environ.get(var)
        if p and os.path.isfile(p):
            return p
    # 2) PATH
    p = shutil.which("ffmpeg")
    if p:
        return p
    # 3) bundled by imageio-ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _get_duration_sec(path: str) -> float:
    """Fast duration without full decode."""
    try:
        import torchaudio  # type: ignore
        info = torchaudio.info(path)
        return float(info.num_frames) / float(info.sample_rate)
    except Exception:
        pass
    try:
        import soundfile as sf  # type: ignore
        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    try:
        import librosa  # type: ignore
        return float(librosa.get_duration(path=path))
    except Exception:
        return 0.0


def _load_via_ffmpeg_cli(path: str, target_sr: int, ffmpeg_exe: Optional[str]) -> np.ndarray:
    import numpy as np
    if not ffmpeg_exe:
        raise RuntimeError("ffmpeg not available")
    cmd = [
        ffmpeg_exe, "-nostdin", "-v", "error", "-i", path,
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(target_sr), "-"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    wav = np.frombuffer(proc.stdout, dtype=np.float32)
    if wav.size == 0:
        raise RuntimeError("ffmpeg produced zero samples")
    max_abs = float(np.max(np.abs(wav))) + 1e-9
    if max_abs > 1.0:
        wav = wav / max_abs
    return wav


def _safe_load_audio(path: str, target_sr: int, prefer_ffmpeg: bool = False) -> np.ndarray:
    """
    Robust loader: (optionally) ffmpeg first -> torchaudio -> soundfile -> librosa -> ffmpeg.
    Returns mono float32 waveform at target_sr.
    """
    errors = []
    ffmpeg_exe = _resolve_ffmpeg_exe()

    # Order of attempts
    attempts = []
    if prefer_ffmpeg:
        attempts.append("ffmpeg")
    attempts.extend(["torchaudio", "soundfile", "librosa"])
    if "ffmpeg" not in attempts:
        attempts.append("ffmpeg")

    for backend in attempts:
        try:
            if backend == "ffmpeg":
                if not ffmpeg_exe:
                    raise RuntimeError("ffmpeg binary not found")
                return _load_via_ffmpeg_cli(path, target_sr, ffmpeg_exe)

            if backend == "torchaudio":
                import torchaudio  # type: ignore
                wav, sr = torchaudio.load(path)
                if wav.ndim == 2:
                    wav = wav.mean(dim=0)
                wav = wav.cpu().numpy()
                if sr != target_sr:
                    import librosa  # type: ignore
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
                max_abs = float(np.max(np.abs(wav))) + 1e-9
                if max_abs > 1.0:
                    wav = wav / max_abs
                return wav.astype(np.float32)

            if backend == "soundfile":
                import soundfile as sf  # type: ignore
                wav, sr = sf.read(path)
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                if sr != target_sr:
                    import librosa  # type: ignore
                    wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
                max_abs = float(np.max(np.abs(wav))) + 1e-9
                if max_abs > 1.0:
                    wav = wav / max_abs
                return wav.astype(np.float32)

            if backend == "librosa":
                import librosa  # type: ignore
                wav, _ = librosa.load(path, sr=target_sr, mono=True)
                max_abs = float(np.max(np.abs(wav))) + 1e-9
                if max_abs > 1.0:
                    wav = wav / max_abs
                return wav.astype(np.float32)

        except Exception as e:
            errors.append(f"{backend}: {e}")

    raise RuntimeError(f"Failed to load audio '{path}'. {' | '.join(errors)}")


def load_asr_datasets(
    model_name: str,
    dataset_name: str,
    dataset_config: Optional[str],
    text_column: str,
    train_split: str,
    eval_split: str,
    cache_dir: Optional[str] = None,
    use_auth_token: bool = False,
) -> tuple[DatasetDict, AutoProcessor]:
    """Returns DatasetDict with 'train' and 'eval' and a compatible AutoProcessor."""
    processor = AutoProcessor.from_pretrained(model_name)
    sampling_rate = processor.feature_extractor.sampling_rate

    token = True if use_auth_token else None
    raw = load_dataset(dataset_name, dataset_config, cache_dir=cache_dir, token=token)

    assert text_column in raw[train_split].column_names, \
        f"text column '{text_column}' not in: {raw[train_split].column_names}"

    # Keep file paths only (no decode here)
    for split in raw.keys():
        raw[split] = raw[split].cast_column("audio", Audio(sampling_rate=sampling_rate, decode=False))

    ds = DatasetDict({"train": raw[train_split], "eval": raw[eval_split]})
    return ds, processor


def prepare_datasets(
    ds: DatasetDict,
    processor: AutoProcessor,
    text_column: str,
    min_duration: float,
    max_duration: float,
    num_proc: int = 4,
    use_augmentations: bool = True,
    noise_dir: Optional[str] = None,
    language_whitelist: bool = True,
) -> DatasetDict:
    """
    Filters by duration (header), decodes robustly (preferring ffmpeg on Windows),
    and *never* crashes on a single bad file: invalid items are marked and filtered out.
    """
    sr = processor.feature_extractor.sampling_rate
    prefer_ffmpeg = IS_WINDOWS  # ffmpeg is the most reliable MP3 decoder on Windows

    # Normalizers
    # Normalizers — CASE-AWARE with respect to tokenizer vocab
    vocab = processor.tokenizer.get_vocab()
    id2tok = {i: t for t, i in vocab.items()}
    single_chars = [tok for i, tok in sorted(id2tok.items(), key=lambda x: x[0]) if len(tok) == 1]

    # Detect whether tokenizer expects lowercase or uppercase alpha
    alpha_chars = [ch for ch in single_chars if ch.isalpha()]
    has_lower = any(ch.islower() for ch in alpha_chars)
    has_upper = any(ch.isupper() for ch in alpha_chars)

    if has_upper and not has_lower:
        # facebook/wav2vec2-base-960h typically falls here: uppercase alphabet
        base_norm = default_english_normalizer(case="upper")
    elif has_lower and not has_upper:
        base_norm = default_english_normalizer(case="lower")
    else:
        # Mixed or unknown: don't force case
        base_norm = default_english_normalizer(case="auto")

    # Keep only characters represented in the tokenizer (plus literal space ' ')
    normalizer = whitelist_normalizer(single_chars, base_norm)


    augmenter = WaveformAugment(AugmentConfig(), noise_dir=noise_dir, sample_rate=sr) if use_augmentations else None

    # --- duration filter without decoding samples ---
    def _duration_filter(batch: Dict[str, Any]) -> bool:
        path = batch["audio"]["path"]
        dur = _get_duration_sec(path)
        return (dur >= min_duration) and (dur <= max_duration)

    # Concurrency: keep modest on Windows
    filter_workers = max(1, min(2, num_proc)) if IS_WINDOWS else max(1, num_proc)
    map_workers = max(1, min(2, num_proc)) if IS_WINDOWS else max(1, num_proc)

    ds = ds.filter(_duration_filter, num_proc=10)

    # --- map: decode + featurize; never raise, mark invalid instead ---
    def _map_fn(batch: Dict[str, Any]) -> Dict[str, Any]:
        path = batch["audio"]["path"]
        try:
            wav = _safe_load_audio(path, target_sr=sr, prefer_ffmpeg=prefer_ffmpeg)
        except Exception as e:
            logger.warning(f"Skipping unreadable audio: {path} ({e})")
            return {"input_values": [0.0], "labels": [processor.tokenizer.pad_token_id], "text": "", "valid": False}

        if augmenter is not None:
            wav = augmenter(wav)

        input_values = processor(wav, sampling_rate=sr).input_values[0]

        # Text normalization -> labels
        raw_text = batch[text_column]
        if not isinstance(raw_text, str):
            raw_text = str(raw_text)
        text = normalizer(raw_text)

        # Skip rows whose normalized text became empty (common if case mismatches vocab)
        if len(text.strip()) == 0:
            logger.warning(f"Skipping empty-label example (after normalization): {path}")
            return {"input_values": [0.0], "labels": [processor.tokenizer.pad_token_id], "text": "", "valid": False}

        labels = processor.tokenizer(text, add_special_tokens=False).input_ids
        if len(labels) == 0:
            logger.warning(f"Skipping zero-length labels (tokenizer) for: {path} | text='{text}'")
            return {"input_values": [0.0], "labels": [processor.tokenizer.pad_token_id], "text": text, "valid": False}

        return {
            "input_values": input_values,
            "labels": labels,
            "text": text,
            "input_length": len(input_values),  # keep if you added length bucketing
            "valid": True,
        }


    # Remove everything except the columns we need to compute; 'valid' is new and kept.
    cols_to_remove = [c for c in ds["train"].column_names if c not in ("audio", text_column)]
    ds = ds.map(_map_fn, remove_columns=cols_to_remove, num_proc=10)

    # Drop invalid rows produced by map
    ds = ds.filter(lambda ex: bool(ex["valid"]), num_proc=10)
    ds = ds.remove_columns(["valid"])

    logger.info(f"Prepared dataset: train={len(ds['train'])}, eval={len(ds['eval'])}")
    return ds
