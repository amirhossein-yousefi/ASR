import argparse
from typing import List
import torch
from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModelForCTC

from src.utils.log import get_logger
from src.utils.metrics import compute_wer_cer
from src.utils.text_normalization import default_english_normalizer

logger = get_logger("evaluate")


def _safe_load_audio(path: str, target_sr: int):
    import numpy as np
    errors = []
    try:
        import torchaudio
        wav, sr = torchaudio.load(path)
        if wav.ndim == 2:
            wav = wav.mean(dim=0)
        wav = wav.cpu().numpy()
        if sr != target_sr:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        return wav.astype(np.float32)
    except Exception as e:
        errors.append(f"torchaudio: {e}")
    try:
        import soundfile as sf
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        return wav.astype(np.float32)
    except Exception as e:
        errors.append(f"soundfile: {e}")
    try:
        import librosa
        wav, sr = librosa.load(path, sr=target_sr, mono=True)
        return wav.astype(np.float32)
    except Exception as e:
        errors.append(f"librosa: {e}")
        raise RuntimeError(f"Failed to load audio '{path}'. {' | '.join(errors)}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset_name", type=str, default="mozilla-foundation/common_voice_17_0")
    p.add_argument("--dataset_config", type=str, default="en")
    p.add_argument("--text_column", type=str, default="sentence")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max_samples", type=int, default=None)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    model = AutoModelForCTC.from_pretrained(args.checkpoint).to(device)
    sr = processor.feature_extractor.sampling_rate

    # Do not decode in datasets; we will load by path.
    ds = load_dataset(args.dataset_name, args.dataset_config)
    assert args.text_column in ds[args.split].column_names
    ds[args.split] = ds[args.split].cast_column("audio", Audio(sampling_rate=sr, decode=False))
    dset = ds[args.split]
    if args.max_samples:
        dset = dset.select(range(min(args.max_samples, len(dset))))

    norm = default_english_normalizer()
    preds: List[str] = []
    refs: List[str] = []

    for ex in dset:
        path = ex["audio"]["path"]
        wav = _safe_load_audio(path, target_sr=sr)
        inputs = processor(wav, sampling_rate=sr, return_tensors="pt")
        logits = model(**{k: v.to(device) for k, v in inputs.items()}).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

        ref = ex[args.text_column]
        ref = norm(ref) if isinstance(ref, str) else str(ref)

        preds.append(pred_str)
        refs.append(ref)

    metrics = compute_wer_cer(preds, refs)
    logger.info(f"Eval on split='{args.split}': {metrics}")
    print(metrics)


if __name__ == "__main__":
    main()
