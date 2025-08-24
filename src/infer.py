import argparse
import os
from typing import List, Tuple, Optional
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForCTC
from .utils.log import get_logger

logger = get_logger("infer")

try:
    from pyctcdecode import build_ctcdecoder  # optional
    _HAS_PYCTC = True
except Exception:
    _HAS_PYCTC = False


# ... keep the rest of your file the same ...

def load_audio_file(path: str, target_sr: int):
    import numpy as np
    errors = []
    # 1) torchaudio
    try:
        import torchaudio
        wav, sr = torchaudio.load(path)
        if wav.ndim == 2:
            wav = wav.mean(dim=0)
        wav = wav.cpu().numpy()
        if sr != target_sr:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        # normalize if needed
        max_abs = float(np.max(np.abs(wav))) + 1e-9
        if max_abs > 1.0:
            wav = wav / max_abs
        return wav.astype(np.float32)
    except Exception as e:
        errors.append(f"torchaudio: {e}")

    # 2) soundfile
    try:
        import soundfile as sf
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        max_abs = float(np.max(np.abs(wav))) + 1e-9
        if max_abs > 1.0:
            wav = wav / max_abs
        return wav.astype(np.float32)
    except Exception as e:
        errors.append(f"soundfile: {e}")

    # 3) librosa
    try:
        import librosa
        wav, _ = librosa.load(path, sr=target_sr, mono=True)
        max_abs = float(np.max(np.abs(wav))) + 1e-9
        if max_abs > 1.0:
            wav = wav / max_abs
        return wav.astype(np.float32)
    except Exception as e:
        errors.append(f"librosa: {e}")
        raise RuntimeError(f"Failed to load audio '{path}'. {' | '.join(errors)}")



def collect_audio_paths(root: str) -> List[str]:
    if os.path.isfile(root):
        return [root]
    paths: List[str] = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".wav", ".flac", ".ogg", ".mp3")):
                paths.append(os.path.join(r, f))
    return sorted(paths)


@torch.no_grad()
def decode_with_lm(
    logits: np.ndarray,
    processor: AutoProcessor,
    lm_path: str,
    beam_width: int = 100,
) -> str:
    # build decoder once per process; cache on function attribute
    if not _HAS_PYCTC:
        raise RuntimeError("pyctcdecode not available. Install pyctcdecode and kenlm.")

    if not hasattr(decode_with_lm, "_decoder"):
        # Extract tokens in index order
        vocab = processor.tokenizer.get_vocab()
        id2tok = {i: t for t, i in vocab.items()}
        tokens = [id2tok[i] for i in range(len(id2tok))]
        decode_with_lm._decoder = build_ctcdecoder(
            labels=tokens,
            kenlm_model_path=lm_path
        )
    log_probs = logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)
    text = decode_with_lm._decoder.decode(log_probs, beam_width=beam_width)
    return text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to HF checkpoint directory")
    p.add_argument("--audio_path", type=str, required=True, help="File or folder with audio")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lm_path", type=str, default=None, help="Optional KenLM .arpa or binary")
    p.add_argument("--beam_width", type=int, default=100)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    model = AutoModelForCTC.from_pretrained(args.checkpoint).to(device)
    sr = processor.feature_extractor.sampling_rate

    paths = collect_audio_paths(args.audio_path)
    assert len(paths) > 0, f"No audio found at {args.audio_path}"

    logger.info(f"Loaded {len(paths)} audio files. Batch size: {args.batch_size}")

    batch: List[np.ndarray] = []
    batch_paths: List[str] = []

    for pth in paths:
        wav = load_audio_file(pth, target_sr=sr)
        batch.append(wav)
        batch_paths.append(pth)

        if len(batch) == args.batch_size:
            inputs = processor(batch, sampling_rate=sr, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)

            if args.lm_path:
                # LM decode per item (logits are padded)
                for i in range(len(batch)):
                    T = int((inputs["attention_mask"][i] == 1).sum().item())
                    logit_np = logits[i, :T, :].cpu().numpy()
                    text = decode_with_lm(logit_np, processor, args.lm_path, args.beam_width)
                    print(f"{batch_paths[i]} => {text}")
            else:
                texts = processor.batch_decode(pred_ids)
                for pth_i, txt in zip(batch_paths, texts):
                    print(f"{pth_i} => {txt}")

            batch, batch_paths = [], []

    # trailing partial batch
    if batch:
        inputs = processor(batch, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1)

        if args.lm_path:
            for i in range(len(batch)):
                T = int((inputs["attention_mask"][i] == 1).sum().item())
                logit_np = logits[i, :T, :].cpu().numpy()
                text = decode_with_lm(logit_np, processor, args.lm_path, args.beam_width)
                print(f"{batch_paths[i]} => {text}")
        else:
            texts = processor.batch_decode(pred_ids)
            for pth_i, txt in zip(batch_paths, texts):
                print(f"{pth_i} => {txt}")


if __name__ == "__main__":
    main()
