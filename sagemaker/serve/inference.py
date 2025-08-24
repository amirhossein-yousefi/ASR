import base64, io, json, os
import torch
import torchaudio
from transformers import AutoModelForCTC, AutoProcessor

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_SR = 16000

def model_fn(model_dir):
    # works for wav2vec2/whisper-ctc style; adjust to your model label if needed
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForCTC.from_pretrained(model_dir)
    model.to(_DEVICE).eval()
    return {"model": model, "processor": processor}

def _load_audio_from_base64(b64: str, sr: int = _SR):
    wav = base64.b64decode(b64)
    wav_tensor, in_sr = torchaudio.load(io.BytesIO(wav))
    if in_sr != sr:
        wav_tensor = torchaudio.functional.resample(wav_tensor, in_sr, sr)
    return wav_tensor.squeeze(0), sr

def input_fn(request_body, content_type):
    if content_type == "application/json":
        p = json.loads(request_body)
        if "base64" in p:
            return _load_audio_from_base64(p["base64"], p.get("sample_rate", _SR))
        elif "array" in p:
            arr = torch.tensor(p["array"], dtype=torch.float32)
            sr = int(p.get("sample_rate", _SR))
            if sr != _SR:
                arr = torchaudio.functional.resample(arr.unsqueeze(0), sr, _SR).squeeze(0)
            return arr, _SR
    elif content_type in ("audio/wav", "audio/x-wav"):
        wav_tensor, sr = torchaudio.load(io.BytesIO(request_body))
        if sr != _SR:
            wav_tensor = torchaudio.functional.resample(wav_tensor, sr, _SR)
        return wav_tensor.squeeze(0), _SR
    raise ValueError(f"Unsupported content_type: {content_type}")

def predict_fn(data, model_bundle):
    waveform, sr = data
    processor = model_bundle["processor"]
    model = model_bundle["model"]
    inputs = processor(waveform.numpy(), sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(pred_ids.cpu().numpy())[0]
    return {"text": text}

def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
