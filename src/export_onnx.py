import argparse
import os
import torch
from transformers import AutoModelForCTC, AutoProcessor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--onnx_path", type=str, required=True)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    model = AutoModelForCTC.from_pretrained(args.checkpoint)
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    sr = processor.feature_extractor.sampling_rate

    model.eval()

    # Dummy input: 5s of silence at sampling rate
    dummy = torch.zeros(1, int(sr * 5.0), dtype=torch.float32)

    # Prepare dynamic axes for variable length
    dynamic_axes = {
        "input_values": {0: "batch", 1: "time"},
        "logits": {0: "batch", 1: "time"},
    }

    os.makedirs(os.path.dirname(args.onnx_path), exist_ok=True)

    torch.onnx.export(
        model,
        (dummy,),
        args.onnx_path,
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported ONNX to {args.onnx_path}")


if __name__ == "__main__":
    main()
