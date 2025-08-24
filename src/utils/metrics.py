from typing import Dict, List
import numpy as np
import evaluate

# evaluate uses jiwer under the hood
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def compute_wer_cer(preds: List[str], refs: List[str]) -> Dict[str, float]:
    return {
        "wer": float(wer_metric.compute(predictions=preds, references=refs)),
        "cer": float(cer_metric.compute(predictions=preds, references=refs)),
    }
