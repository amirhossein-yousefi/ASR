from dataclasses import dataclass
from typing import Any, Dict, List, Union


@dataclass
class DataCollatorCTCWithPadding:
    """
    Pads batch dynamically for Wav2Vec2-CTC.
    Expects each feature to have:
      - "input_values": List[float]
      - "labels": List[int]
    """
    processor: Any
    padding: Union[bool, str] = "longest"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Inputs: use processor.pad (feature extractor path)
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        batch["labels"] = labels
        return batch
