import argparse
import os
from typing import Dict, List

import numpy as np
import torch
from transformers import TrainingArguments, Trainer, AutoProcessor
from transformers.trainer_callback import EarlyStoppingCallback
from datasets import DatasetDict

from src.config import TrainConfig
from src.utils.seed import set_seed
from src.utils.log import get_logger
from src.utils.metrics import compute_wer_cer
from src.data.collator import DataCollatorCTCWithPadding
from src.data.datamodule import load_asr_datasets, prepare_datasets
from src.models.modeling import load_ctc_model

logger = get_logger("train")


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    # Model & data
    p.add_argument("--model_name", type=str, default="facebook/wav2vec2-base-960h")
    p.add_argument("--dataset_name", type=str, default="mozilla-foundation/common_voice_17_0")
    p.add_argument("--dataset_config", type=str, default="en")
    p.add_argument("--text_column", type=str, default="sentence")
    p.add_argument("--train_split", type=str, default="train")
    p.add_argument("--eval_split", type=str, default="validation")
    p.add_argument("--cache_dir", type=str, default=None)
    # p.add_argument("--output_dir", type=str, default="./outputs/asr")
    p.add_argument("--output_dir", default=os.environ.get("SM_MODEL_DIR", "./outputs/asr"))
    p.add_argument("--train_dir", default=os.environ.get("SM_CHANNEL_TRAIN"))
    p.add_argument("--eval_dir", default=os.environ.get("SM_CHANNEL_EVAL"))

    # Duration constraints
    p.add_argument("--max_duration", type=float, default=18.0)
    p.add_argument("--min_duration", type=float, default=1.0)

    # Optimization
    p.add_argument("--learning_rate", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    # Precision & memory
    p.add_argument("--fp16", type=lambda x: str(x).lower() == "true", default=True)
    p.add_argument("--bf16", type=lambda x: str(x).lower() == "true", default=False)
    p.add_argument("--gradient_checkpointing", type=lambda x: str(x).lower() == "true", default=False)
    p.add_argument("--freeze_feature_encoder", type=lambda x: str(x).lower() == "true", default=False)

    # Logging
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    # Data processing
    p.add_argument("--use_augmentations", type=lambda x: str(x).lower() == "true", default=False)
    p.add_argument("--noise_dir", type=str, default=None)
    p.add_argument("--num_proc", type=int, default=10)

    # Optional limits
    p.add_argument("--max_train_samples", type=int, default=50000)
    p.add_argument("--max_eval_samples", type=int, default=None)

    # HuggingFace auth
    p.add_argument("--use_auth_token", type=lambda x: str(x).lower() == "true", default=False)

    cfg = TrainConfig(**vars(p.parse_args()))
    return cfg
import numpy as np
import torch
from transformers.trainer_callback import TrainerCallback

def preprocess_logits_for_metrics(logits, labels):
    # Keep only argmax IDs to avoid storing full logits in memory
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    pred_ids = torch.argmax(logits, dim=-1)  # (batch, time)
    return pred_ids  # Trainer will move to CPU/NumPy

def _to_2d_int_list(x):
    """
    Convert nested predictions/labels into List[List[int]].
    Handles torch.Tensor, np.ndarray (incl. dtype=object), lists of arrays/lists, etc.
    """
    # -> numpy
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            out = []
            for elem in x:
                out.extend(_to_2d_int_list(elem))
            return out
        if x.ndim == 1:
            return [x.astype(int).tolist()]
        if x.ndim >= 2:
            # flatten all leading dims except last
            x = x.reshape(-1, x.shape[-1])
            return [row.astype(int).tolist() for row in x]
    if isinstance(x, (list, tuple)):
        # list could be: [array(BxT), array(BxT2), ...] or [list(T), ...]
        out = []
        for elem in x:
            rows = _to_2d_int_list(elem)
            # rows is always List[List[int]]
            out.extend(rows)
        return out
    # scalar fallback
    return [[int(x)]]

class CudaEmptyCacheCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()



def main():
    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")  # enables TF32 matmuls
        except Exception:
            pass
    cfg = parse_args()
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Load data + processor
    ds, processor = load_asr_datasets(
        model_name=cfg.model_name,
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        text_column=cfg.text_column,
        train_split=cfg.train_split,
        eval_split=cfg.eval_split,
        cache_dir=cfg.cache_dir,
    )
    if cfg.max_train_samples:
        ds["train"] = ds["train"].select(range(min(cfg.max_train_samples, len(ds["train"]))))
    if cfg.max_eval_samples:
        ds["eval"] = ds["eval"].select(range(min(cfg.max_eval_samples, len(ds["eval"]))))
    ds = prepare_datasets(
        ds,
        processor,
        text_column=cfg.text_column,
        min_duration=cfg.min_duration,
        max_duration=cfg.max_duration,
        num_proc=cfg.num_proc,
        use_augmentations=cfg.use_augmentations,
        noise_dir=cfg.noise_dir,
    )

    if cfg.max_train_samples:
        ds["train"] = ds["train"].select(range(min(cfg.max_train_samples, len(ds["train"]))))
    if cfg.max_eval_samples:
        ds["eval"] = ds["eval"].select(range(min(cfg.max_eval_samples, len(ds["eval"]))))

    # Model
    model = load_ctc_model(
        model_name=cfg.model_name,
        freeze_feature_encoder=cfg.freeze_feature_encoder,
        gradient_checkpointing=cfg.gradient_checkpointing,
    )

    # Collator
    collator = DataCollatorCTCWithPadding(processor=processor)

    # Metrics fn
    def compute_metrics(eval_pred):
        # Support both tuple-style and EvalPrediction object
        try:
            pred_ids, labels = eval_pred
        except Exception:
            pred_ids, labels = eval_pred.predictions, eval_pred.label_ids

        # Coerce to List[List[int]]
        pred_ids_2d = _to_2d_int_list(pred_ids)
        labels_2d = _to_2d_int_list(labels)

        # Replace -100 with pad token id for decoding
        pad_id = processor.tokenizer.pad_token_id
        labels_2d = [[(pad_id if int(t) == -100 else int(t)) for t in row] for row in labels_2d]

        # Decode
        pred_str = processor.batch_decode(pred_ids_2d, skip_special_tokens=True)
        try:
            label_str = processor.batch_decode(labels_2d, group_tokens=False, skip_special_tokens=True)
        except TypeError:
            label_str = processor.batch_decode(labels_2d, skip_special_tokens=True)

        return compute_wer_cer(pred_str, label_str)

    has_len_col = "input_length" in ds["train"].column_names

    # Training args
    ta_kwargs  = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=min(cfg.per_device_eval_batch_size, 8),
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        eval_strategy="steps",
        save_strategy="steps",
        logging_steps=cfg.logging_steps,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        num_train_epochs=cfg.num_train_epochs,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        warmup_ratio=cfg.warmup_ratio,
        gradient_checkpointing=cfg.gradient_checkpointing,
        dataloader_num_workers=0,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        # group_by_length=True,
        # dataloader_persistent_workers = False,
        skip_memory_metrics=True,
        eval_accumulation_steps=1,
        remove_unused_columns=False,
        logging_dir=os.path.join(cfg.output_dir, "logs"),
        report_to=['tensorboard'],
        max_grad_norm=cfg.max_grad_norm,
        optim="adamw_torch_fused",
    )

    args = ta_kwargs
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["eval"],
        tokenizer=processor,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), CudaEmptyCacheCallback()],
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete. Saving model...")
    trainer.save_model(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)
    logger.info(f"Saved to {cfg.output_dir}")


if __name__ == "__main__":
    main()
