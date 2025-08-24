from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    # Model & data
    model_name: str = "facebook/wav2vec2-base-960h"
    dataset_name: str = "mozilla-foundation/common_voice_17_0"
    dataset_config: str = "en"
    text_column: str = "sentence"
    train_split: str = "train"
    eval_split: str = "validation"
    cache_dir: Optional[str] = None
    output_dir: str = "./outputs/asr"

    # Audio length constraints (seconds)
    max_duration: float = 18.0
    min_duration: float = 1.0

    # Optimization
    learning_rate: float = 3e-5
    weight_decay: float = 0.0
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0

    # Mixed precision / memory
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    freeze_feature_encoder: bool = False

    # Logging
    logging_steps: int = 50
    eval_steps: int = 1000
    save_steps: int = 1000
    save_total_limit: int = 2
    seed: int = 42

    # Data processing
    use_augmentations: bool = True
    noise_dir: Optional[str] = None   # folder with wav noise files (optional)
    num_proc: int = 1                 # multiprocessing for dataset.map

    # Optional LM decoding in inference & evaluation
    lm_path: Optional[str] = None     # path to ARPA or binary kenlm
    beam_width: int = 100

    # Samples limit (for quick dry runs)
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None

    # Hugging Face auth (if needed for gated models)
    use_auth_token: bool = False
