from typing import Optional
from transformers import AutoModelForCTC
from ..utils.log import get_logger

logger = get_logger("model")


def load_ctc_model(
    model_name: str,
    freeze_feature_encoder: bool = True,
    gradient_checkpointing: bool = True,
) -> AutoModelForCTC:
    model = AutoModelForCTC.from_pretrained(model_name)
    if freeze_feature_encoder and hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
        logger.info("Feature encoder frozen.")
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")
    return model
