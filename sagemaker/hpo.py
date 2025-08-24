import argparse, time
from pathlib import Path
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter

p = argparse.ArgumentParser()
p.add_argument("--role", required=True)
p.add_argument("--bucket", required=True)
p.add_argument("--train_s3", required=True)
p.add_argument("--eval_s3", required=True)
args = p.parse_args()

root = Path(__file__).resolve().parents[1]

base = PyTorch(
    entry_point="train.py",
    source_dir=str(root),
    role=args.role,
    framework_version="2.3",
    py_version="py310",
    instance_type="ml.g4dn.xlarge",
    instance_count=1,
    output_path=f"s3://{args.bucket}/asr/output",
    hyperparameters={"num_train_epochs": 1},
)

tuner = HyperparameterTuner(
    base,
    objective_metric_name="eval_wer",
    hyperparameter_ranges={
        "per_device_train_batch_size": IntegerParameter(2, 16),
        "learning_rate": ContinuousParameter(1e-5, 5e-4),
    },
    metric_definitions=[{"Name": "eval_wer", "Regex": "eval_wer\\W*=?([0-9\\.]+)"}],
    max_jobs=6,
    max_parallel_jobs=2,
)

tuner.fit({"train": args.train_s3, "eval": args.eval_s3})
