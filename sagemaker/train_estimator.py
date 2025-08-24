import argparse, os, time
from pathlib import Path
import sagemaker
from sagemaker.pytorch import PyTorch

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True)                  # IAM role ARN
    p.add_argument("--region", default=os.getenv("AWS_REGION", "us-east-1"))
    p.add_argument("--bucket", required=True)
    p.add_argument("--train_s3", required=True)              # s3://.../train/
    p.add_argument("--eval_s3", default="")                  # s3://.../eval/ (optional)
    p.add_argument("--instance_type", default="ml.g4dn.xlarge")
    p.add_argument("--instance_count", type=int, default=1)
    p.add_argument("--framework_version", default="2.3")
    p.add_argument("--py_version", default="py310")
    p.add_argument("--use_spot", action="store_true")
    return p.parse_args()

def main():
    a = parse()
    sess = sagemaker.Session()
    root = Path(__file__).resolve().parents[1]              # repo root (contains train.py & requirements.txt)
    output_path = f"s3://{a.bucket}/asr/output"
    job_name = f"asr-{int(time.time())}"

    estimator = PyTorch(
        entry_point="train.py",                              # <-- reuse your script
        source_dir=str(root),                                # <-- package repo root
        role=a.role,
        framework_version=a.framework_version,
        py_version=a.py_version,
        instance_type=a.instance_type,
        instance_count=a.instance_count,
        output_path=output_path,
        use_spot_instances=a.use_spot,
        max_run=36*3600,
        max_wait=36*3600 if a.use_spot else None,
        environment={"TOKENIZERS_PARALLELISM": "false"},
        hyperparameters={
            # pass through what your train.py expects
            # (these are examples—you can remove or rename as needed)
            "num_train_epochs": 1,
            "learning_rate": 1e-4,
        },
    )

    inputs = {"train": a.train_s3}
    if a.eval_s3:
        inputs["eval"] = a.eval_s3

    estimator.fit(inputs=inputs, job_name=job_name)
    print("Training job:", job_name)

if __name__ == "__main__":
    main()
