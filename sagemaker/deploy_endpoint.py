import argparse
from pathlib import Path
import sagemaker
from sagemaker.pytorch import PyTorchModel

p = argparse.ArgumentParser()
p.add_argument("--role", required=True)
p.add_argument("--model_data", required=True)
p.add_argument("--instance_type", default="ml.g4dn.xlarge")
p.add_argument("--endpoint_name", default=None)
p.add_argument("--framework_version", default="2.3")
p.add_argument("--py_version", default="py310")
args = p.parse_args()

serve_dir = Path(__file__).resolve().parent / "serve"

model = PyTorchModel(
    role=args.role,
    model_data=args.model_data,
    framework_version=args.framework_version,
    py_version=args.py_version,
    entry_point="inference.py",
    source_dir=str(serve_dir)
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type=args.instance_type,
    endpoint_name=args.endpoint_name,
)
print("Endpoint:", predictor.endpoint_name)
