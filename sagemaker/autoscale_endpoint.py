import argparse, boto3
p = argparse.ArgumentParser()
p.add_argument("--endpoint_name", required=True)
p.add_argument("--min_capacity", type=int, default=1)
p.add_argument("--max_capacity", type=int, default=2)
p.add_argument("--target_value", type=float, default=60.0)  # Invocations/instance target
p.add_argument("--region", default="us-east-1")
a = p.parse_args()

aas = boto3.client("application-autoscaling", region_name=a.region)
resource_id = f"endpoint/{a.endpoint_name}/variant/AllTraffic"
aas.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId=resource_id,
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=a.min_capacity,
    MaxCapacity=a.max_capacity,
)
aas.put_scaling_policy(
    PolicyName="sagemaker-target-invocations",
    ServiceNamespace="sagemaker",
    ResourceId=resource_id,
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": a.target_value,
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
        },
        "ScaleInCooldown": 120,
        "ScaleOutCooldown": 60,
    },
)
print("Autoscaling policy attached.")
