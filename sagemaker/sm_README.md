# SageMaker: training + inference for ASR

## Train (reuses ./train.py)
python sagemaker/train_estimator.py \
  --role <ROLE_ARN> \
  --bucket <BUCKET> \
  --train_s3 s3://<BUCKET>/asr/train/ \
  --eval_s3  s3://<BUCKET>/asr/eval/ \
  --use_spot

## Deploy endpoint
python sagemaker/deploy_endpoint.py \
  --role <ROLE_ARN> \
  --model_data s3://<BUCKET>/asr/output/<job>/output/model.tar.gz

## Invoke
aws sagemaker-runtime invoke-endpoint \
  --endpoint-name <name> \
  --content-type application/json \
  --body '{"base64":"<...>","sample_rate":16000}' 

### Notes
- SageMaker exposes data channels via `SM_CHANNEL_*` and expects artifacts in `SM_MODEL_DIR`. :contentReference[oaicite:12]{index=12}
- `requirements.txt` in the training `source_dir` is installed at training and bundled under `code/` in the model artifact so the hosting container can install it. :contentReference[oaicite:13]{index=13}
- The PyTorch model server uses `model_fn`, `input_fn`, `predict_fn`, `output_fn`. :contentReference[oaicite:14]{index=14}
