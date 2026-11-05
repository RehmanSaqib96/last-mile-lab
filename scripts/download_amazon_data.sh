#!/usr/bin/env bash
# Mac/Linux version of download_amazon_data.ps1
set -euo pipefail
BUCKET="s3://amazon-last-mile-challenges/almrrc2021/"
DEST="data/raw/almrrc2021"
aws s3 ls --no-sign-request "$BUCKET"
aws s3 ls --no-sign-request --recursive --summarize --human-readable \
  "${BUCKET}almrrc2021-data-training/model_build_inputs/"
aws s3 sync --no-sign-request "${BUCKET}almrrc2021-data-training/model_build_inputs/" \
  "$DEST/almrrc2021-data-training/model_build_inputs/"
