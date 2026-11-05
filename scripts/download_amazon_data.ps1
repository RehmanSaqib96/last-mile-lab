# Download the Amazon Last Mile training inputs (no AWS account needed).
# Run from the repo root:  .\scripts\download_amazon_data.ps1
$ErrorActionPreference = "Stop"
$Bucket = "s3://amazon-last-mile-challenges/almrrc2021/"
$Dest = "data/raw/almrrc2021"

Write-Host "Folders in the bucket:"
aws s3 ls --no-sign-request $Bucket

Write-Host "`nSize of the training inputs:"
aws s3 ls --no-sign-request --recursive --summarize --human-readable "$($Bucket)almrrc2021-data-training/model_build_inputs/"

$answer = Read-Host "`nDownload the training inputs now? (y/n)"
if ($answer -eq "y") {
    aws s3 sync --no-sign-request "$($Bucket)almrrc2021-data-training/model_build_inputs/" "$Dest/almrrc2021-data-training/model_build_inputs/"
    Write-Host "Done. Files are in $Dest"
}
