#!/bin/bash
# Script to download models and upload them to S3

# Check if bucket name was provided
if [ -z "$1" ]; then
    echo "Usage: $0 <s3_bucket_name>"
    echo "Example: $0 my-sdxl-models-bucket"
    exit 1
fi

BUCKET_NAME=$1

# Install dependencies
echo "Installing dependencies..."
pip install -r builder/s3_upload_requirements.txt

# Create temp directories
mkdir -p controlnet reactor

# Run the upload script
echo "Running upload script to download models and upload to S3..."
python builder/upload_models_to_s3.py --bucket "$BUCKET_NAME"

echo "Setup complete! Models should now be available in your S3 bucket: $BUCKET_NAME"
echo "The GitHub Actions workflow will now be able to download these models during build." 