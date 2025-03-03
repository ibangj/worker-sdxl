# Setting Up Models in S3

This guide explains how to set up your S3 bucket with the necessary models for the SDXL worker.

## Prerequisites

1. **AWS CLI**: Make sure you have the AWS CLI installed and configured with appropriate credentials.
   ```
   # To install AWS CLI
   pip install awscli
   
   # To configure your credentials
   aws configure
   ```

2. **S3 Bucket**: Create an S3 bucket to store your models.
   ```
   aws s3 mb s3://your-bucket-name
   ```

3. **Python 3.8+**: Make sure you have Python 3.8 or higher installed.

## Setting Up Your S3 Bucket with Models

### Option 1: Using the Automated Script (Recommended)

1. Run the setup script with your bucket name:
   
   **On Windows:**
   ```
   setup_s3_models.bat your-bucket-name
   ```

   **On Linux/Mac:**
   ```
   ./setup_s3_models.sh your-bucket-name
   ```

2. The script will:
   - Install the required dependencies
   - Download the ControlNet models from Hugging Face
   - Download the Reactor models from their sources
   - Upload everything to your S3 bucket with the correct structure

3. Wait for the process to complete. This may take some time depending on your internet connection, as the models are large files.

### Option 2: Manual Setup

If you prefer to manually upload the models:

1. Download the models you need from their respective sources (ControlNet from Hugging Face, Reactor from appropriate sources).

2. Create the following structure in your S3 bucket:
   ```
   s3://your-bucket-name/
   ├── models/
   │   ├── controlnet/
   │   │   ├── control_v11p_sd15_canny.pth
   │   │   ├── control_v11p_sd15_openpose.pth
   │   │   └── control_v11p_sd15_depth.pth
   │   └── reactor/
   │       ├── ip-adapter-plus_sd15.safetensors
   │       └── ip-adapter-full-face_sd15.safetensors
   ```

3. Upload the models to your S3 bucket:
   ```
   aws s3 cp /path/to/local/controlnet/ s3://your-bucket-name/models/controlnet/ --recursive
   aws s3 cp /path/to/local/reactor/ s3://your-bucket-name/models/reactor/ --recursive
   ```

## Customizing the Models

If you need different models, you can modify the `CONTROLNET_MODELS` and `REACTOR_MODELS` lists in the `builder/upload_models_to_s3.py` script.

## GitHub Actions Configuration

After setting up your S3 bucket, make sure the following secrets are configured in your GitHub repository:

- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
- `AWS_REGION`: The AWS region where your bucket is located
- `AWS_S3_BUCKET`: The name of your S3 bucket

With these set up, your GitHub Actions workflow will be able to download the models from S3 during the build process instead of downloading them directly, which will help avoid disk space issues. 