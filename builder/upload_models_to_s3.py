#!/usr/bin/env python3
"""
Script to download ControlNet and Reactor models and upload them to S3.
This is a one-time setup script to prepare your S3 bucket with the needed models.
"""

import os
import boto3
import argparse
import requests
import zipfile
import shutil
import time
from pathlib import Path
from huggingface_hub import hf_hub_download

# Hugging Face models to download
CONTROLNET_MODELS = [
    # ControlNet models - add or modify based on your needs
    {"repo_id": "lllyasviel/ControlNet-v1-1", "filename": "control_v11p_sd15_canny.pth"},
    {"repo_id": "lllyasviel/ControlNet-v1-1", "filename": "control_v11p_sd15_openpose.pth"},
    {"repo_id": "lllyasviel/ControlNet-v1-1", "filename": "control_v11p_sd15_depth.pth"},
]

# Reactor models - add URLs for any Reactor models you need
REACTOR_MODELS = [
    # Example URL - replace with actual URLs to the models you need
    {"name": "ip-adapter-plus_sd15.safetensors", "url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors"},
    {"name": "ip-adapter-full-face_sd15.safetensors", "url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-full-face_sd15.safetensors"},
]

def download_hf_model(repo_id, filename, dest_dir):
    """Download a model from Hugging Face Hub."""
    print(f"Downloading {filename} from {repo_id}...")
    os.makedirs(dest_dir, exist_ok=True)
    
    try:
        file_path = hf_hub_download(repo_id=repo_id, filename=filename)
        dest_path = os.path.join(dest_dir, os.path.basename(filename))
        shutil.copy(file_path, dest_path)
        print(f"Successfully downloaded {filename} to {dest_path}")
        return dest_path
    except Exception as e:
        print(f"Error downloading {filename} from {repo_id}: {e}")
        return None

def download_direct_model(url, name, dest_dir):
    """Download a model directly from a URL."""
    print(f"Downloading {name} from {url}...")
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, name)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024  # 1MB
        
        with open(dest_path, 'wb') as f:
            downloaded = 0
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded += len(data)
                print(f"\rDownloading: {downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB ({100*downloaded/total_size:.1f}%)", end="")
        
        print(f"\nSuccessfully downloaded {name} to {dest_path}")
        return dest_path
    except Exception as e:
        print(f"Error downloading {name}: {e}")
        return None

def upload_to_s3(file_path, bucket, s3_key):
    """Upload a file to S3."""
    print(f"Uploading {file_path} to s3://{bucket}/{s3_key}...")
    s3_client = boto3.client('s3')
    try:
        with open(file_path, "rb") as f:
            # Get file size for progress reporting
            file_size = os.path.getsize(file_path)
            
            # Create a callback to track upload progress
            class ProgressPercentage:
                def __init__(self, filename, size):
                    self._filename = filename
                    self._size = size
                    self._seen_so_far = 0
                    self._start_time = time.time()
                
                def __call__(self, bytes_amount):
                    self._seen_so_far += bytes_amount
                    percentage = (self._seen_so_far / self._size) * 100
                    elapsed_time = time.time() - self._start_time
                    speed = self._seen_so_far / (elapsed_time * 1024 * 1024)
                    print(f"\rUploading {self._filename}: {self._seen_so_far/1024/1024:.1f}MB / {self._size/1024/1024:.1f}MB "
                          f"({percentage:.1f}%) at {speed:.2f}MB/s", end="")
            
            # Upload with progress tracking
            s3_client.upload_fileobj(
                f, 
                bucket, 
                s3_key,
                Callback=ProgressPercentage(os.path.basename(file_path), file_size)
            )
            
        print(f"\nSuccessfully uploaded {file_path} to s3://{bucket}/{s3_key}")
        return True
    except Exception as e:
        print(f"Error uploading {file_path} to S3: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download models and upload to S3")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--controlnet-dir", default="controlnet", help="Directory to store ControlNet models")
    parser.add_argument("--reactor-dir", default="reactor", help="Directory to store Reactor models")
    args = parser.parse_args()
    
    # Create temporary directories
    os.makedirs(args.controlnet_dir, exist_ok=True)
    os.makedirs(args.reactor_dir, exist_ok=True)
    
    # 1. Download ControlNet models
    for model in CONTROLNET_MODELS:
        model_path = download_hf_model(
            model["repo_id"], 
            model["filename"], 
            args.controlnet_dir
        )
        if model_path:
            # Upload to S3
            s3_key = f"models/controlnet/{os.path.basename(model_path)}"
            upload_to_s3(model_path, args.bucket, s3_key)
    
    # 2. Download Reactor models
    for model in REACTOR_MODELS:
        model_path = download_direct_model(
            model["url"], 
            model["name"], 
            args.reactor_dir
        )
        if model_path:
            # Upload to S3
            s3_key = f"models/reactor/{os.path.basename(model_path)}"
            upload_to_s3(model_path, args.bucket, s3_key)
    
    print("Finished uploading models to S3")

if __name__ == "__main__":
    main() 