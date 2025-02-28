#!/usr/bin/env python3
import os
import sys
import requests
import time
from tqdm import tqdm

# Model URLs
MODELS = {
    "controlnet_canny": {
        "url": "https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors",
        "path": "/controlnet/control_canny.safetensors"
    },
    "controlnet_depth": {
        "url": "https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors",
        "path": "/controlnet/control_depth.safetensors"
    },
    "reactor_face": {
        "url": "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
        "path": "/reactor/inswapper_128.onnx"
    }
}

def verify_url(url):
    """Verify that a URL exists and is accessible"""
    try:
        response = requests.head(url, timeout=10)
        if response.status_code >= 400:
            return False
        return True
    except requests.exceptions.RequestException:
        return False

def download_file(url, destination):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        print(f"Downloading: {os.path.basename(destination)}")
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        
        with open(destination, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        
        progress_bar.close()
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        # Remove partially downloaded file
        if os.path.exists(destination):
            os.remove(destination)
        return False

def main():
    """Main function to download models"""
    print("🔍 Verifying and downloading models...")
    success = True
    
    for model_name, model_info in MODELS.items():
        url = model_info["url"]
        path = model_info["path"]
        
        # Skip if model already exists
        if os.path.exists(path):
            print(f"✅ {model_name} already exists at {path}")
            continue
        
        # Verify URL before downloading
        print(f"🔍 Verifying URL for {model_name}...")
        if not verify_url(url):
            print(f"❌ URL verification failed for {model_name}: {url}")
            success = False
            continue
        
        # Download the model
        print(f"⬇️ Downloading {model_name}...")
        if not download_file(url, path):
            print(f"❌ Failed to download {model_name}")
            success = False
            continue
        
        print(f"✅ Successfully downloaded {model_name}")
    
    if success:
        print("✅ All models verified and downloaded successfully!")
        return 0
    else:
        print("⚠️ Some models failed to download. Check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 