# builder/model_fetcher.py

import torch
import os
import requests
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL


def fetch_pretrained_model(model_class, model_name, **kwargs):
    '''
    Fetches a pretrained model from the HuggingFace model hub.
    '''
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return model_class.from_pretrained(model_name, **kwargs)
        except OSError as err:
            if attempt < max_retries - 1:
                print(
                    f"Error encountered: {err}. Retrying attempt {attempt + 1} of {max_retries}...")
            else:
                raise


def get_diffusion_pipelines():
    '''
    Fetches the Stable Diffusion XL pipelines from the HuggingFace model hub.
    '''
    common_args = {
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "use_safetensors": True
    }

    pipe = fetch_pretrained_model(StableDiffusionXLPipeline,
                                  "stabilityai/stable-diffusion-xl-base-1.0", **common_args)
    vae = fetch_pretrained_model(
        AutoencoderKL, "madebyollin/sdxl-vae-fp16-fix", **{"torch_dtype": torch.float16}
    )
    print("Loaded VAE")
    refiner = fetch_pretrained_model(StableDiffusionXLImg2ImgPipeline,
                                     "stabilityai/stable-diffusion-xl-refiner-1.0", **common_args)

    return pipe, refiner, vae


def download_additional_models():
    '''
    Downloads additional models needed for the worker:
    - ControlNet models (canny, depth)
    - Face swap model (inswapper_128.onnx)
    '''
    # Create necessary directories
    os.makedirs("/controlnet", exist_ok=True)
    os.makedirs("/models/face", exist_ok=True)

    # URLs for models
    MODELS = {
        "/controlnet/control_canny.safetensors": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.safetensors",
        "/controlnet/control_depth.safetensors": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.safetensors",
        "/models/face/inswapper_128.onnx": "https://huggingface.co/netrunner-exe/Insight-Swap-models/resolve/main/inswapper_128.fp16.onnx"
    }

    # Download each model
    for path, url in MODELS.items():
        if os.path.exists(path):
            print(f"✅ {path} already exists, skipping download")
            continue
            
        print(f"⬇️ Downloading {url} to {path}")
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(path, 'wb') as file, tqdm(
            desc=path,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
                
        print(f"✅ Successfully downloaded {path}")

    print("All additional models downloaded successfully!")


if __name__ == "__main__":
    # First get the diffusion pipelines
    get_diffusion_pipelines()
    
    # Then download additional models
    download_additional_models()
