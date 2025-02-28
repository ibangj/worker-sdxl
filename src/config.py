"""
Shared configuration for SDXL extensions.
Centralizes paths and settings to avoid redundancy.
"""

# Model paths - used in download_models.py, extensions.py, and test_extensions.py
MODEL_PATHS = {
    "controlnet_canny": "/controlnet/control_canny.safetensors",
    "controlnet_depth": "/controlnet/control_depth.safetensors", 
    "reactor_face": "/reactor/inswapper_128.onnx"
}

# Model URLs - used in download_models.py
MODEL_URLS = {
    "controlnet_canny": "https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors",
    "controlnet_depth": "https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors",
    "reactor_face": "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx"
}

# Base directories
DIRS = {
    "controlnet": "/controlnet",
    "reactor": "/reactor"
}

# Model providers - used in extensions.py and test_extensions.py
PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider'] 