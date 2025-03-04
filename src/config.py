"""
Shared configuration for SDXL extensions.
Centralizes paths and settings to avoid redundancy.
"""

# Model paths - used in download_models.py, extensions.py, and test_extensions.py
MODEL_PATHS = {
    "controlnet_canny": "/controlnet/control_canny.safetensors",
    "controlnet_depth": "/controlnet/control_depth.safetensors", 
    "face_swap": "/models/face/inswapper_128.onnx"
}

# Model URLs - used in download_models.py
MODEL_URLS = {
    "controlnet_canny": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.safetensors",
    "controlnet_depth": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.safetensors",
    "face_swap": "https://huggingface.co/netrunner-exe/Insight-Swap-models/resolve/main/inswapper_128.fp16.onnx"
}

# Base directories
DIRS = {
    "controlnet": "/controlnet",
    "face": "/models/face"
}

# Model providers - used in extensions.py and test_extensions.py
PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider'] 