"""
Extensions module for SDXL adding ControlNet and ReActor functionality
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
import requests
from io import BytesIO
import base64
from insightface.app import FaceAnalysis
import onnxruntime

# Import shared configuration
from config import MODEL_PATHS, PROVIDERS

# Initialize global variables for models
controlnet_models = {}
face_analyzer = None
face_swapper = None

def init_extensions():
    """Initialize ControlNet and ReActor models"""
    global controlnet_models, face_analyzer, face_swapper
    
    # Load ControlNet models
    try:
        controlnet_canny = ControlNetModel.from_pretrained(
            MODEL_PATHS["controlnet_canny"], torch_dtype=torch.float16
        )
        controlnet_depth = ControlNetModel.from_pretrained(
            MODEL_PATHS["controlnet_depth"], torch_dtype=torch.float16
        )
        
        controlnet_models = {
            "canny": controlnet_canny,
            "depth": controlnet_depth,
            # Add other models as needed
        }
        print("✅ ControlNet models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading ControlNet models: {e}")
    
    # Initialize face analyzer and swapper for ReActor
    try:
        face_analyzer = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)
        face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        
        face_swapper = onnxruntime.InferenceSession(
            MODEL_PATHS["face_swap"], 
            providers=PROVIDERS
        )
        print("✅ ReActor models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading ReActor models: {e}")

def download_image(image_url_or_base64):
    """Download or decode an image from URL or base64"""
    try:
        if image_url_or_base64.startswith('http'):
            # It's a URL
            response = requests.get(image_url_or_base64, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        elif image_url_or_base64.startswith('data:image'):
            # It's base64
            base64_data = image_url_or_base64.split(',')[1]
            image_data = base64.b64decode(base64_data)
            return Image.open(BytesIO(image_data)).convert("RGB")
        else:
            raise ValueError("Image must be a URL or base64 data")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error downloading image: {str(e)}")
    except base64.binascii.Error as e:
        raise ValueError(f"Error decoding base64 image: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def preprocess_image(image, control_type):
    """Preprocess image for ControlNet"""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if control_type == "canny":
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
    elif control_type == "depth":
        # Use a depth estimator or preprocess accordingly
        # This is a placeholder
        pass
    
    return Image.fromarray(image)

def apply_reactor(image, source_image, target_faces_index=None):
    """Apply ReActor face swap"""
    global face_analyzer, face_swapper
    
    if face_analyzer is None or face_swapper is None:
        return image, "ReActor models not loaded"
    
    # Convert PIL image to numpy array if needed
    if isinstance(image, Image.Image):
        target_img = np.array(image)
    else:
        target_img = image
    
    if isinstance(source_image, Image.Image):
        source_img = np.array(source_image)
    else:
        source_img = source_image
    
    # Get source face
    source_faces = face_analyzer.get(source_img)
    if len(source_faces) == 0:
        return image, "No source face found"
    
    source_face = source_faces[0]
    
    # Get target faces
    target_faces = face_analyzer.get(target_img)
    if len(target_faces) == 0:
        return image, "No target face found"
    
    # Filter target faces if index provided
    if target_faces_index is not None:
        if isinstance(target_faces_index, int) and 0 <= target_faces_index < len(target_faces):
            target_faces = [target_faces[target_faces_index]]
        elif isinstance(target_faces_index, list):
            target_faces = [face for i, face in enumerate(target_faces) if i in target_faces_index]
    
    # Apply face swap
    result_image = target_img.copy()
    for target_face in target_faces:
        result_image = face_swapper.run(None, {
            'source': source_face,
            'target': target_face,
            'image': result_image
        })[0]
    
    return Image.fromarray(result_image), "Success"

def process_with_controlnet(base_model, prompt, negative_prompt, control_image, control_type, 
                           width, height, num_inference_steps, guidance_scale, 
                           controlnet_conditioning_scale=0.5, seed=None):
    """Process an image with ControlNet"""
    global controlnet_models
    
    if not controlnet_models:
        return None, "ControlNet models not loaded"
    
    selected_controlnet = controlnet_models.get(control_type)
    if not selected_controlnet:
        return None, f"ControlNet type '{control_type}' not found"
    
    # Create generator for reproducibility
    if seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        generator = None
    
    # Process control image
    try:
        if isinstance(control_image, str):
            control_image = download_image(control_image)
        
        # Preprocess for specific control type
        processed_image = preprocess_image(control_image, control_type)
        
        # Create specific pipeline for ControlNet
        controlnet_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model.config.name_or_path,
            controlnet=selected_controlnet,
            torch_dtype=torch.float16,
            vae=base_model.vae
        )
        controlnet_pipe.to("cuda")
        controlnet_pipe.enable_xformers_memory_efficient_attention()
        
        # Generate image
        result = controlnet_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=processed_image,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator
        )
        
        return result.images, "Success"
    
    except Exception as e:
        return None, f"Error processing with ControlNet: {e}"

# Extensions schema to be merged with rp_schemas
EXTENSIONS_SCHEMA = {
    # ControlNet parameters
    "use_controlnet": {
        "type": bool,
        "required": False,
        "default": False,
        "description": "Whether to use ControlNet"
    },
    "control_type": {
        "type": str,
        "required": False,
        "default": "canny",
        "description": "Type of ControlNet to use (canny, depth, etc.)"
    },
    "control_image": {
        "type": str,
        "required": False,
        "default": None,
        "description": "Image for ControlNet conditioning (URL or base64)"
    },
    "controlnet_conditioning_scale": {
        "type": float,
        "required": False,
        "default": 0.5,
        "description": "Conditioning scale for ControlNet"
    },
    # ReActor parameters
    "use_reactor": {
        "type": bool,
        "required": False,
        "default": False,
        "description": "Whether to use ReActor face swap"
    },
    "source_face_image": {
        "type": str,
        "required": False,
        "default": None,
        "description": "Source face image for ReActor (URL or base64)"
    },
    "target_faces_index": {
        "type": [int, list],
        "required": False,
        "default": None,
        "description": "Index or list of indices of target faces to swap"
    }
} 