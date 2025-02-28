INPUT_SCHEMA = {
    "prompt": {
        "type": str,
        "required": True,
        "default": None,
        "description": "The prompt to generate images from"
    },
    "negative_prompt": {
        "type": str,
        "required": False,
        "default": "",
        "description": "The negative prompt to use for generation"
    },
    "width": {
        "type": int,
        "required": False,
        "default": 1024,
        "description": "Width of the generated image"
    },
    "height": {
        "type": int,
        "required": False,
        "default": 1024,
        "description": "Height of the generated image"
    },
    "num_inference_steps": {
        "type": int,
        "required": False,
        "default": 30,
        "description": "Number of denoising steps"
    },
    "guidance_scale": {
        "type": float,
        "required": False,
        "default": 7.5,
        "description": "Scale for classifier-free guidance"
    },
    "seed": {
        "type": int,
        "required": False,
        "default": None,
        "description": "Random seed for generation"
    },
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