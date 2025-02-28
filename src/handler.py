from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis
import onnxruntime

# Initialize global variables for models
pipe = None
controlnet_models = {}
face_analyzer = None
face_swapper = None

def init():
    global pipe, controlnet_models, face_analyzer, face_swapper
    
    # Load standard SDXL model
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"  # Standard SDXL model
    
    # Load ControlNet models
    controlnet_canny = ControlNetModel.from_pretrained(
        "/controlnet/control_canny.safetensors", torch_dtype=torch.float16
    )
    controlnet_depth = ControlNetModel.from_pretrained(
        "/controlnet/control_depth.safetensors", torch_dtype=torch.float16
    )
    
    controlnet_models = {
        "canny": controlnet_canny,
        "depth": controlnet_depth,
        # Add other models as needed
    }
    
    # Initialize face analyzer and swapper for ReActor
    face_analyzer = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    face_swapper = onnxruntime.InferenceSession("/reactor/inswapper_128.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    # Initialize the pipeline with your model
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        model_id,
        controlnet=[controlnet_canny, controlnet_depth],
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    pipe.to("cuda")
    
    # ... existing code ...

def preprocess_image(image_path, control_type):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if control_type == "canny":
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
    elif control_type == "depth":
        # Use a depth estimator here or preprocess accordingly
        pass
    
    return image

def apply_reactor(image, source_image_path, target_faces_index=None):
    source_img = cv2.imread(source_image_path)
    source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
    
    source_faces = face_analyzer.get(source_img)
    if len(source_faces) == 0:
        return image, "No source face found"
    
    source_face = source_faces[0]
    
    target_img = np.array(image)
    target_faces = face_analyzer.get(target_img)
    
    if len(target_faces) == 0:
        return image, "No target face found"
    
    if target_faces_index is not None:
        if isinstance(target_faces_index, int) and 0 <= target_faces_index < len(target_faces):
            target_faces = [target_faces[target_faces_index]]
        elif isinstance(target_faces_index, list):
            target_faces = [face for i, face in enumerate(target_faces) if i in target_faces_index]
    
    result_image = target_img.copy()
    for target_face in target_faces:
        result_image = face_swapper.run(None, {
            'source': source_face,
            'target': target_face,
            'image': result_image
        })[0]
    
    return Image.fromarray(result_image), "Success"

def inference(event):
    global pipe, controlnet_models
    
    # ... existing code ...
    
    # Extract parameters
    prompt = event.get("prompt", "")
    negative_prompt = event.get("negative_prompt", "")
    width = event.get("width", 1024)
    height = event.get("height", 1024)
    num_inference_steps = event.get("num_inference_steps", 30)
    guidance_scale = event.get("guidance_scale", 7.5)
    
    # ControlNet parameters
    use_controlnet = event.get("use_controlnet", False)
    control_type = event.get("control_type", "canny")
    control_image = event.get("control_image", None)
    controlnet_conditioning_scale = event.get("controlnet_conditioning_scale", 0.5)
    
    # ReActor parameters
    use_reactor = event.get("use_reactor", False)
    source_face_image = event.get("source_face_image", None)
    target_faces_index = event.get("target_faces_index", None)
    
    # Set up pipeline parameters
    generator = torch.Generator(device="cuda").manual_seed(event.get("seed", 42))
    
    # Regular SDXL generation without ControlNet
    if not use_controlnet:
        images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images
    else:
        # Process the control image
        if control_image:
            # Assuming control_image is a URL or base64 string
            # Download or decode the image and save it to a temp file
            temp_control_path = "/tmp/control_image.jpg"
            # ... code to save the control image to temp_control_path ...
            
            control_image = preprocess_image(temp_control_path, control_type)
            control_image = Image.fromarray(control_image)
            
            # Use ControlNet pipeline
            selected_controlnet = controlnet_models.get(control_type)
            if selected_controlnet:
                controlnet_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                    pipe.config.name_or_path,
                    controlnet=selected_controlnet,
                    torch_dtype=torch.float16
                )
                controlnet_pipe.to("cuda")
                
                images = controlnet_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=control_image,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    controlnet_conditioning_scale=controlnet_conditioning_scale,
                    generator=generator
                ).images
            else:
                return {"error": f"ControlNet type '{control_type}' not found"}
        else:
            return {"error": "Control image is required for ControlNet"}
    
    # Apply ReActor face swap if requested
    if use_reactor and source_face_image:
        # Assuming source_face_image is a URL or base64 string
        # Download or decode the image and save it to a temp file
        temp_source_path = "/tmp/source_face.jpg"
        # ... code to save the source face image to temp_source_path ...
        
        result_image, message = apply_reactor(images[0], temp_source_path, target_faces_index)
        if message != "Success":
            return {"error": message}
        images = [result_image]
    
    # ... existing code to return the images ...

# ... existing code ... 