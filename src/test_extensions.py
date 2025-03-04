import os
import torch
from diffusers import ControlNetModel
import onnxruntime

# Import shared configuration
from config import MODEL_PATHS, PROVIDERS

def test_controlnet_models():
    # Check if ControlNet models exist
    canny_path = MODEL_PATHS["controlnet_canny"]
    depth_path = MODEL_PATHS["controlnet_depth"]
    
    if not os.path.exists(canny_path):
        print("❌ ControlNet Canny model not found")
        return False
    
    if not os.path.exists(depth_path):
        print("❌ ControlNet Depth model not found")
        return False
    
    # Try loading models
    try:
        controlnet_canny = ControlNetModel.from_pretrained(canny_path, torch_dtype=torch.float16)
        print("✅ ControlNet Canny model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load ControlNet Canny: {e}")
        return False
        
    print("✅ All ControlNet tests passed")
    return True

def test_reactor():
    # Check if ReActor model exists
    face_swap_path = MODEL_PATHS["face_swap"]
    
    if not os.path.exists(face_swap_path):
        print("❌ Face swap model not found")
        return False
    
    # Try loading model
    try:
        face_swapper = onnxruntime.InferenceSession(
            face_swap_path, 
            providers=PROVIDERS
        )
        print("✅ Face swap model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load face swap model: {e}")
        return False
        
    print("✅ All ReActor tests passed")
    return True

if __name__ == "__main__":
    controlnet_success = test_controlnet_models()
    reactor_success = test_reactor()
    
    if controlnet_success and reactor_success:
        print("✅ All extension tests passed!")
        exit(0)
    else:
        print("❌ Some extension tests failed!")
        exit(1) 