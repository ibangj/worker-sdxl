'''
Contains the handler function that will be called by the serverless.
'''

import os
import base64
import concurrent.futures

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL
from diffusers.utils import load_image

from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

import runpod
from runpod.serverless.utils import rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from rp_schemas import INPUT_SCHEMA
# Import the extensions module
from extensions import init_extensions, process_with_controlnet, apply_reactor, download_image, EXTENSIONS_SCHEMA
from config import MODEL_PATHS

# Check if models exist, download if needed
missing_models = False
for model_path in MODEL_PATHS.values():
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        missing_models = True

if missing_models:
    print("Some models are missing, running download script...")
    import download_models
    download_models.main()

torch.cuda.empty_cache()

# Merge schemas
MERGED_SCHEMA = {**INPUT_SCHEMA, **EXTENSIONS_SCHEMA}

# ------------------------------- Model Handler ------------------------------ #


class ModelHandler:
    def __init__(self):
        self.base = None
        self.refiner = None
        self.load_models()
        # Initialize extensions
        init_extensions()

    def load_base(self):
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        base_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", vae=vae,
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False
        )
        base_pipe = base_pipe.to("cuda", silence_dtype_warnings=True)
        base_pipe.enable_xformers_memory_efficient_attention()
        self.base = base_pipe

    def load_refiner(self):
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", vae=vae,
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False
        )
        refiner_pipe = refiner_pipe.to("cuda", silence_dtype_warnings=True)
        refiner_pipe.enable_xformers_memory_efficient_attention()
        self.refiner = refiner_pipe

    def load_models(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            base_future = executor.submit(self.load_base)
            refiner_future = executor.submit(self.load_refiner)
            
            # Wait for both futures to complete
            concurrent.futures.wait([base_future, refiner_future])


MODELS = ModelHandler()

# ---------------------------------- Helper ---------------------------------- #


def _save_and_upload_images(images, job_id):
    os.makedirs(f"/{job_id}", exist_ok=True)
    image_urls = []
    for index, image in enumerate(images):
        image_path = os.path.join(f"/{job_id}", f"{index}.png")
        image.save(image_path)

        if os.environ.get('BUCKET_ENDPOINT_URL', False):
            image_url = rp_upload.upload_image(job_id, image_path)
            image_urls.append(image_url)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(
                    image_file.read()).decode("utf-8")
                image_urls.append(f"data:image/png;base64,{image_data}")

    rp_cleanup.clean([f"/{job_id}"])
    return image_urls


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]


@torch.inference_mode()
def generate_image(job):
    '''
    Generate an image from text using your Model
    '''
    job_input = job["input"]

    # Input validation
    validated_input = validate(job_input, MERGED_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    job_input = validated_input['validated_input']

    starting_image = job_input.get('image_url')
    
    # Extension flags
    use_controlnet = job_input.get('use_controlnet', False)
    use_reactor = job_input.get('use_reactor', False)

    if job_input['seed'] is None:
        job_input['seed'] = int.from_bytes(os.urandom(2), "big")

    generator = torch.Generator("cuda").manual_seed(job_input['seed'])

    MODELS.base.scheduler = make_scheduler(
        job_input['scheduler'], MODELS.base.scheduler.config)

    # Handle ControlNet if requested
    if use_controlnet and job_input.get('control_image'):
        control_images, message = process_with_controlnet(
            MODELS.base,
            prompt=job_input['prompt'],
            negative_prompt=job_input.get('negative_prompt'),
            control_image=job_input['control_image'],
            control_type=job_input.get('control_type', 'canny'),
            width=job_input['width'],
            height=job_input['height'],
            num_inference_steps=job_input['num_inference_steps'],
            guidance_scale=job_input['guidance_scale'],
            controlnet_conditioning_scale=job_input.get('controlnet_conditioning_scale', 0.5),
            seed=job_input['seed']
        )
        
        if message != "Success":
            return {"error": message}
            
        output = control_images
    
    elif starting_image:  # If image_url is provided, run only the refiner pipeline
        init_image = load_image(starting_image).convert("RGB")
        output = MODELS.refiner(
            prompt=job_input['prompt'],
            num_inference_steps=job_input['refiner_inference_steps'],
            strength=job_input['strength'],
            image=init_image,
            generator=generator
        ).images
    
    else:
        # Generate latent image using pipe
        image = MODELS.base(
            prompt=job_input['prompt'],
            negative_prompt=job_input['negative_prompt'],
            height=job_input['height'],
            width=job_input['width'],
            num_inference_steps=job_input['num_inference_steps'],
            guidance_scale=job_input['guidance_scale'],
            denoising_end=job_input['high_noise_frac'],
            output_type="latent",
            num_images_per_prompt=job_input['num_images'],
            generator=generator
        ).images

        try:
            output = MODELS.refiner(
                prompt=job_input['prompt'],
                num_inference_steps=job_input['refiner_inference_steps'],
                strength=job_input['strength'],
                image=image,
                num_images_per_prompt=job_input['num_images'],
                generator=generator
            ).images
        except RuntimeError as err:
            return {
                "error": f"RuntimeError: {err}, Stack Trace: {err.__traceback__}",
                "refresh_worker": True
            }

    # Apply ReActor face swap if requested
    if use_reactor and job_input.get('source_face_image'):
        try:
            # Download source face image
            source_face = download_image(job_input['source_face_image'])
            
            new_output = []
            for img in output:
                result_image, message = apply_reactor(
                    img, 
                    source_face, 
                    job_input.get('target_faces_index')
                )
                if message != "Success":
                    return {"error": message}
                new_output.append(result_image)
            
            output = new_output
        except Exception as e:
            return {"error": f"Error in ReActor: {str(e)}"}

    image_urls = _save_and_upload_images(output, job['id'])

    results = {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": job_input['seed']
    }

    if starting_image:
        results['refresh_worker'] = True

    return results


def check_and_download_model(local_path, url):
    """
    Check if a model exists locally and download it if not.
    This is a utility function that can be used for on-demand model downloading.
    """
    if os.path.exists(local_path):
        print(f"✅ Model already exists at {local_path}")
        return True
    
    print(f"🔍 Model not found locally, downloading from {url}...")
    try:
        import requests
        from tqdm import tqdm
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download with progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(local_path, 'wb') as file, tqdm(
            desc=local_path,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
                
        print(f"✅ Successfully downloaded to {local_path}")
        return True
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False


runpod.serverless.start({"handler": generate_image})
