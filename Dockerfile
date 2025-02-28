# Base image
FROM runpod/base:0.4.2-cuda11.8.0

ENV HF_HUB_ENABLE_HF_TRANSFER=0

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Cache Models
COPY builder/cache_models.py /cache_models.py
RUN python3.11 /cache_models.py && \
    rm /cache_models.py

# Add src files (Worker Template)
ADD src .

# Install ControlNet and ReActor dependencies
RUN pip install diffusers==0.21.4 transformers==4.31.0 accelerate==0.21.0
RUN pip install opencv-python-headless insightface onnxruntime

# Download ControlNet models
RUN mkdir -p /controlnet
RUN wget https://huggingface.co/lllyasviel/sd-controlnet-canny/resolve/main/model.safetensors -O /controlnet/control_canny.safetensors
RUN wget https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/model.safetensors -O /controlnet/control_depth.safetensors
# Add other ControlNet models as needed

# Download ReActor face model
RUN mkdir -p /reactor
RUN wget https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx -O /reactor/inswapper_128.onnx

CMD python3.11 -u /rp_handler.py
