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

# Create directories for models
RUN mkdir -p /controlnet /reactor

# Add src files (Worker Template)
ADD src .

# Add startup script to download models at runtime
COPY src/config.py /config.py
COPY src/download_models.py /download_models.py
COPY src/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Use the entrypoint script
ENTRYPOINT ["/entrypoint.sh"]
