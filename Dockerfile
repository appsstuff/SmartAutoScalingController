FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt  && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/pip

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for performance
ENV TF_ENABLE_ONEDNN_OPTS=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONUNBUFFERED=1

# Upgrade pip and install optimized TensorFlow
RUN pip install --upgrade pip && \
    pip install intel-extension-for-tensorflow
    
# RUN pip install --no-cache-dir -r requirements.txt && \
#     rm -rf /var/lib/apt/lists/* && \
#     rm -rf /root/.cache/pip


RUN mkdir -p /data/live_data
COPY . .

CMD ["python", "autoscaler_controller.py"]