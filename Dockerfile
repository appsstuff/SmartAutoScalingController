FROM python:3.10-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libopenblas-dev libomp-dev && \
    pip install --upgrade pip && \
    pip install intel-openmp && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.10-slim

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .
RUN mkdir -p /live_data
RUN chmod -R 777 /live_data

EXPOSE 8900
ENV TF_ENABLE_ONEDNN_OPTS=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONUNBUFFERED=1

# Start Prometheus metrics server + main loop
CMD ["python", "autoscaler_controller.py"]