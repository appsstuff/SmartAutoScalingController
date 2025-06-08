FROM python:3.11-slim-bullseye

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/pip
RUN mkdir -p /data/live_data
COPY . .

CMD ["python", "autoscaler_controller.py"]