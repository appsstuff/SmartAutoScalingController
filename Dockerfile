FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt  && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/pip
    
# RUN pip install --no-cache-dir -r requirements.txt && \
#     rm -rf /var/lib/apt/lists/* && \
#     rm -rf /root/.cache/pip


RUN mkdir -p /data/live_data
COPY . .

CMD ["python", "autoscaler_controller.py"]