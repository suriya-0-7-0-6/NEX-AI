# Stage 1 - Base image
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/root/.local/bin:$PATH" \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    unzip \
    ffmpeg \
    cmake \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Stage 2 - Builder
FROM base AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --user --no-cache-dir --use-pep517 onnxsim
RUN pip install --user -r requirements.txt 

# Stage 3 - Runtime
FROM base AS runtime
WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY . .

EXPOSE 5000 6379

CMD ["python", "start_app.py"]
