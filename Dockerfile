# ---------- Stage 1: Builder ----------
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu20.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python + pip (builder needs full dev stack)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-distutils \
        python3-pip \
        build-essential \
        git \
        curl \
        wget \
        unzip \
        cmake \
        ninja-build \
        ffmpeg \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip + install deps
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install --use-pep517 onnxsim \
 && pip install -r requirements.txt


# ---------- Stage 2: Runtime ----------
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu20.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install only what runtime needs (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-distutils \
        python3-pip \
        libglib2.0-0 \
        libgl1 \
        ffmpeg \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

EXPOSE 5000 6379

CMD ["python", "start_app.py"]
