# ---------- Stage 1: Builder ----------
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu20.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore


WORKDIR /app

# Install Python 3.8 + pip + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-distutils \
        python3-pip \
        build-essential \
        git \
        curl \
        wget \
        unzip \
        cmake \
        ninja-build \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Ensure "python" command points to python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Upgrade pip, setuptools, wheel (industry standard best practice)
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements early (better caching)
COPY requirements.txt .

# Install build-time deps + project requirements
RUN python -m pip install --use-pep517 onnxsim \
    && python -m pip install -r requirements.txt


# ---------- Stage 2: Runtime ----------
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu20.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install minimal runtime dependencies (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-distutils \
        python3-pip \
        libglib2.0-0 \
        libgl1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Ensure "python" command points to python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Upgrade pip (so runtime matches builder)
RUN python -m pip install --upgrade pip setuptools wheel

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.8 /usr/local/lib/python3.8
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

EXPOSE 5000

CMD ["python", "start_app.py"]
