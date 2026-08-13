FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake g++ gcc git git-lfs wget curl unzip \
    ffmpeg \
    libeigen3-dev \
    libboost-all-dev \
    gosu \
    xvfb \
    libegl1-mesa-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"
RUN conda init bash

# Create grail conda environment with Python 3.10
RUN conda create -y -n grail python=3.10

# Install PyTorch (CUDA 12.8 for RTX 5090)
RUN conda run -n grail pip install \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Headless rendering environment
ENV PYOPENGL_PLATFORM=egl
ENV EGL_PLATFORM=surfaceless

# Activate grail env by default in bash
RUN echo "conda activate grail" >> /root/.bashrc

# Embed entrypoint
COPY tools/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

WORKDIR /workspace/grail

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["bash"]
