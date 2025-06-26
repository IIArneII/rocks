# ---------- base image --------------------------------------------------------
ARG CUDA_VERSION=11.3.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu20.04

# ---------- system packages ---------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential \
        libglib2.0-0 libsm6 libxrender-dev libxext6 \
        && rm -rf /var/lib/apt/lists/*

# ---------- micromamba --------------------------------------------------------
ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV PATH=/opt/conda/bin:$PATH
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# ---------- create non-root user ---------------------------------------------
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && usermod -aG sudo ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USERNAME}
WORKDIR /home/${USERNAME}

# ---------- conda env ---------------------------------------------------------
RUN micromamba create -y -n detectron2 \
        python=3.7 \
&& micromamba clean -y --all
SHELL ["micromamba", "run", "-n", "detectron2", "/bin/bash", "-c"]

# ---------- python deps -------------------------------------------------------
RUN pip install --no-cache-dir torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install --no-cache-dir tensorboard cmake onnx jupyterlab
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

# ---------- setup workspace ---------------------------------------------------
WORKDIR /workspace

# ---------- ports & cmd -------------------------------------------------------
EXPOSE 8889
CMD ["micromamba", "run", "-n", "detectron2", "jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]