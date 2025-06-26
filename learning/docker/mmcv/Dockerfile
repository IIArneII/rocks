# ---------- base image --------------------------------------------------------
ARG CUDA_VERSION=12.4.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

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
RUN micromamba create -y -n sam \
        python=3.11 \
        pytorch=2.2.2 torchvision torchaudio \
        pytorch-cuda=${CUDA_VERSION%%.*}.1 \
        -c defaults -c conda-forge -c pytorch -c nvidia \
&& micromamba clean -y --all
SHELL ["micromamba", "run", "-n", "sam", "/bin/bash", "-c"]

# ---------- python deps -------------------------------------------------------
RUN pip install --no-cache-dir \
        git+https://github.com/facebookresearch/segment-anything.git \
        git+https://github.com/huggingface/peft.git \
        bitsandbytes==0.43.1 \
        accelerate>=0.27 \
        wandb jupyterlab tqdm tensorboard

# ---------- copy workspace (optional) ----------------------------------------
COPY --chown=${USERNAME}:${USERNAME} ./src /workspace
WORKDIR /workspace

# ---------- ports & cmd -------------------------------------------------------
EXPOSE 8888
CMD ["micromamba", "run", "-n", "sam", "jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]