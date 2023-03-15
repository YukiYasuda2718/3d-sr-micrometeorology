FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

ENV HOME=/workspace \
    LD_LIBRARY_PATH=/usr/local/lib/ \
    PIP_NO_CACHE_DIR="off"

ARG DEBIAN_FRONTEND="noninteractive"

EXPOSE 6006 8888 5000

WORKDIR ${HOME}

RUN useradd -M vscode && \
    chmod -R 777 ${HOME}

# Update GPG key
# https://github.com/NVIDIA/nvidia-docker/issues/1631
# https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
# https://www.nemotos.net/?p=5178
RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# libgl1-mesa-dev and libopencv-dev are necessary for opencv2
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    less \
    git \
    curl \
    zsh \
    libgl1-mesa-dev \
    libopencv-dev \
    ttf-mscorefonts-installer && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN chsh -s /usr/bin/zsh

COPY Pipfile Pipfile.lock ${HOME}/

# Python libraries such as black are installed in ${HOME}/.local/bin 
RUN cd ${HOME} && \
    pip install -U pip==23.0.1 pipenv==2023.2.18 && \
    export PATH="$PATH:${HOME}/.local/bin" && \
    pipenv install --system --dev --ignore-pipfile && \
    rm -rf ${HOME}/.cache/pip ${HOME}/.cache/pipenv

# Copy JupyterLab formatter configs
COPY .jupyter/ ${HOME}/.jupyter/
