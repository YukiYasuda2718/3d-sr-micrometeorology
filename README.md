# 3d-sr-micrometeorology <!-- omit in toc -->

[![license](https://img.shields.io/badge/license-CC%20BY--NC--SA-informational)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)  [![reference](https://img.shields.io/badge/reference-arXiv-important)](https://arxiv.org/abs/2303.16684) [![pytorch](https://img.shields.io/badge/PyTorch-1.11.0-informational)](https://pytorch.org/)

This repository contains the source code used in *Super-Resolution of Three-Dimensional Temperature and Velocity for Building-Resolving Urban Micrometeorology Using Physics-Guided Convolutional Neural Networks with Image Inpainting Techniques* ([arXiv](https://arxiv.org/abs/2303.16684)).

- [Setup](#setup)
  - [Docker Containers](#docker-containers)
  - [Singularity Containers](#singularity-containers)
- [Code used in experiments](#code-used-in-experiments)

## Setup

- The Singularity containers were used for experiments.
- The Docker containers provide the same environments as in the Singularity containers.

### Docker Containers

1. Install [Docker](https://docs.docker.com/get-started/).
1. Build docker containers: `$ docker compose build`
1. Start docker containers: `$ docker compose up -d`

### Singularity Containers

1. Install [Singularity](https://docs.sylabs.io/guides/3.0/user-guide/quick_start.html).
1. Build Singularity containers:
    - `$ singularity build -f datascience.sif ./singularity/datascience.sif.def`
    - `$ singularity build -f pytorch.sif ./singularity/pytorch.def`
2. Start singularity containers:
    - The following command is an exmple for local.

```sh
export PORT=8888 # your own port
singularity exec --nv --env PYTHONPATH="$(pwd)/pytorch" \
  ./pytorch.sif jupyter lab --no-browser --ip=0.0.0.0 --allow-root --LabApp.token='' --port=$PORT
```

## Code used in experiments

- [Data generation for deep learning](./datascience/script/make_dl_data_using_outside_lr_builds.py)
  - This code was run using the `datascience` container on [the Earth Simulator](https://www.jamstec.go.jp/es/en/).
- [CNN training](./pytorch/script/train_model.sh)
  - The root directory and config paths are need to be specified.
  - This shell script runs [python script](./pytorch/script/train_model.py).
    - This python script was run using the `pytorch` container on [the Earth Simulator](https://www.jamstec.go.jp/es/en/).
- [CNN evaluation](./pytorch/notebook)
  - The CNNs were evaluated using notebooks in [the above dir](./pytorch/notebook).
  - These notebooks wre run using the `pytorch` container on a local environment.

