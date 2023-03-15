# 3d-sr-micrometeorology <!-- omit in toc -->

[![license](https://img.shields.io/badge/license-CC%20BY--NC--SA-informational)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) [![pytorch](https://img.shields.io/badge/PyTorch-1.11.0-informational)](https://pytorch.org/)

This repository contains the source code used in *Super-Resolution of Three-Dimensional Temperature and Velocity for Building-Resolving Urban Micrometeorology Using Physics-Guided Convolutional Neural Networks with Image Inpainting Techniques.*

- [Setup](#setup)
  - [Docker Containers](#docker-containers)
  - [Singularity Containers](#singularity-containers)

## Setup

- The Singularity containers were used for experiments.
  - At least, 1 GPU board of NVIDIA A100 40GB is required.
  - The same singularity containers were used on local and [Earth Simulator (ES)](https://www.jamstec.go.jp/es/en/)
- The Docker containers have the same environments as in the Singularity containers.
  - These Docker containers were used to test if the Singularity containers work.

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
    - The following command is an exmple for local

```sh
$ export PORT=8888 # your own port
$ singularity exec --nv --env PYTHONPATH="$(pwd)/pytorch" \
    ./pytorch.sif jupyter lab --no-browser --ip=0.0.0.0 --allow-root --LabApp.token='' --port=$PORT
```
