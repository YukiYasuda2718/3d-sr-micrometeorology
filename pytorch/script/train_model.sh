#!/bin/bash

ROOT_DIR="/home/yuki_yasuda/workspace_lab/3d-scalar-sr" ;;

IMAGE_PATH="${ROOT_DIR}/pytorch.sif"
SCRIPT_PATH="${ROOT_DIR}/pytorch/script/train_model.py"
CONFIG_PATH="${ROOT_DIR}/pytorch/config/default.yml"

echo "image path = ${IMAGE_PATH}"
echo "script path = ${SCRIPT_PATH}"
echo "config path = ${CONFIG_PATH}"

singularity exec \
  --nv \
  --env PYTHONPATH=$ROOT_DIR/pytorch \
  ${IMAGE_PATH} python3 ${SCRIPT_PATH} --config_path ${CONFIG_PATH} --world_size 1
