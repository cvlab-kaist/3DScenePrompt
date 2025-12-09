#!/usr/bin/env bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1   # ~/.local 무시

base_path=/mnt/SpatialVID-HQ/validation
video_base_path=${base_path}/video
output_base_path=${base_path}/captions
GPUS=0



CUDA_VISIBLE_DEVICES=${GPUS} python ./vid2caption.py --batch ${GPUS} \
    --video_base_path ${video_base_path} \
    --output_base_path ${output_base_path}

