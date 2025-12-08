#!/usr/bin/env bash
unset PYTHONPATH
export PYTHONNOUSERSITE=1   # ~/.local 무시


python cli_demo_from_folder.py \
    --validation_prompts_dir "../dataset/captions.txt" \
    --validation_videos_dir "../dataset/cond_video.txt" \
    --model_path THUDM/CogVideoX-5b-V2V-mine \
    --output_path ../dataset/ours \
    --num_frames 49 \
    