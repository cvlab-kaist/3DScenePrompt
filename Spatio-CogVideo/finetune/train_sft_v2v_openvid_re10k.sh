# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=''

# Model Configuration
MODEL_ARGS=(
    --model_path "THUDM/CogVideoX-5b-I2V"  # ["THUDM/CogVideoX-5b-I2V", "THUDM/CogVideoX-2b-I2V"]
    --model_name "cogvideox-i2v"  # ["cogvideox-i2v", "cogvideox1.5-i2v"]
    --model_type "v2v_openvid"
    --training_type "v2v_sft"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "./output/"
    --report_to "wandb"
)

# Data Configuration
DATA_ARGS=(
    --data_root "{data_path}"
    --caption_column "caption.txt"
    --video_column "video.txt"
    --cond_video_column "cond_video.txt"
    # --image_column "images.txt"  # comment this line will use first frame of video as image conditioning
    --train_resolution "49x480x720"  # (frames x height x width), frames should be 8N+1
    --short_video_length 9  # number of frames to condition on
    --cache_config "openvid_re10k_34000_v2v"
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 40 # number of training epochs
    --seed 42 # random seed
    --batch_size 4
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"] # Only CogVideoX-2B supports fp16 training
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 16
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 500 # save checkpoint every x steps
    --checkpointing_limit 2 # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume_from_checkpoint ""  # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true  # ["true", "false"]
    --validation_dir "{validation_data_path}"
    --validation_steps 500  # should be multiple of checkpointing_steps
    --validation_prompts "prompt.txt"
    --validation_cond_videos "cond_video.txt"
    --gen_fps 16
)

# Combine all arguments and launch training
accelerate launch --config_file accelerate_config_4gpu.yaml --main_process_port 10543 train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"