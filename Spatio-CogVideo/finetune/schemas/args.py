import argparse
import datetime
import logging
from pathlib import Path
from typing import Any, List, Literal, Tuple

from pydantic import BaseModel, ValidationInfo, field_validator


class Args(BaseModel):
    ########## Model ##########
    model_path: Path
    model_name: str
    model_type: Literal["i2v", "t2v", "i2v_short", 'i2v_short_dl3dv', 'i2v_short_dl3dv_latent_warping', 'v2v', 'v2v_dl3dv', 
                        'v2v_openvid', 'v2v_controlnet_openvid', 't2v_openvid', 'i2v_openvid', 'i2v_openvid_uni3c', 'v2v_openvid_uni3c']
    
    training_type: Literal["lora", "sft", "short_lora", "concat_lora", 'v2v_lora', 'v2v_cond_lora', 'v2v_sft', 'v2v_lora2', 
                           'v2v_controlnet_sft', 'ip_adapter', 'das', 'das_cut3r','das_uni3c', 'das_uni3c_cut3r', 'v2v_das_uni3c_lora'] = "lora"
    controlnet_transformer_num_attn_heads: int = 4
    controlnet_transformer_attention_head_dim: int = 64
    controlnet_transformer_out_proj_dim_factor: int = 64
    controlnet_transformer_out_proj_dim_zero_init: bool = True
    controlnet_model_path: Path | None = None
    

    ########## Output ##########
    output_dir: Path = Path("train_results/{:%Y-%m-%d-%H-%M-%S}".format(datetime.datetime.now()))
    report_to: Literal["tensorboard", "wandb", "all"] | None = None
    tracker_name: str = "finetrainer-cogvideo"

    ########## Data ###########
    data_root: Path
    caption_column: Path
    image_column: Path | None = None
    video_column: Path
    short_video_column: Path | None = None # for v2v model
    cond_video_column: Path | None = None # for v2v model
    camera_column: Path | None = None # for v2v model
    cut3r_column: Path | None = None # for das_cut3r model
    data_preprocess_batch: int = -1  # -1 means no limit
    total_gpus: int = 5 # data preprocess total gpus
    cache_config: str = 'openvid_12000_i2v' # openvid_12000_i2v, openvid_12000_v2v, openvid_12000_v2v_cond
    short_video_length: int = 5
    short_loader: bool = False
    inverse: bool = False

    ########## Training #########
    resume_from_checkpoint: Path | None = None

    seed: int | None = None
    train_epochs: int
    train_steps: int | None = None
    checkpointing_steps: int = 200
    checkpointing_limit: int = 10

    batch_size: int
    gradient_accumulation_steps: int = 1

    train_resolution: Tuple[int, int, int]  # shape: (frames, height, width)

    #### deprecated args: video_resolution_buckets
    # if use bucket for training, should not be None
    # Note1: At least one frame rate in the bucket must be less than or equal to the frame rate of any video in the dataset
    # Note2:  For cogvideox, cogvideox1.5
    #   The frame rate set in the bucket must be an integer multiple of 8 (spatial_compression_rate[4] * path_t[2] = 8)
    #   The height and width set in the bucket must be an integer multiple of 8 (temporal_compression_rate[8])
    # video_resolution_buckets: List[Tuple[int, int, int]] | None = None

    mixed_precision: Literal["no", "fp16", "bf16"]

    learning_rate: float = 2e-5
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    beta3: float = 0.98
    epsilon: float = 1e-8
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 100
    lr_num_cycles: int = 1
    lr_power: float = 1.0

    num_workers: int = 8
    pin_memory: bool = True

    gradient_checkpointing: bool = True
    enable_slicing: bool = True
    enable_tiling: bool = True
    nccl_timeout: int = 1800
    zeropad: bool = False

    ########## Lora ##########
    rank: int = 128
    lora_alpha: int = 64
    init_lora_weights: str = "gaussian"
    # target_modules: List[str] = ["to_q"]
    # target_modules: List[str] = ["to_q", "to_k", "to_v", "to_out.0", "ff.net.2", "ff.net.0.proj", "norm1.linear", "norm2.linear"]
    target_modules: List[str] = ["to_q", "to_k", "to_v", "to_out.0"]
    # target_modules = "(.*x_embedder|.*(?<!single_)transformer_blocks\\.[0-9]+\\.norm1\\.linear|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_k|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_q|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_v|.*(?<!single_)transformer_blocks\\.[0-9]+\\.attn\\.to_out\\.0|.*(?<!single_)transformer_blocks\\.[0-9]+\\.ff\\.net\\.2|.*single_transformer_blocks\\.[0-9]+\\.norm\\.linear|.*single_transformer_blocks\\.[0-9]+\\.proj_mlp|.*single_transformer_blocks\\.[0-9]+\\.proj_out|.*single_transformer_blocks\\.[0-9]+\\.attn.to_k|.*single_transformer_blocks\\.[0-9]+\\.attn.to_q|.*single_transformer_blocks\\.[0-9]+\\.attn.to_v|.*single_transformer_blocks\\.[0-9]+\\.attn.to_out)"

    ########## Validation ##########
    do_validation: bool = False
    validation_steps: int | None  # if set, should be a multiple of checkpointing_steps
    validation_dir: Path | None  # if set do_validation, should not be None
    validation_prompts: str | None  # if set do_validation, should not be None
    validation_images: str | None  # if set do_validation and model_type == i2v, should not be None
    validation_videos: str | None  # if set do_validation and model_type == v2v, should not be None
    validation_cond_videos: str | None  # if set do_validation and model_type == v2v, should not be None
    validation_camera:str | None = None  # if set do_validation and model_type == v2v, should not be None
    validation_cut3r_state:str | None = None  # if set do_validation and model_type == v2v, should not be None
    gen_fps: int = 15
    num_tracking_blocks:int = 18

    #### deprecated args: gen_video_resolution
    # 1. If set do_validation, should not be None
    # 2. Suggest selecting the bucket from `video_resolution_buckets` that is closest to the resolution you have chosen for fine-tuning
    #        or the resolution recommended by the model
    # 3. Note:  For cogvideox, cogvideox1.5
    #        The frame rate set in the bucket must be an integer multiple of 8 (spatial_compression_rate[4] * path_t[2] = 8)
    #        The height and width set in the bucket must be an integer multiple of 8 (temporal_compression_rate[8])
    # gen_video_resolution: Tuple[int, int, int] | None  # shape: (frames, height, width)

    @field_validator("image_column")
    def validate_image_column(cls, v: str | None, info: ValidationInfo) -> str | None:
        values = info.data
        if values.get("model_type") == "i2v" and not v:
            logging.warning(
                "No `image_column` specified for i2v model. Will automatically extract first frames from videos as shortitioning images."
            )
        return v
    
    @field_validator("short_video_column")
    def validate_image_column(cls, v: str | None, info: ValidationInfo) -> str | None:
        values = info.data
        if values.get("model_type") == "v2v" and not v:
            logging.warning(
                "No `short_video_column` specified for v2v model. Will automatically extract serveral frames from videos as shortitioning images."
            )
        return v

    @field_validator("validation_dir", "validation_prompts")
    def validate_validation_required_fields(cls, v: Any, info: ValidationInfo) -> Any:
        values = info.data
        if values.get("do_validation") and not v:
            field_name = info.field_name
            raise ValueError(f"{field_name} must be specified when do_validation is True")
        return v

    @field_validator("validation_images")
    def validate_validation_images(cls, v: str | None, info: ValidationInfo) -> str | None:
        values = info.data
        if values.get("do_validation") and values.get("model_type") == "i2v" and not v:
            raise ValueError("validation_images must be specified when do_validation is True and model_type is i2v")
        return v

    @field_validator("validation_videos")
    def validate_validation_videos(cls, v: str | None, info: ValidationInfo) -> str | None:
        values = info.data
        if values.get("do_validation") and values.get("model_type") == "v2v" and not v:
            raise ValueError("validation_videos must be specified when do_validation is True and model_type is v2v")
        return v
    
    @field_validator("validation_cut3r_state")
    def validate_validation_videos(cls, v: str | None, info: ValidationInfo) -> str | None:
        values = info.data
        if values.get("do_validation") and values.get("model_type") == "t2v_openvid" and not v:
            raise ValueError("validation_videos must be specified when do_validation is True and model_type is v2v")
        return v

    @field_validator("validation_steps")
    def validate_validation_steps(cls, v: int | None, info: ValidationInfo) -> int | None:
        values = info.data
        if values.get("do_validation"):
            if v is None:
                raise ValueError("validation_steps must be specified when do_validation is True")
            if values.get("checkpointing_steps") and v % values["checkpointing_steps"] != 0:
                raise ValueError("validation_steps must be a multiple of checkpointing_steps")
        return v

    @field_validator("train_resolution")
    def validate_train_resolution(cls, v: Tuple[int, int, int], info: ValidationInfo) -> str:
        try:
            frames, height, width = v

            # Check if (frames - 1) is multiple of 8
            if (frames - 1) % 8 != 0:
                raise ValueError("Number of frames - 1 must be a multiple of 8")

            # Check resolution for cogvideox-5b models
            model_name = info.data.get("model_name", "")
            if model_name in ["cogvideox-5b-i2v", "cogvideox-5b-t2v"]:
                if (height, width) != (480, 720):
                    raise ValueError("For cogvideox-5b models, height must be 480 and width must be 720")

            return v

        except ValueError as e:
            if (
                str(e) == "not enough values to unpack (expected 3, got 0)"
                or str(e) == "invalid literal for int() with base 10"
            ):
                raise ValueError("train_resolution must be in format 'frames x height x width'")
            raise e

    @field_validator("mixed_precision")
    def validate_mixed_precision(cls, v: str, info: ValidationInfo) -> str:
        if v == "fp16" and "cogvideox-2b" not in str(info.data.get("model_path", "")).lower():
            logging.warning(
                "All CogVideoX models except cogvideox-2b were trained with bfloat16. "
                "Using fp16 precision may lead to training instability."
            )
        return v

    @classmethod
    def parse_args(cls):
        """Parse command line arguments and return Args instance"""
        parser = argparse.ArgumentParser()
        # Required arguments
        parser.add_argument("--model_path", type=str, required=True)
        parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--model_type", type=str, required=True)
        parser.add_argument("--training_type", type=str, required=True)
        parser.add_argument("--output_dir", type=str, required=True)
        parser.add_argument("--data_root", type=str, required=True)
        parser.add_argument("--caption_column", type=str, required=True)
        parser.add_argument("--video_column", type=str, required=True)
        parser.add_argument("--camera_column", type=str, default=None)
        parser.add_argument("--cut3r_column", type=str, default=None)
        parser.add_argument("--train_resolution", type=str, required=True)
        parser.add_argument("--report_to", type=str, required=True)
        parser.add_argument("--data_preprocess_batch", type=int, default=-1)
        parser.add_argument("--total_gpus", type=int, default=5)
        parser.add_argument("--cache_config", type=str, default='openvid_12000_i2v')
        parser.add_argument("--short_video_length", type=int, default=5)
        parser.add_argument("--controlnet_transformer_num_attn_heads", type=int, default=4)
        parser.add_argument("--controlnet_transformer_attention_head_dim", type=int, default=64)
        parser.add_argument("--controlnet_transformer_out_proj_dim_factor", type=int, default=64)
        parser.add_argument("--controlnet_transformer_out_proj_dim_zero_init", type=bool, default=False)
        parser.add_argument("--num_tracking_blocks", type=int, default=18)
        

        # Training hyperparameters
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--train_epochs", type=int, default=10)
        parser.add_argument("--train_steps", type=int, default=None)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--optimizer", type=str, default="adamw")
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.95)
        parser.add_argument("--beta3", type=float, default=0.98)
        parser.add_argument("--epsilon", type=float, default=1e-8)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--max_grad_norm", type=float, default=1.0)
        parser.add_argument("--controlnet_model_path", type=str, default=None)

        # Learning rate scheduler
        parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
        parser.add_argument("--lr_warmup_steps", type=int, default=100)
        parser.add_argument("--lr_num_cycles", type=int, default=1)
        parser.add_argument("--lr_power", type=float, default=1.0)

        # Data loading
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--pin_memory", type=bool, default=True)
        parser.add_argument("--image_column", type=str, default=None)
        parser.add_argument("--short_video_column", type=str, default=None)
        parser.add_argument("--cond_video_column", type=str, default=None)
        parser.add_argument("--short_loader", action="store_true", help="Enable short_loader mode")
        parser.add_argument("--inverse", action="store_true", help="Enable inverse mode")

        # Model configuration
        parser.add_argument("--mixed_precision", type=str, default="no")
        parser.add_argument("--gradient_checkpointing", type=bool, default=True)
        parser.add_argument("--enable_slicing", type=bool, default=True)
        parser.add_argument("--enable_tiling", type=bool, default=True)
        parser.add_argument("--nccl_timeout", type=int, default=1800)
        parser.add_argument("--zeropad", action="store_true", help="Enable zero padding for input videos")

        # LoRA parameters
        parser.add_argument("--rank", type=int, default=128)
        parser.add_argument("--lora_alpha", type=int, default=64)
        parser.add_argument("--target_modules", type=str, nargs="+", default=["to_q", "to_k", "to_v", "to_out.0"])

        # Checkpointing
        parser.add_argument("--checkpointing_steps", type=int, default=200)
        parser.add_argument("--checkpointing_limit", type=int, default=10)
        parser.add_argument("--resume_from_checkpoint", type=str, default=None)

        # Validation
        parser.add_argument("--do_validation", type=lambda x: x.lower() == 'true', default=False)
        parser.add_argument("--validation_steps", type=int, default=None)
        parser.add_argument("--validation_dir", type=str, default=None)
        parser.add_argument("--validation_prompts", type=str, default=None)
        parser.add_argument("--validation_images", type=str, default=None)
        parser.add_argument("--validation_videos", type=str, default=None)
        parser.add_argument("--validation_cond_videos", type=str, default=None)
        parser.add_argument("--validation_camera", type=str, default=None)
        parser.add_argument("--validation_cut3r_state", type=str, default=None)
        parser.add_argument("--gen_fps", type=int, default=15)

        args = parser.parse_args()

        # Convert video_resolution_buckets string to list of tuples
        frames, height, width = args.train_resolution.split("x")
        args.train_resolution = (int(frames), int(height), int(width))

        return cls(**vars(args))
