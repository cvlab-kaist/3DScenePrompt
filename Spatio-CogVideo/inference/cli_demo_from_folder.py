"""
This script demonstrates how to generate a video using the CogVideoX model with the Hugging Face `diffusers` pipeline.
The script supports different types of video generation, including text-to-video (t2v), image-to-video (i2v),
and video-to-video (v2v), depending on the input data and different weight.

- text-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- video-to-video: THUDM/CogVideoX-5b, THUDM/CogVideoX-2b or THUDM/CogVideoX1.5-5b
- image-to-video: THUDM/CogVideoX-5b-I2V or THUDM/CogVideoX1.5-5b-I2V

Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
$ python cli_demo.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX1.5-5b --generate_type "t2v"
```

You can change `pipe.enable_sequential_cpu_offload()` to `pipe.enable_model_cpu_offload()` to speed up inference, but this will use more GPU memory

Additional options are available to specify the model path, guidance scale, number of inference steps, video generation type, and output paths.

"""

import os
PATH = '/mnt/cache'
os.environ['TRANSFORMERS_CACHE'] = PATH
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH

import argparse
import logging
from typing import Literal, Optional

import torch

from diffusers import (
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXPipeline,
    # CogVideoXVideoToVideoPipeline,
)
import sys
sys.path.append("/root/JB/project/Spatio-CogVideo")
import cv2

from finetune.models.pipelines.pipeline import CogVideoXVideoToVideoPipeline
from finetune.my_datasets.utils import (
    load_images,
    load_prompts,
    load_videos,
    preprocess_image_with_resize,
    preprocess_video_with_resize,
)
from diffusers.utils import export_to_video, load_image, load_video

import shutil
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO)

# Recommended resolution for each model (width, height)
RESOLUTION_MAP = {
    # cogvideox1.5-*
    "cogvideox1.5-5b-i2v": (768, 1360),
    "cogvideox1.5-5b": (768, 1360),
    # cogvideox-*
    "cogvideox-5b-i2v": (480, 720),
    "cogvideox-5b": (480, 720),
    "cogvideox-2b": (480, 720),
    "cogvideox-5b-i2v-mine": (480, 720),
    "cogvideox-5b-v2v-mine": (480, 720),
}

def video2image(video_path, output_base_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_vid_paht = os.path.join(output_base_path, f"{video_name}", 'color')
    output_path = os.path.join(output_base_path, f"{video_name}")
    if os.path.exists(output_path):
        print(f"video exsist :{output_path}")
        if os.path.exists(os.path.join(output_path, 'cache_flow')):shutil.rmtree(os.path.join(output_path, 'cache_flow'))
        if os.path.exists(os.path.join(output_path, 'color')):shutil.rmtree(os.path.join(output_path, 'color'))
        if os.path.exists(os.path.join(output_path, 'depth_warped')):shutil.rmtree(os.path.join(output_path, 'depth_warped'))
        if os.path.exists(os.path.join(output_path, 'megasam')):shutil.rmtree(os.path.join(output_path, 'megasam'))
        if os.path.exists(os.path.join(output_path, 'megasam_refine')):shutil.rmtree(os.path.join(output_path, 'megasam_refine'))
        if os.path.exists(os.path.join(output_path, 'depth_warped_mask')):shutil.rmtree(os.path.join(output_path, 'depth_warped_mask'))
        if os.path.exists(os.path.join(output_path, 'dynamic_mask')):shutil.rmtree(os.path.join(output_path, 'dynamic_mask'))
        if os.path.exists(os.path.join(output_path, 'Depth-Anything')):shutil.rmtree(os.path.join(output_path, 'Depth-Anything'))
        if os.path.exists(os.path.join(output_path, 'UniDepth')):shutil.rmtree(os.path.join(output_path, 'UniDepth'))
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    frame_idx = 0
    os.makedirs(output_vid_paht, exist_ok=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        image_name = f"{frame_idx:06d}.png"
        image_path = os.path.join(output_vid_paht, image_name)
        cv2.imwrite(image_path, frame)
        frame_idx += 1
    cap.release()

def should_skip_video(save_path, expected_frame_count=97):
    if not os.path.exists(save_path):
        return False 

    cap = cv2.VideoCapture(save_path)
    if not cap.isOpened():
        print(f"Warning: failed to open {save_path}")
        return False  
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return frame_count >= expected_frame_count

def cal_min_frame(validation_videos):
    """
    Calculate the minimum number of frames among a list of .mp4 video files,
    and adjust it by adding 40 unless all videos have 0 frames, in which case return 49.

    Args:
        validation_videos (list of str): List of video file paths.

    Returns:
        int: Adjusted minimum frame count.
    """
    frame_counts = []
    
    for video_path in validation_videos:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            frame_counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
    
    if not frame_counts or min(frame_counts) == 0:
        return 49  # fallback for empty or unreadable videos

    return min(frame_counts) + 40

def generate_video(
    validation_prompts_dir: str,
    validation_videos_dir: str,
    validation_previous_videos_dir:str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 81,
    width: Optional[int] = None,
    height: Optional[int] = None,
    output_path: str = "./output.mp4",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    spatail_guidance_scale: float = 0.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    fps: int = 16,
    batch: int = -1,
    total_batch: int = 5,
    zero_pad: bool = False,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - num_frames (int): Number of frames to generate. CogVideoX1.0 generates 49 frames for 6 seconds at 8 fps, while CogVideoX1.5 produces either 81 or 161 frames, corresponding to 5 seconds or 10 seconds at 16 fps.
    - width (int): The width of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - height (int): The height of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    """

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    image = None
    video = None
    output_base_path = output_path
    validation_prompts = load_prompts(validation_prompts_dir)
    validation_videos = load_prompts(validation_videos_dir)
    if validation_previous_videos_dir!= None and os.path.exists(validation_previous_videos_dir):
        validation_previous_videos = load_prompts(validation_previous_videos_dir)
    else:
        validation_previous_videos = [None] * len(validation_videos)

    
    model_name = model_path.split("/")[-1].lower()
    desired_resolution = RESOLUTION_MAP[model_name]

    pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype,)

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(
            lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1"
        )
        pipe.fuse_lora(components=["transformer"], lora_scale=1 / lora_rank)


        # pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        # pipe.fuse_lora(components=["transformer"], lora_scale=1.0)

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
    # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")
    pipe = pipe.to("cuda")
    
    # pipe.enable_model_cpu_offload()
    # pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    expected_frame_count = cal_min_frame(validation_previous_videos)
    expected_frame_count=180
    print(f"expected_frame_count : {expected_frame_count}")
    
    total_validation_prompts = []
    total_validation_videos = []
    total_validation_previous_videos = []
    
    for prompt, video, previous_video in zip(validation_prompts, validation_videos, validation_previous_videos):
        if previous_video is not None and should_skip_video(previous_video, expected_frame_count=expected_frame_count):
            print(f"Video already exists and has the expected number of frames. Skipping generation for {previous_video}")
            continue
        total_validation_prompts.append(prompt)
        total_validation_videos.append(video)
        total_validation_previous_videos.append(previous_video)
    validation_prompts = total_validation_prompts
    validation_videos = total_validation_videos
    validation_previous_videos = total_validation_previous_videos
    
    if batch!=  -1:
        data_len = len(validation_prompts)
        validation_prompts = validation_prompts[int(batch/total_batch* data_len):int((batch+1)/total_batch* data_len)]
        validation_videos = validation_videos[int(batch/total_batch* data_len):int((batch+1)/total_batch* data_len)]
        validation_previous_videos = validation_previous_videos[int(batch/total_batch* data_len):int((batch+1)/total_batch* data_len)]
    
    
    # for prompt,video,previous_video in zip(validation_prompts, validation_videos, validation_previous_videos):
    for prompt, video, previous_video in tqdm(zip(validation_prompts, validation_videos, validation_previous_videos), desc="Processing videos", total=len(validation_prompts)):
        try:
            video_name = os.path.basename(video)
            output_path = os.path.join(output_base_path, f"{video_name}")
            if previous_video is not None and should_skip_video(previous_video, expected_frame_count=expected_frame_count):
                print(f"Video already exists and has the expected number of frames. Skipping generation for {previous_video}")
                continue
            video = load_video(video)
            # 4. Generate the video frames based on the prompt.
            # `num_frames` is the Number of frames to generate.

            video_generate = pipe(
                height=height,
                width=width,
                prompt=prompt,
                video=video,  # The path of the video to be used as the background of the video
                cond_video=video,  # The path of the video to be used as the background of the video
                num_videos_per_prompt=num_videos_per_prompt,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                use_dynamic_cfg=True,
                guidance_scale=guidance_scale,
                spatail_guidance_scale = spatail_guidance_scale,
                generator=torch.Generator().manual_seed(seed),  # Set the seed for reproducibility
                zeropad=zero_pad,
            ).frames[0]
            
            if previous_video is None:
                export_to_video(video_generate, output_path, fps=fps)
                
            elif os.path.exists(previous_video):
                previous_video_video = load_video(previous_video)
                video_generate = previous_video_video + video_generate[9:] # for dynamic
                # video_generate = previous_video_video + video_generate[1:] # for re10k
                export_to_video(video_generate, previous_video, fps=fps)
                video2image(previous_video, output_base_path)
            else:
                os.makedirs(os.path.dirname(previous_video), exist_ok=True)
                export_to_video(video_generate, previous_video, fps=fps)
                video2image(previous_video, output_base_path)
            
        except:pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument(
        "--validation_prompts_dir",
        type=str,
        default=None,
        help="The path of the validation prompts to be used for evaluation",
    )
    parser.add_argument(
        "--validation_videos_dir",
        type=str,
        default=None,
        help="The path of the validation videos to be used for evaluation",
    )
    parser.add_argument(
        "--validation_previous_videos_dir",
        type=str,
        default=None,
        help="The path of the validation previous videos to be used for evaluation",
    )
    parser.add_argument(
        "--model_path", type=str, default="THUDM/CogVideoX1.5-5B", help="Path of the pre-trained model use"
    )
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument("--output_path", type=str, default="./output.mp4", help="The path save generated video")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument("--spatail_guidance_scale", type=float, default=0.0, help="The scale for classifier-free guidance")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of steps for the inference process")
    parser.add_argument("--width", type=int, default=None, help="The width of the generated video")
    parser.add_argument("--height", type=int, default=None, help="The height of the generated video")
    parser.add_argument("--fps", type=int, default=16, help="The frames per second for the generated video")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument("--generate_type", type=str, default="t2v", help="The type of video generation")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="The data type for computation")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--batch", type=int, default=-1, help="batch size")
    parser.add_argument("--total_batch", type=int, default=5, help="batch size")
    parser.add_argument("--expected_frame_count", type=int, default=49, help="batch size")
    parser.add_argument("--zero_pad", action='store_true', help="use zero pad or not")
    

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    
    generate_video(
        validation_prompts_dir=args.validation_prompts_dir,
        validation_videos_dir=args.validation_videos_dir,
        validation_previous_videos_dir=args.validation_previous_videos_dir,
        model_path=args.model_path,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        output_path=args.output_path,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        spatail_guidance_scale=args.spatail_guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        dtype=dtype,
        generate_type=args.generate_type,
        seed=args.seed,
        fps=args.fps,
        batch=args.batch,
        total_batch=args.total_batch,
        zero_pad=args.zero_pad,
    )
