import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override
import os
import natsort
import numpy as np
from tqdm import tqdm
from packaging import version as pver

from finetune.constants import LOG_LEVEL, LOG_NAME

from .utils import (
    load_images,
    load_images_from_videos,
    load_prompts,
    load_videos,
    preprocess_image_with_resize,
    preprocess_video_with_buckets,
    preprocess_video_with_resize,
    load_short_videos_from_videos,
)


if TYPE_CHECKING:
    from finetune.trainer import Trainer

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(LOG_NAME, LOG_LEVEL)


class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        c2w_mat = np.array(entry[5:]).reshape(3, 4) # for megasam
        c2w_mat_4x4 = np.eye(4)
        c2w_mat_4x4[:3, :] = c2w_mat
        w2c_mat_4x4 = np.linalg.inv(c2w_mat_4x4)
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = c2w_mat_4x4

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def ray_condition(K, c2w, H, W, device, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)                          # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker


class BaseV2VDataset(Dataset):
    """
    Base dataset class for Image-to-Video (I2V) training.

    This dataset loads prompts, videos and corresponding conditioning images for I2V training.

    Args:
        data_root (str): Root directory containing the dataset files
        caption_column (str): Path to file containing text prompts/captions
        video_column (str): Path to file containing video paths
        image_column (str): Path to file containing image paths
        device (torch.device): Device to load the data on
        encode_video_fn (Callable[[torch.Tensor], torch.Tensor], optional): Function to encode videos
    """

    def __init__(
        self,
        data_root: str,
        caption_column: str,
        video_column: str,
        device: torch.device,
        cond_video_column: str = None,
        short_video_column: str = None,
        trainer: "Trainer" = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        
        if str(data_root).endswith("1K"): 
            # DL3DV dataset
            self.short_video_length = trainer.args.short_video_length
            self.short_loader = trainer.args.short_loader
            
            data_root = Path(data_root)
            self.prompts = load_prompts(data_root.parent.parent / caption_column)
            self.videos = load_videos(data_root.parent.parent / video_column)
            
            self.prompts = self.prompts[:2940]
            self.videos = self.videos[:2940]
            
            if short_video_column is not None:
                self.short_videos = load_prompts(data_root.parent.parent / short_video_column)
            else:
                self.short_videos = load_short_videos_from_videos(self.videos, self.short_video_length)
                
            if cond_video_column is not None:
                self.cond_videos = load_prompts(data_root / 'data' / cond_video_column)
            else :
                self.cond_videos = None
            
        else:        
            # for Openvid1m
            self.short_video_length = trainer.args.short_video_length
            self.short_loader = trainer.args.short_loader
            data_root = Path(data_root)
            self.prompts = load_prompts(data_root / caption_column)
            self.videos = load_videos(data_root / video_column)
            self.cond_videos = load_prompts(data_root / cond_video_column)
            
            if self.short_loader:
                self.prompts = self.prompts[:5]
                self.videos = self.videos[:5]
                self.cond_videos = self.cond_videos[:5]
                
            # if trainer.args.data_preprocess_batch != -1:
            #     data_len = len(self.videos)
            #     total_gpus = trainer.args.total_gpus
            #     self.videos = self.videos[int(trainer.args.data_preprocess_batch/total_gpus* data_len):int((trainer.args.data_preprocess_batch+1)/total_gpus* data_len)]    
            #     self.cond_videos = self.cond_videos[int(trainer.args.data_preprocess_batch/total_gpus* data_len):int((trainer.args.data_preprocess_batch+1)/total_gpus* data_len)]    
            #     self.prompts = self.prompts[int(trainer.args.data_preprocess_batch/total_gpus* data_len):int((trainer.args.data_preprocess_batch+1)/total_gpus* data_len)]    
                
            if trainer.args.inverse:
                self.videos = self.videos[::-1]
                self.cond_videos = self.cond_videos[::-1]
                self.prompts = self.prompts[::-1]
                
        self.trainer = trainer
        self.device = device
        self.encode_video = trainer.encode_video
        self.encode_text = trainer.encode_text
        
        from diffusers.video_processor import VideoProcessor
        vae_scale_factor_spatial = (2 ** (len(self.trainer.components.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8)
        self.video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)
    

        # Check if number of prompts matches number of videos and conditional videos
        # if not (len(self.videos) == len(self.prompts) == len(self.cond_videos)):
        #     raise ValueError(
        #         f"Expected length of prompts, videos and conditional videos to be the same but found {len(self.prompts)=}, {len(self.videos)=} and {len(self.cond_videos)=}. Please ensure that the number of caption prompts, videos and conditional videos match in your dataset."
        #     )

        # Check if all video files exist
        if any(not path.is_file() for path in self.videos):
            raise ValueError(
                f"Some video files were not found. Please ensure that all video files exist in the dataset directory. Missing file: {next(path for path in self.videos if not path.is_file())}"
            )

        # Check if all conditional video files exist
        # if any(not path.is_file() for path in self.short_videos):
        #     raise ValueError(
        #         f"Some conditional video files were not found. Please ensure that all conditional video files exist in the dataset directory. Missing file: {next(path for path in self.cond_videos if not path.is_file())}"
        #     )


    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index

        prompt = self.prompts[index]
        video = self.videos[index]
        image = self.images[index]
        train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)

        cache_dir = self.trainer.args.data_root / "cache"
        # video_latent_dir = cache_dir / "video_latent" / self.trainer.args.model_name / train_resolution_str
        video_latent_dir = cache_dir / "video_latent" / self.trainer.args.cache_config / train_resolution_str
        prompt_embeddings_dir = cache_dir / "prompt_embeddings"
        video_latent_dir.mkdir(parents=True, exist_ok=True)
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)

        prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
        prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
        encoded_video_path = video_latent_dir / (video.stem + ".safetensors")

        if prompt_embedding_path.exists():
            prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
            logger.debug(
                f"process {self.trainer.accelerator.process_index}: Loaded prompt embedding from {prompt_embedding_path}",
                main_process_only=False,
            )
        else:
            prompt_embedding = self.encode_text(prompt)
            prompt_embedding = prompt_embedding.to("cpu")
            # [1, seq_len, hidden_size] -> [seq_len, hidden_size]
            prompt_embedding = prompt_embedding[0]
            save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
            logger.info(f"Saved prompt embedding to {prompt_embedding_path}", main_process_only=False)

        if encoded_video_path.exists():
            encoded_video = load_file(encoded_video_path)["encoded_video"]
            logger.debug(f"Loaded encoded video from {encoded_video_path}", main_process_only=False)
            # shape of image: [C, H, W]
            _, image = self.preprocess(None, self.images[index])
            image = self.image_transform(image)
        else:
            frames, image = self.preprocess(video, image)
            frames = frames.to(self.device)
            image = image.to(self.device)
            image = self.image_transform(image)
            # Current shape of frames: [F, C, H, W]
            frames = self.video_transform(frames)

            # Convert to [B, C, F, H, W]
            frames = frames.unsqueeze(0)
            frames = frames.permute(0, 2, 1, 3, 4).contiguous()
            encoded_video = self.encode_video(frames)

            # [1, C, F, H, W] -> [C, F, H, W]
            encoded_video = encoded_video[0]
            encoded_video = encoded_video.to("cpu")
            image = image.to("cpu")
            save_file({"encoded_video": encoded_video}, encoded_video_path)
            logger.info(f"Saved encoded video to {encoded_video_path}", main_process_only=False)

        # shape of encoded_video: [C, F, H, W]
        # shape of image: [C, H, W]
        return {
            "image": image,
            "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,
            "video_metadata": {
                "num_frames": encoded_video.shape[1],
                "height": encoded_video.shape[2],
                "width": encoded_video.shape[3],
            },
        }

    def preprocess(self, video_path: Path | None, image_path: Path | None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and preprocesses a video and an image.
        If either path is None, no preprocessing will be done for that input.

        Args:
            video_path: Path to the video file to load
            image_path: Path to the image file to load

        Returns:
            A tuple containing:
                - video(torch.Tensor) of shape [F, C, H, W] where F is number of frames,
                  C is number of channels, H is height and W is width
                - image(torch.Tensor) of shape [C, H, W]
        """
        raise NotImplementedError("Subclass must implement this method")

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to a video.

        Args:
            frames (torch.Tensor): A 4D tensor representing a video
                with shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed video tensor
        """
        raise NotImplementedError("Subclass must implement this method")

    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to an image.

        Args:
            image (torch.Tensor): A 3D tensor representing an image
                with shape [C, H, W] where:
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed image tensor
        """
        raise NotImplementedError("Subclass must implement this method")


class V2VDatasetWithResize(BaseV2VDataset):
    """
    A dataset class for image-to-video generation that resizes inputs to fixed dimensions.

    This class preprocesses videos and images by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width
    - Images are resized to height x width

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos and images
        width (int): Target width for resizing videos and images
    """

    def __init__(self, max_num_frames: int, height: int, width: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

        self.__frame_transforms = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(self, video_path: Path | None, image_path: Path | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if video_path is not None:
            video = preprocess_video_with_resize(video_path, self.max_num_frames, self.height, self.width)
        else:
            video = None
        if image_path is not None:
            image = preprocess_image_with_resize(image_path, self.height, self.width)
        else:
            image = None
        return video, image

    @override
    def preprocess_from_list(self, video_paths: Path | None, image_paths: Path | None, max_num_frames: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if max_num_frames == None:max_num_frames=self.max_num_frames
        if video_paths is not None:
            video = preprocess_video_with_resize(video_paths, max_num_frames, self.height, self.width)
        else:
            video = None
        if image_paths is not None:
            images = []
            for image_path in image_paths:
                image = preprocess_image_with_resize(image_path , self.height, self.width)
                images.append(image)
            image = torch.stack(images, dim=0)
        else:
            image = None
        return video, image
    
    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)



class ConditionalV2V(V2VDatasetWithResize):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @override
    def __getitem__(self, index: int) -> Dict[str, Any]:

        # almost copied from the parent class

        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index
        
        prompt = self.prompts[index]
        video_dir = self.videos[index]
        short_video_dir = self.short_videos[index]
        scene_base_name = video_dir.stem
        
        # image = video_list[0]
        train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)
        cache_dir = self.trainer.args.data_root.parent / "cache"
        
        video_latent_dir = cache_dir / self.trainer.args.cache_config / train_resolution_str / "video_latent" 
        prompt_embeddings_dir = cache_dir / self.trainer.args.cache_config / train_resolution_str / "prompt_embeddings" 
        
        prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
        
        encoded_video_path = video_latent_dir / (scene_base_name + ".safetensors")
        prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
        

        video_latent_dir.mkdir(parents=True, exist_ok=True)
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)

        # prompt
        if prompt_embedding_path.exists():
            prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
            logger.debug(
                f"process {self.trainer.accelerator.process_index}: Loaded prompt embedding from {prompt_embedding_path}",
                main_process_only=False,)
        else:
            prompt_embedding = self.encode_text(prompt)
            prompt_embedding = prompt_embedding.to("cpu")
            # [1, seq_len, hidden_size] -> [seq_len, hidden_size]
            prompt_embedding = prompt_embedding[0]
            save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
            logger.info(f"Saved prompt embedding to {prompt_embedding_path}", main_process_only=False)

        
        # original video
        if encoded_video_path.exists():
            encoded_video = load_file(encoded_video_path)["encoded_video"]
            logger.debug(f"Loaded encoded video from {encoded_video_path}", main_process_only=False)
        else:
            frames, _ = self.preprocess_from_list(video_dir, None)
            frames = frames.to(self.device)
            # Current shape of frames: [F, C, H, W]
            frames = self.video_transform(frames)

            # Convert to [B, C, F, H, W]
            frames = frames.unsqueeze(0)
            frames = frames.permute(0, 2, 1, 3, 4).contiguous()
            encoded_video = self.encode_video(frames)
            
            # [1, C, F, H, W] -> [C, F, H, W]
            encoded_video = encoded_video[0]
            encoded_video = encoded_video.to("cpu")
            
            save_file({"encoded_video": encoded_video}, encoded_video_path)
            logger.info(f"Saved encoded video to {encoded_video_path}", main_process_only=False)
            
            # fixed_encoded_video = []
            # for i in range(13):
            #     fixed_encoded_video.append(self.encode_video(frames[:,:,29:30]))
            # fixed_encoded_video = torch.cat(fixed_encoded_video, dim=2)
            # # warping
            # fixed_encoded_video = torch.cat([encoded_video[:, :, 1:2] for i in range(13)], dim=2)
            # decoded_video = self.decode_latents(latents = encoded_video.permute(0, 2, 1, 3, 4)) # channel = 16
            # video = self.video_processor.postprocess_video(video=decoded_video, output_type='np')
            # self.save_video_frames(video, '/root/JB/project/Spatio-CogVideo/test2/test.mp4')
            

            # Save each frame as an image
            # output_dir.mkdir(parents=True, exist_ok=True)
            # from torchvision.utils import save_image
            # for i, frame in enumerate(video.permute(0, 2, 1, 3, 4)[0]):
            #     save_image(frame,  f"{output_dir}/frame_{i:04d}.png")
            
            
        # Current shape of short_video: [F, C, H, W]    
        short_video, _ = self.preprocess_from_list(short_video_dir, None, max_num_frames=self.short_video_length)
        short_video = self.video_transform(short_video)
        # Convert to [C, F, H, W]
        short_video = short_video.permute(1, 0, 2, 3).contiguous()
        
        
        
        # short_video = encoded_video
        
        # shape of encoded_video: [C, F, H, W]
        # shape of short_video: [C, F', H, W]
        return {
            "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,
            "short_video": short_video,
            "video_metadata": {
                "num_frames": encoded_video.shape[1],
                "height": encoded_video.shape[2],
                "width": encoded_video.shape[3],
            },
        }
        
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode the latents using the VAE.

        Args:
            latents (torch.Tensor): The latents to decode.

        Returns:
            torch.Tensor: The decoded video.
        """
        # [B, C, F, H, W] -> [B, F, C, H, W]
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.trainer.components.vae.config.scaling_factor * latents

        frames = self.trainer.components.vae.decode(latents).sample
        return frames
    
    def save_video_frames(self, video: np.ndarray, output_dir: str) -> None:
        import imageio
        import os

        """
        Save video frames as an MP4 file using imageio.

        Args:
            video (np.ndarray): The video array to save.
            output_dir (str): The directory to save the video.
        """
        # os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)

        video_np = (video[0] * 255).clip(0, 255).astype('uint8')  # (49, 480, 720, 3)
        # video_path = os.path.join(output_dir, 'output_video.mp4')
        video_path = output_dir

        fps = 16  # Adjust FPS as needed
        writer = imageio.get_writer(video_path, fps=fps)

        for frame in video_np:
            writer.append_data(frame)

        writer.close()
        print(f'Video saved at {video_path}')
        
        
    def dataset2safetensors(self):
        train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)
        cache_dir = self.trainer.args.data_root.parent / "cache"
        video_latent_dir = cache_dir / self.trainer.args.cache_config / train_resolution_str / "video_latent" 
        prompt_embeddings_dir = cache_dir / self.trainer.args.cache_config / train_resolution_str / "prompt_embeddings" 
        
        video_latent_dir.mkdir(parents=True, exist_ok=True)
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)

        batch = 32
        
        video_dirs = []
        video_save_paths = []
        scene_base_names = []
        prompt_save_dirs = []
        prompts = []    
        for index in tqdm(range(len(self.videos)), desc="Processing videos"):
            prompt = self.prompts[index]
            video_dir = self.videos[index]
            scene_base_name = video_dir.stem
            
            prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
            encoded_video_path = video_latent_dir / (scene_base_name + ".safetensors")
            prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
            if prompt_embedding_path.exists()!=True:
                prompt_embedding = self.encode_text(prompt)
                prompt_embedding = prompt_embedding.to("cpu")
                # [1, seq_len, hidden_size] -> [seq_len, hidden_size]
                prompt_embedding = prompt_embedding[0]
                save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
                logger.info(f"Saved prompt embedding to {prompt_embedding_path}", main_process_only=False)

            if encoded_video_path.exists()!= True:
                video_dirs.append(video_dir)
                video_save_paths.append(encoded_video_path)
                scene_base_names.append(scene_base_name)
               
            if len(video_dirs) == batch:
                self.save_safetensor_from_vid(video_dirs, video_save_paths)
                video_dirs = []
                video_save_paths = []
                scene_base_names = []
                
        if len(video_dirs) > 0:
            self.save_safetensor_from_vid(video_dirs, video_save_paths)


    def save_safetensor_from_vid(self, video_dirs, encoded_video_paths):        
        frames = []
        for video_dir, encoded_video_path in zip(video_dirs, encoded_video_paths):
            frame, _ = self.preprocess_from_list(video_dir, None)
            frame = self.video_transform(frame)
            frame = frame.unsqueeze(0)
            frame = frame.permute(0, 2, 1, 3, 4).contiguous()
            frames.append(frame)
        frames = torch.cat(frames, dim=0).to(self.device)
        encoded_videos = self.encode_video(frames)
        for i, encoded_video_path in enumerate(encoded_video_paths):
            save_file({"encoded_video": encoded_videos[i].to("cpu")}, encoded_video_path)
            logger.info(f"Saved encoded video to {encoded_video_path}", main_process_only=False)
            
            
            

class ConditionalV2VDataset_DL3DV(V2VDatasetWithResize):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @override
    def __getitem__(self, index: int) -> Dict[str, Any]:

        # almost copied from the parent class

        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index

        prompt = self.prompts[index]
        video_dir = self.videos[index]
        short_video_dir = self.short_videos[index]
        scene_base_name = video_dir.stem
        
        train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)

        cache_dir = self.trainer.args.data_root.parent / "cache"
        
        video_latent_dir = cache_dir / self.trainer.args.cache_config / train_resolution_str / "video_latent" 
        cond_latent_dir = cache_dir / self.trainer.args.cache_config / train_resolution_str / "condition_latent"
        prompt_embeddings_dir = cache_dir / self.trainer.args.cache_config / train_resolution_str / "prompt_embeddings" 
        
        
        prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
        
        encoded_video_path = video_latent_dir / (scene_base_name + ".safetensors")
        prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
        encoded_cond_video_path = cond_latent_dir / (scene_base_name + ".safetensors")

        # prompt
        if prompt_embedding_path.exists():
            prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
            logger.debug(
                f"process {self.trainer.accelerator.process_index}: Loaded prompt embedding from {prompt_embedding_path}",
                main_process_only=False,)
        
        # original video
        if encoded_video_path.exists():
            encoded_video = load_file(encoded_video_path)["encoded_video"]
            logger.debug(f"Loaded encoded video from {encoded_video_path}", main_process_only=False)
        
        # conditional video
        if encoded_cond_video_path.exists():
            encoded_condition = load_file(encoded_cond_video_path)["encoded_condition"]
            logger.debug(f"Loaded encoded condition video from {encoded_cond_video_path}", main_process_only=False)
            
        # short video
        if short_video_dir.exists():
            short_video, _ = self.preprocess_from_list(short_video_dir, None, max_num_frames=self.short_video_length)
            short_video = self.video_transform(short_video)
            # Convert to [C, F, H, W]
            short_video = short_video.permute(1, 0, 2, 3).contiguous()
            
        # shape of encoded_video: [C, F, H, W]
        # shape of image: [C, H, W]
        return {
            "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,
            "encoded_condition":encoded_condition,
            "short_video": short_video,
            "video_metadata": {
                "num_frames": encoded_video.shape[1],
                "height": encoded_video.shape[2],
                "width": encoded_video.shape[3],
            },
        }
        
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode the latents using the VAE.

        Args:
            latents (torch.Tensor): The latents to decode.

        Returns:
            torch.Tensor: The decoded video.
        """
        # [B, C, F, H, W] -> [B, F, C, H, W]
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.trainer.components.vae.config.scaling_factor * latents

        frames = self.trainer.components.vae.decode(latents).sample
        return frames
        
    def save_video_frames(self, video: np.ndarray, output_dir: str) -> None:
        import imageio
        import os

        """
        Save video frames as an MP4 file using imageio.

        Args:
            video (np.ndarray): The video array to save.
            output_dir (str): The directory to save the video.
        """
        os.makedirs(output_dir, exist_ok=True)

        video_np = (video[0] * 255).clip(0, 255).astype('uint8')  # (49, 480, 720, 3)
        video_path = os.path.join(output_dir, 'output_video.mp4')

        fps = 3  # Adjust FPS as needed
        writer = imageio.get_writer(video_path, fps=fps)

        for frame in video_np:
            writer.append_data(frame)

        writer.close()
        print(f'Video saved at {video_path}')
        
        
        
    def dataset2safetensors(self):
        train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)
        cache_dir = self.trainer.args.data_root.parent / "cache"
        video_latent_dir = cache_dir / self.trainer.args.cache_config / train_resolution_str / "video_latent" 
        prompt_embeddings_dir = cache_dir / self.trainer.args.cache_config / train_resolution_str / "prompt_embeddings" 
        
        video_latent_dir.mkdir(parents=True, exist_ok=True)
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)

        batch = 32
        
        video_dirs = []
        video_save_paths = []
        scene_base_names = []
        
        for index in tqdm(range(len(self.videos)), desc="Processing videos"):
            prompt = self.prompts[index]
            video_dir = self.videos[index]
            scene_base_name = video_dir.stem
            
            prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
            encoded_video_path = video_latent_dir / (scene_base_name + ".safetensors")
            prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
            if prompt_embedding_path.exists()!=True:
                prompt_embedding = self.encode_text(prompt)
                prompt_embedding = prompt_embedding.to("cpu")
                # [1, seq_len, hidden_size] -> [seq_len, hidden_size]
                prompt_embedding = prompt_embedding[0]
                save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
                logger.info(f"Saved prompt embedding to {prompt_embedding_path}", main_process_only=False)

            if encoded_video_path.exists()!= True:
                video_dirs.append(video_dir)
                video_save_paths.append(encoded_video_path)
                scene_base_names.append(scene_base_name)
               
            if len(video_dirs) == batch:
                self.save_safetensor_from_vid(video_dirs, video_save_paths)
                video_dirs = []
                video_save_paths = []
                scene_base_names = []
                
        if len(video_dirs) > 0:
            self.save_safetensor_from_vid(video_dirs, video_save_paths)


    def save_safetensor_from_vid(self, video_dirs, encoded_video_paths):        
        frames = []
        for video_dir, encoded_video_path in zip(video_dirs, encoded_video_paths):
            frame, _ = self.preprocess_from_list(video_dir, None)
            frame = self.video_transform(frame)
            frame = frame.unsqueeze(0)
            frame = frame.permute(0, 2, 1, 3, 4).contiguous()
            frames.append(frame)
        frames = torch.cat(frames, dim=0).to(self.device)
        encoded_videos = self.encode_video(frames)
        for i, encoded_video_path in enumerate(encoded_video_paths):
            save_file({"encoded_video": encoded_videos[i].to("cpu")}, encoded_video_path)
            logger.info(f"Saved encoded video to {encoded_video_path}", main_process_only=False)
            
            
            
class ConditionalV2VDataset_OpenVid(V2VDatasetWithResize):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @override
    def __getitem__(self, index: int) -> Dict[str, Any]:

        # almost copied from the parent class

        if isinstance(index, list):
            # Here, index is actually a list of data objects that we need to return.
            # The BucketSampler should ideally return indices. But, in the sampler, we'd like
            # to have information about num_frames, height and width. Since this is not stored
            # as metadata, we need to read the video to get this information. You could read this
            # information without loading the full video in memory, but we do it anyway. In order
            # to not load the video twice (once to get the metadata, and once to return the loaded video
            # based on sampled indices), we cache it in the BucketSampler. When the sampler is
            # to yield, we yield the cache data instead of indices. So, this special check ensures
            # that data is not loaded a second time. PRs are welcome for improvements.
            return index

        prompt = self.prompts[index]
        video_dir = self.videos[index]
        cond_video_dir = self.cond_videos[index]
        scene_base_name = video_dir.stem
        
        train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)

        cache_dir = self.trainer.args.data_root.parent / "cache"
        
        video_latent_dir = cache_dir / self.trainer.args.cache_config / train_resolution_str / "video_latent" 
        cond_latent_dir = cache_dir / self.trainer.args.cache_config / train_resolution_str / "cond_video_latent"
        prompt_embeddings_dir = cache_dir / self.trainer.args.cache_config / train_resolution_str / "prompt_embeddings" 
        
        
        prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
        
        encoded_video_path = video_latent_dir / (scene_base_name + ".safetensors")
        prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
        encoded_cond_video_path = cond_latent_dir / (scene_base_name + ".safetensors")
        

        # prompt
        if prompt_embedding_path.exists():
            prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
            logger.debug(
                f"process {self.trainer.accelerator.process_index}: Loaded prompt embedding from {prompt_embedding_path}",
                main_process_only=False,)
        
        # original video
        if encoded_video_path.exists():
            encoded_video = load_file(encoded_video_path)["encoded_video"]
            logger.debug(f"Loaded encoded video from {encoded_video_path}", main_process_only=False)
        
        # conditional video
        if encoded_cond_video_path.exists():
            encoded_condition = load_file(encoded_cond_video_path)["encoded_video"]
            logger.debug(f"Loaded encoded condition video from {encoded_cond_video_path}", main_process_only=False)
            
        # shape of encoded_video: [C, F, H, W]
        # shape of image: [C, H, W]
        return {
            "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,
            "encoded_condition":encoded_condition,
            "video_metadata": {
                "num_frames": encoded_video.shape[1],
                "height": encoded_video.shape[2],
                "width": encoded_video.shape[3],
            },
        }
        
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode the latents using the VAE.

        Args:
            latents (torch.Tensor): The latents to decode.

        Returns:
            torch.Tensor: The decoded video.
        """
        # [B, C, F, H, W] -> [B, F, C, H, W]
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.trainer.components.vae.config.scaling_factor * latents

        frames = self.trainer.components.vae.decode(latents).sample
        return frames
        
    def save_video_frames(self, video: np.ndarray, output_dir: str) -> None:
        import imageio
        import os

        """
        Save video frames as an MP4 file using imageio.

        Args:
            video (np.ndarray): The video array to save.
            output_dir (str): The directory to save the video.
        """
        os.makedirs(output_dir, exist_ok=True)

        video_np = (video[0] * 255).clip(0, 255).astype('uint8')  # (49, 480, 720, 3)
        video_path = os.path.join(output_dir, 'output_video.mp4')

        fps = 3  # Adjust FPS as needed
        writer = imageio.get_writer(video_path, fps=fps)

        for frame in video_np:
            writer.append_data(frame)

        writer.close()
        print(f'Video saved at {video_path}')
        
        
        
    def dataset2safetensors(self):
        train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)
        
        cache_dir = self.trainer.args.data_root.parent / "cache"
        video_latent_dir = cache_dir / self.trainer.args.cache_config / train_resolution_str / "video_latent" 
        cond_video_latent_dir = cache_dir / self.trainer.args.cache_config / train_resolution_str / "cond_video_latent"
        prompt_embeddings_dir = cache_dir / self.trainer.args.cache_config / train_resolution_str / "prompt_embeddings" 
        
        video_latent_dir.mkdir(parents=True, exist_ok=True)
        cond_video_latent_dir.mkdir(parents=True, exist_ok=True)
        prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)

        batch = 32
        
        video_dirs_temp = []    
        cond_video_dirs_temp = []
        prompt_dirs_temp = []
        
        for index in tqdm(range(len(self.videos)), desc="Processing videos"):
            prompt = self.prompts[index]
            video_dir = self.videos[index]
            cond_video_dir = self.cond_videos[index]
            scene_base_name = video_dir.stem
            
            prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
            encoded_video_path = video_latent_dir / (scene_base_name + ".safetensors")
            encoded_cond_video_path = cond_video_latent_dir / (scene_base_name + ".safetensors")
            prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
            
            
            if prompt_embedding_path.exists()!=True or encoded_video_path.exists()!= True or encoded_cond_video_path.exists()!= True:    
                video_dirs_temp.append(video_dir)
                cond_video_dirs_temp.append(cond_video_dir)
                prompt_dirs_temp.append(prompt)
                
            
                
        if self.trainer.args.data_preprocess_batch != -1:
            data_len = len(video_dirs_temp)
            total_gpus = self.trainer.args.total_gpus
            video_dirs_temp = video_dirs_temp[int(self.trainer.args.data_preprocess_batch/total_gpus* data_len):int((self.trainer.args.data_preprocess_batch+1)/total_gpus* data_len)]    
            cond_video_dirs_temp = cond_video_dirs_temp[int(self.trainer.args.data_preprocess_batch/total_gpus* data_len):int((self.trainer.args.data_preprocess_batch+1)/total_gpus* data_len)]    
            prompt_dirs_temp = prompt_dirs_temp[int(self.trainer.args.data_preprocess_batch/total_gpus* data_len):int((self.trainer.args.data_preprocess_batch+1)/total_gpus* data_len)]    
        
        video_dirs = []
        video_save_paths = []
        scene_base_names = []
        
        # for index in tqdm(range(len(self.videos)), desc="Processing videos"):
        for index in tqdm(range(len(video_dirs_temp)), desc="Processing videos"):
            # prompt = self.prompts[index]
            # video_dir = self.videos[index]
            # cond_video_dir = self.cond_videos[index]
            prompt = prompt_dirs_temp[index]
            video_dir = video_dirs_temp[index]
            cond_video_dir = cond_video_dirs_temp[index]
            scene_base_name = video_dir.stem
            
            prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
            encoded_video_path = video_latent_dir / (scene_base_name + ".safetensors")
            encoded_cond_video_path = cond_video_latent_dir / (scene_base_name + ".safetensors")
            prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
            
            if prompt_embedding_path.exists()!=True:
                prompt_embedding = self.encode_text(prompt)
                prompt_embedding = prompt_embedding.to("cpu")
                # [1, seq_len, hidden_size] -> [seq_len, hidden_size]
                prompt_embedding = prompt_embedding[0]
                save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
                logger.info(f"Saved prompt embedding to {prompt_embedding_path}", main_process_only=False)
            if encoded_video_path.exists()!= True:
                video_dirs.append(video_dir)
                video_save_paths.append(encoded_video_path)
                scene_base_names.append(scene_base_name)
            if encoded_cond_video_path.exists()!= True:
                video_dirs.append(cond_video_dir)
                video_save_paths.append(encoded_cond_video_path)
                scene_base_names.append(scene_base_name)
               
            if len(video_dirs) >= batch:
                self.save_safetensor_from_vid(video_dirs, video_save_paths)
                video_dirs = []
                video_save_paths = []
                scene_base_names = []
        if len(video_dirs) > 0:
            self.save_safetensor_from_vid(video_dirs, video_save_paths)


    def save_safetensor_from_vid(self, video_dirs, encoded_video_paths):        
        frames = []
        for video_dir, encoded_video_path in zip(video_dirs, encoded_video_paths):
            print(video_dir)
            frame, _ = self.preprocess_from_list(video_dir, None)
            frame = self.video_transform(frame)
            frame = frame.unsqueeze(0)
            frame = frame.permute(0, 2, 1, 3, 4).contiguous()
            frames.append(frame)
        frames = torch.cat(frames, dim=0).to(self.device)
        encoded_videos = self.encode_video(frames)
        for i, encoded_video_path in enumerate(encoded_video_paths):
            save_file({"encoded_video": encoded_videos[i].to("cpu")}, encoded_video_path)
            logger.info(f"Saved encoded video to {encoded_video_path}", main_process_only=False)
            
            
            