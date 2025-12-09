import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from typing_extensions import override

from finetune.constants import LOG_LEVEL, LOG_NAME

from .utils import load_prompts, load_videos, preprocess_video_with_buckets, preprocess_video_with_resize
import numpy as np
from tqdm import tqdm


if TYPE_CHECKING:
    from finetune.trainer import Trainer

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

logger = get_logger(LOG_NAME, LOG_LEVEL)


class BaseT2VDataset(Dataset):
    """
    Base dataset class for Text-to-Video (T2V) training.

    This dataset loads prompts and videos for T2V training.

    Args:
        data_root (str): Root directory containing the dataset files
        caption_column (str): Path to file containing text prompts/captions
        video_column (str): Path to file containing video paths
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
            self.prompts = load_prompts(data_root / caption_column)[:12162]
            self.videos = load_videos(data_root / video_column)[:12162]
            
            if self.short_loader:
                self.prompts = self.prompts[:5]
                self.videos = self.videos[:5]
                
            if trainer.args.data_preprocess_batch != -1:
                data_len = len(self.videos)
                total_gpus = trainer.args.total_gpus
                self.videos = self.videos[int(trainer.args.data_preprocess_batch/total_gpus* data_len):int((trainer.args.data_preprocess_batch+1)/total_gpus* data_len)]    
                self.prompts = self.prompts[int(trainer.args.data_preprocess_batch/total_gpus* data_len):int((trainer.args.data_preprocess_batch+1)/total_gpus* data_len)]    
                
            if trainer.args.inverse:
                self.videos = self.videos[::-1]
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


    def preprocess(self, video_path: Path) -> torch.Tensor:
        """
        Loads and preprocesses a video.

        Args:
            video_path: Path to the video file to load.

        Returns:
            torch.Tensor: Video tensor of shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width
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
            torch.Tensor: The transformed video tensor with the same shape as the input
        """
        raise NotImplementedError("Subclass must implement this method")


class T2VDatasetWithResize(BaseT2VDataset):
    """
    A dataset class for text-to-video generation that resizes inputs to fixed dimensions.

    This class preprocesses videos by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos
        width (int): Target width for resizing videos
    """

    def __init__(self, max_num_frames: int, height: int, width: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_num_frames = max_num_frames
        self.height = height
        self.width = width

        self.__frame_transform = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])

    @override
    def preprocess(self, video_path: Path) -> torch.Tensor:
        return preprocess_video_with_resize(
            video_path,
            self.max_num_frames,
            self.height,
            self.width,
        )

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transform(f) for f in frames], dim=0)


class T2VDatasetWithBuckets(BaseT2VDataset):
    def __init__(
        self,
        video_resolution_buckets: List[Tuple[int, int, int]],
        vae_temporal_compression_ratio: int,
        vae_height_compression_ratio: int,
        vae_width_compression_ratio: int,
        *args,
        **kwargs,
    ) -> None:
        """ """
        super().__init__(*args, **kwargs)

        self.video_resolution_buckets = [
            (
                int(b[0] / vae_temporal_compression_ratio),
                int(b[1] / vae_height_compression_ratio),
                int(b[2] / vae_width_compression_ratio),
            )
            for b in video_resolution_buckets
        ]

        self.__frame_transform = transforms.Compose([transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)])

    @override
    def preprocess(self, video_path: Path) -> torch.Tensor:
        return preprocess_video_with_buckets(video_path, self.video_resolution_buckets)

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transform(f) for f in frames], dim=0)





class T2VDataset_OpenVid_ipadapter(BaseT2VDataset):
    """
    A dataset class for text-to-video generation that resizes inputs to fixed dimensions.

    This class preprocesses videos by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos
        width (int): Target width for resizing videos
    """

    def __init__(self, max_num_frames: int, height: int, width: int, *args, **kwargs) -> None:
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
        scene_base_name = video_dir.stem
        
        train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)

        cache_dir = self.trainer.args.data_root.parent / "cache"
        
        video_latent_dir = cache_dir / self.trainer.args.cache_config / train_resolution_str / "video_latent" 
        prompt_embeddings_dir = cache_dir / self.trainer.args.cache_config / train_resolution_str / "prompt_embeddings" 
        cut3r_state_dir = cache_dir / 'cut3r_cache'
        
        # video_latent_dir = cache_dir / self.trainer.args.model_name / train_resolution_str / "video_latent" 
        # cond_latent_dir = cache_dir / self.trainer.args.model_name / train_resolution_str / "cond_video_latent"
        # prompt_embeddings_dir = cache_dir / self.trainer.args.model_name / train_resolution_str / "prompt_embeddings" 
        
        
        prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
        
        encoded_video_path = video_latent_dir / (scene_base_name + ".safetensors")
        prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
        cut3r_state_path = cut3r_state_dir / ('cut3r_state_'+ scene_base_name + ".npz")

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
            
        # cut3r state
        if cut3r_state_path.exists():
            cut3r_state = torch.from_numpy(np.load(cut3r_state_path)['state_feat'])
            logger.debug(f"Loaded cut3r state from {cut3r_state_path}", main_process_only=False)
            
        # shape of encoded_video: [C, F, H, W]
        # shape of image: [C, H, W]
        return {
            "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,
            "cut3r_state": cut3r_state,
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
        
        video_dirs = []
        video_save_paths = []
        scene_base_names = []
        
        for index in tqdm(range(len(self.videos)), desc="Processing videos"):
            prompt = self.prompts[index]
            video_dir = self.videos[index]
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
            
