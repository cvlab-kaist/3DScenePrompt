from typing_extensions import override
import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import PIL
import torch
from transformers import T5EncoderModel, T5Tokenizer

from diffusers.pipelines.cogvideo.pipeline_cogvideox_image2video import *

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import CogVideoXLoraLoaderMixin
from diffusers.models import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
from diffusers import CogVideoXPipeline
from ..transformers.transformer_das import CogVideoXTransformer3DModelTracking

from PIL import Image

class CogVideoXI2VDASPipeline(CogVideoXImageToVideoPipeline, DiffusionPipeline):

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModelTracking,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)
        
        # if not isinstance(self.transformer, CogVideoXTransformer3DModelTracking):
        #     raise ValueError("The transformer in this pipeline must be of type CogVideoXTransformer3DModelTracking")
            
        # 打印transformer blocks的数量
        # print(f"Number of transformer blocks: {len(self.transformer.transformer_blocks)}")
        # print(f"Number of tracking transformer blocks: {len(self.transformer.transformer_blocks_copy)}")
        self.transformer = torch.compile(self.transformer)

    @torch.no_grad()
    def __call__(
        self,
        image: Union[torch.Tensor, Image.Image],
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        tracking_maps: Optional[torch.Tensor] = None,
        tracking_image: Optional[torch.Tensor] = None,
        plucker_embbeding: Optional[torch.Tensor] = None,
        cut3r_state: Optional[torch.Tensor] = None,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        # Most of the implementation remains the same as the parent class
        # We will modify the parts that need to handle tracking_maps

        # 1. Check inputs and set default values
        self.check_inputs(
            image,
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            del negative_prompt_embeds

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents
        image = self.video_processor.preprocess(image, height=height, width=width).to(
            device, dtype=prompt_embeds.dtype
        )

        if self.transformer.config.in_channels != 16:
            latent_channels = self.transformer.config.in_channels // 2
        else:
            latent_channels = self.transformer.config.in_channels
        latents, image_latents = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        del image

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        
        self.transformer.to(device=device)
        
        plucker_embbeding = plucker_embbeding.unsqueeze(0).to(latents.device, latents.dtype)  # [B, F, C, H, W]
        # camera_embbeding_temp = self.camera_embedder(plucker_embbeding)  # [1, F, C, H, W]
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            old_pred_original_sample = None
            # breakpoint()
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                latent_image_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents
                latent_model_input = torch.cat([latent_model_input, latent_image_input], dim=2)
                
                plucker_embbeding_input = torch.cat([plucker_embbeding] * 2) if do_classifier_free_guidance else plucker_embbeding
                # tracking_maps_input = torch.cat([camera_embbeding_temp, latent_image_input], dim=2)
                del latent_image_input
                
                
                cut3r_state_batch = torch.cat([cut3r_state] * 2) if cut3r_state is not None and do_classifier_free_guidance else cut3r_state.unsqueeze(dim=0)
                
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])
                
                # Predict noise
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    tracking_maps=plucker_embbeding_input,
                    cut3r_state=cut3r_state_batch,
                    return_dict=False,
                )[0]
                del latent_model_input
                # if tracking_maps_input is not None:
                    # del tracking_maps_input
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    del noise_pred_uncond, noise_pred_text

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                del noise_pred
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # 9. Post-processing
        if not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)
