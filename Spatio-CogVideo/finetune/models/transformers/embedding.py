import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from diffusers.models.embeddings import CogVideoXPatchEmbed
from diffusers.models.activations import FP32SiLU, get_activation
from diffusers.models.attention_processor import Attention
from typing_extensions import override

class CondPatchEmbed(CogVideoXPatchEmbed):

    @override
    def forward(self, text_embeds: torch.Tensor, image_embeds: torch.Tensor, cond_embeds: torch.Tensor):
        r"""
        Args:
            text_embeds (`torch.Tensor`):
                Input text embeddings. Expected shape: (batch_size, seq_length, embedding_dim).
            image_embeds (`torch.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
        """
        text_embeds = self.text_proj(text_embeds)

        batch_size, num_frames, channels, height, width = image_embeds.shape

        if self.patch_size_t is None:
            image_embeds = image_embeds.reshape(-1, channels, height, width)
            image_embeds = self.proj(image_embeds)
            image_embeds = image_embeds.view(batch_size, num_frames, *image_embeds.shape[1:])
            image_embeds = image_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
            image_embeds = image_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]

            cond_embeds = cond_embeds.reshape(-1, channels, height, width)
            cond_embeds = self.proj(cond_embeds)
            cond_embeds = cond_embeds.view(batch_size, num_frames, *cond_embeds.shape[1:])
            cond_embeds = cond_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
            cond_embeds = cond_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]
        else:
            p = self.patch_size
            p_t = self.patch_size_t

            image_embeds = image_embeds.permute(0, 1, 3, 4, 2)
            image_embeds = image_embeds.reshape(
                batch_size, num_frames // p_t, p_t, height // p, p, width // p, p, channels
            )
            image_embeds = image_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(4, 7).flatten(1, 3)
            image_embeds = self.proj(image_embeds)

            cond_embeds = cond_embeds.permute(0, 1, 3, 4, 2)
            cond_embeds = cond_embeds.reshape(
                batch_size, num_frames // p_t, p_t, height // p, p, width // p, p, channels
            )
            cond_embeds = cond_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(4, 7).flatten(1, 3)
            cond_embeds = self.proj(cond_embeds)

        embeds = torch.cat(
            [text_embeds, image_embeds], dim=1
        ).contiguous()  # [batch, seq_length + num_frames x height x width, channels]

        cond_full_embeds = torch.cat(
            [text_embeds, cond_embeds], dim=1
        ).contiguous()  # [batch, seq_length + num_frames x height x width, channels]

        if self.use_positional_embeddings or self.use_learned_positional_embeddings:
            if self.use_learned_positional_embeddings and (self.sample_width != width or self.sample_height != height):
                raise ValueError(
                    "It is currently not possible to generate videos at a different resolution that the defaults. This should only be the case with 'THUDM/CogVideoX-5b-I2V'."
                    "If you think this is incorrect, please open an issue at https://github.com/huggingface/diffusers/issues."
                )

            pre_time_compression_frames = (num_frames - 1) * self.temporal_compression_ratio + 1

            if (
                self.sample_height != height
                or self.sample_width != width
                or self.sample_frames != pre_time_compression_frames
            ):
                pos_embedding = self._get_positional_embeddings(
                    height, width, pre_time_compression_frames, device=embeds.device
                )
            else:
                pos_embedding = self.pos_embedding

            pos_embedding = pos_embedding.to(dtype=embeds.dtype)
            embeds = embeds + pos_embedding
            cond_full_embeds = cond_full_embeds + pos_embedding

            # [batch, seq_length + 2 x num_frames x height x width, channels]
            embeds = torch.cat([embeds, cond_full_embeds[:, text_embeds.shape[1]:]], dim=1).contiguous()

        return embeds