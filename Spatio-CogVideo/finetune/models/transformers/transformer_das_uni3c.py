from typing import Any, Dict, Optional, Tuple, Union
from typing import List, Any, Optional, Type
from typing import Any, Dict, Optional, Tuple, Union, List, Callable
import torch
from torch import nn

import os
import torch.nn.init as init

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import AttentionProcessor, CogVideoXAttnProcessor2_0, FusedCogVideoXAttnProcessor2_0
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero
from typing_extensions import override
import torch.nn.functional as F
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel, Transformer2DModelOutput

from .embedding import CondPatchEmbed

from peft.tuners.tuners_utils import BaseTunerLayer
import numpy as np
from diffusers.utils import is_torch_version
from types import MethodType


import pdb
import sys
from typing import Any

def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module

class ForkedPdb(pdb.Pdb):
    """
    PDB Subclass for debugging multi-processed code
    Suggested in: https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """

    def interaction(self, *args: Any, **kwargs: Any) -> None:
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # self.pre_norm = nn.LayerNorm(768, device="cpu")
        self.pre_norm = nn.LayerNorm(768, eps=1e-5, device="cpu")
        self.cut3r_mlp = nn.Sequential(
            nn.Linear(768, 1024, device="cpu"),
            nn.LayerNorm(1024, device="cpu"),
            nn.GELU(),

            nn.Linear(1024, 2048, device="cpu"),
            nn.LayerNorm(2048, device="cpu"),
            nn.GELU(),

            nn.Linear(2048, 3072, device="cpu"),
            nn.LayerNorm(3072, device="cpu"),
            nn.GELU(),

            nn.Linear(3072, 3072, device="cpu"),  # 마지막은 GELU 없이
            nn.LayerNorm(3072, device="cpu"),    # 마지막 정규화 추가
        )

        # ===== Weight Initialization =====
        for i, layer in enumerate(self.cut3r_mlp):
            if isinstance(layer, nn.Linear):
                if i == len(self.cut3r_mlp) - 1:
                    # 마지막 레이어 → Zero init
                    init.zeros_(layer.weight)
                    init.zeros_(layer.bias)
                else:
                    # 앞쪽 레이어들 → Kaiming Normal
                    init.kaiming_normal_(layer.weight, nonlinearity='gelu')
                    init.zeros_(layer.bias)

    def forward(self, x):
        x = self.pre_norm(x)
        x = self.cut3r_mlp(x)
        return x
    

class MaskCamEmbed(nn.Module):
    def __init__(self, add_channels=6, mid_channels=64) -> None:
        super().__init__()

        self.mask_padding = [0, 0, 0, 0, 3, 0]  # 左右上下前后, I2V
        self.mask_proj = nn.Sequential(nn.Conv3d(add_channels, mid_channels, kernel_size=(4, 8, 8), stride=(4, 8, 8), padding=(0, 1, 1)),
                                       nn.GroupNorm(mid_channels // 8, mid_channels), nn.SiLU())
        self.mask_zero_proj = zero_module(nn.Conv3d(mid_channels, 16, kernel_size=(1, 1, 1), stride=(1, 1, 1)))
        
        self._initialize_weights()

    def forward(self, add_inputs: torch.Tensor):
        # render_mask.shape [b,c,f,h,w]
        warp_add_pad = F.pad(add_inputs, self.mask_padding, mode="constant", value=0)
        add_embeds = self.mask_proj(warp_add_pad)  # [B,C,F,H,W]
        add_embeds = self.mask_zero_proj(add_embeds)
        add_embeds = add_embeds.permute(0, 2, 1, 3, 4)  # [B,F,C,H,W]

        return add_embeds

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


@maybe_allow_in_graph
class CogVideoXDASBlock(CogVideoXBlock):

    @override
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            time_embed_dim=time_embed_dim,
            dropout=dropout,
            activation_fn=activation_fn,
            attention_bias=attention_bias,
            qk_norm=qk_norm,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            final_dropout=final_dropout,
            ff_inner_dim=ff_inner_dim,
            ff_bias=ff_bias,
            attention_out_bias=attention_out_bias,
        )
        # self.cut3r_linear = SimpleMLP()


    @override
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        cut3r_state: torch.Tensor = None,
    ) -> torch.Tensor:
        original_text_seq_length = encoder_hidden_states.size(1)
        text_seq_length = encoder_hidden_states.size(1)
        attention_kwargs = attention_kwargs or {}

        # norm & modulate
        # cut3r_state = self.cut3r_linear(cut3r_state.squeeze(dim=0).to(hidden_states.device,hidden_states.dtype))
        # encoder_hidden_states = torch.cat([encoder_hidden_states, cut3r_state], dim=1)
        # text_seq_length = encoder_hidden_states.size(1)
        
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )
        
        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **attention_kwargs,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]
        
        return hidden_states, encoder_hidden_states



@maybe_allow_in_graph
class CogVideoXTransformer3DModelTracking(CogVideoXTransformer3DModel, ModelMixin):
    """
    Add tracking maps to the CogVideoX transformer model.

    Parameters:
        num_tracking_blocks (`int`, defaults to `18`):
            The number of tracking blocks to use. Must be less than or equal to num_layers.
    """

    def __init__(
        self,
        num_tracking_blocks: Optional[int] = 18,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        **kwargs
    ):
        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            time_embed_dim=time_embed_dim,
            text_embed_dim=text_embed_dim,
            num_layers=num_layers,
            dropout=dropout,
            attention_bias=attention_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            patch_size=patch_size,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            activation_fn=activation_fn,
            timestep_activation_fn=timestep_activation_fn,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_rotary_positional_embeddings=use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
            **kwargs
        )
        self.camera_embedder = MaskCamEmbed()
        
        inner_dim = num_attention_heads * attention_head_dim
        self.num_tracking_blocks = num_tracking_blocks

        # Ensure num_tracking_blocks is not greater than num_layers
        if num_tracking_blocks > num_layers:
            raise ValueError("num_tracking_blocks must be less than or equal to num_layers")

        # Create linear layers for combining hidden states and tracking maps
        self.combine_linears = nn.ModuleList(
            [nn.Linear(inner_dim, inner_dim, device="cpu") for _ in range(num_tracking_blocks)]
        )

        # Initialize weights of combine_linears to zero
        for linear in self.combine_linears:
            linear.weight.data.zero_()
            linear.bias.data.zero_()

        # Create transformer blocks for processing tracking maps
        self.transformer_blocks_copy = nn.ModuleList(
            [
                CogVideoXDASBlock(
                    dim=inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    time_embed_dim=self.config.time_embed_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                ).to_empty(device="cpu")
                for _ in range(num_tracking_blocks)
            ]
        )
        
        # self.cut3r_linear = nn.ModuleList(
        #     [
        #         nn.Linear(inner_dim, inner_dim // 2, device="cpu")
        #         for _ in range(num_tracking_blocks)
        #     ]
        # )
        # for linear in self.cut3r_linear:
        #     linear.weight.data.zero_()
        #     linear.bias.data.zero_()

        
        # For initial combination of hidden states and tracking maps
        self.initial_combine_linear = nn.Linear(inner_dim, inner_dim, device="cpu")
        self.initial_combine_linear.weight.data.zero_()
        self.initial_combine_linear.bias.data.zero_()


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        tracking_maps: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        cut3r_state: torch.Tensor = None,
    ):
        # plucker embedding
        tracking_maps = tracking_maps.squeeze(1)  # [b,1,c,f,h,w] -> [b,c,f,h,w]
        tracking_maps = self.camera_embedder(tracking_maps)
        tracking_maps = torch.cat([tracking_maps, hidden_states[:,:,16:]], dim=2)
        
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        # Process tracking maps
        prompt_embed = encoder_hidden_states.clone()
        tracking_maps_hidden_states = self.patch_embed(prompt_embed, tracking_maps)
        tracking_maps_hidden_states = self.embedding_dropout(tracking_maps_hidden_states)
        del prompt_embed

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]
        tracking_maps = tracking_maps_hidden_states[:, text_seq_length:]

        # Combine hidden states and tracking maps initially
        combined = hidden_states + tracking_maps
        tracking_maps = self.initial_combine_linear(combined)
        
        # Process transformer blocks
        for i in range(len(self.transformer_blocks)):
            if self.training and self.gradient_checkpointing:
                # Gradient checkpointing logic for hidden states
                def create_custom_forward(module):
                    # def custom_forward(*inputs):
                    #     return module(*inputs)
                    def custom_forward(*inputs, **kwargs):
                        return module(*inputs, **kwargs)
                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.transformer_blocks[i]),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = self.transformer_blocks[i](
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )
            
            if i < len(self.transformer_blocks_copy):
                if self.training and self.gradient_checkpointing:
                    # Gradient checkpointing logic for tracking maps
                    tracking_maps, _ = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.transformer_blocks_copy[i]),
                        tracking_maps,
                        encoder_hidden_states,
                        emb,
                        image_rotary_emb,
                        cut3r_state=cut3r_state,
                        **ckpt_kwargs,
                    )
                else:
                    tracking_maps, _ = self.transformer_blocks_copy[i](
                        hidden_states=tracking_maps,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                        cut3r_state=cut3r_state,
                    )
                
                # Combine hidden states and tracking maps
                tracking_maps_temp = self.combine_linears[i](tracking_maps)
                # if torch.isnan(tracking_maps_temp).any() or torch.isinf(tracking_maps_temp).any():
                hidden_states = hidden_states + tracking_maps_temp

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        # Note: we use `-1` instead of `channels`:
        #   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
        #   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], load_from_cogvideo=False, **kwargs):
        if load_from_cogvideo:
            print("Attempting to load as CogVideoXTransformer3DModel and convert...")

            base_model = CogVideoXTransformer3DModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
            
            config = dict(base_model.config)
            config["num_tracking_blocks"] = kwargs.pop("num_tracking_blocks", 18)
            
            model = cls(**config)
            model.load_state_dict(base_model.state_dict(), strict=False)

            model.initial_combine_linear.weight.data.zero_()
            model.initial_combine_linear.bias.data.zero_()
            
            for linear in model.combine_linears:
                linear.weight.data.zero_()
                linear.bias.data.zero_()
            
            # for i in range(model.num_tracking_blocks):
            #     model.transformer_blocks_copy[i].load_state_dict(model.transformer_blocks[i].state_dict())
                
            for i in range(model.num_tracking_blocks):
                src_state = model.transformer_blocks[i].state_dict()
                tgt_state = model.transformer_blocks_copy[i].state_dict()
                
                filtered_state = {}
                shape_mismatch = []
                
                for k, v in src_state.items():
                    if k in tgt_state:
                        if tgt_state[k].shape == v.shape:
                            filtered_state[k] = v
                        else:
                            shape_mismatch.append((k, v.shape, tgt_state[k].shape))
                
                # 로드
                missing, unexpected = model.transformer_blocks_copy[i].load_state_dict(filtered_state, strict=False)
                
                # 리포트
                print(f"\n[Block {i}]")
                if missing:
                    print("  Missing keys:", missing)
                if unexpected:
                    print("  Unexpected keys:", unexpected)
                if shape_mismatch:
                    print("  Shape mismatch:")
                    for k, src_shape, tgt_shape in shape_mismatch:
                        print(f"    {k}: src {src_shape} vs tgt {tgt_shape}")
                if not (missing or unexpected or shape_mismatch):
                    print("  ✅ All matched")
            
            for param in model.parameters():
                param.requires_grad = False
            
            for linear in model.combine_linears:
                for param in linear.parameters():
                    param.requires_grad = True
                
            for block in model.transformer_blocks_copy:
                for param in block.parameters():
                    param.requires_grad = True
                
            for param in model.initial_combine_linear.parameters():
                param.requires_grad = True
            
            for param in model.camera_embedder.parameters():
                param.requires_grad = True
                
            
        else:
            model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
            print("Loaded DiffusionAsShader checkpoint directly.")
            
            for param in model.parameters():
                param.requires_grad = False
            
            for linear in model.combine_linears:
                for param in linear.parameters():
                    param.requires_grad = True
                
            for block in model.transformer_blocks_copy:
                for param in block.parameters():
                    param.requires_grad = True
                
            for param in model.initial_combine_linear.parameters():
                param.requires_grad = True
            
            return model
        
        return model

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Optional[Callable] = None,
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        max_shard_size: Union[int, str] = "5GB",
        push_to_hub: bool = False,
        **kwargs,
    ):
        super().save_pretrained(
            save_directory,
            is_main_process=is_main_process,
            save_function=save_function,
            safe_serialization=safe_serialization,
            variant=variant,
            max_shard_size=max_shard_size,
            push_to_hub=push_to_hub,
            **kwargs,
        )
        
        if is_main_process:
            config_dict = dict(self.config)
            config_dict.pop("_name_or_path", None)
            config_dict.pop("_use_default_values", None)
            config_dict["_class_name"] = "CogVideoXTransformer3DModelTracking"
            config_dict["num_tracking_blocks"] = self.num_tracking_blocks
            
            os.makedirs(save_directory, exist_ok=True)
            with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
                import json
                json.dump(config_dict, f, indent=2)

