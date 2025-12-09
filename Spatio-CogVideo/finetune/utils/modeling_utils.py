
import copy
import inspect
import itertools
import json
import os
import re
from collections import OrderedDict
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import safetensors
import torch
import torch.utils.checkpoint
from huggingface_hub import DDUFEntry, create_repo, split_torch_state_dict_into_shards
from huggingface_hub.utils import validate_hf_hub_args
from torch import Tensor, nn
from typing_extensions import Self

from diffusers import __version__
# from diffusers.hooks import apply_layerwise_casting
from diffusers.quantizers import DiffusersAutoQuantizer, DiffusersQuantizer
from diffusers.quantizers.quantization_config import QuantizationMethod
from diffusers.utils import (
    CONFIG_NAME,
    FLAX_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    _add_variant,
    _get_checkpoint_shard_files,
    _get_model_file,
    deprecate,
    is_accelerate_available,
    is_bitsandbytes_available,
    is_bitsandbytes_version,
    is_peft_available,
    is_torch_version,
    logging,
)
from diffusers.utils.hub_utils import (
    PushToHubMixin,
    load_or_create_model_card,
    populate_model_card,
)
from diffusers.models.model_loading_utils import (
    _determine_device_map,
    _fetch_index_file,
    _fetch_index_file_legacy,
    _load_state_dict_into_model,
    _merge_sharded_checkpoints,
    load_model_dict_into_meta,
    load_state_dict,
)
from diffusers.models import ModelMixin
from typing_extensions import override
from diffusers.utils.torch_utils import maybe_allow_in_graph


logger = logging.get_logger(__name__)

_REGEX_SHARD = re.compile(r"(.*?)-\d{5}-of-\d{5}")


if is_torch_version(">=", "1.9.0"):
    _LOW_CPU_MEM_USAGE_DEFAULT = True
else:
    _LOW_CPU_MEM_USAGE_DEFAULT = False


if is_accelerate_available():
    import accelerate


@maybe_allow_in_graph
class ModelMixin_load_v2v(ModelMixin):
    
    @override
    @classmethod
    @validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], **kwargs) -> Self:
            r"""
            Instantiate a pretrained PyTorch model from a pretrained model configuration.

            The model is set in evaluation mode - `model.eval()` - by default, and dropout modules are deactivated. To
            train the model, set it back in training mode with `model.train()`.

            Parameters:
                pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                    Can be either:

                        - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                        the Hub.
                        - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                        with [`~ModelMixin.save_pretrained`].

                cache_dir (`Union[str, os.PathLike]`, *optional*):
                    Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                    is not used.
                torch_dtype (`str` or `torch.dtype`, *optional*):
                    Override the default `torch.dtype` and load the model with another dtype. If `"auto"` is passed, the
                    dtype is automatically derived from the model's weights.
                force_download (`bool`, *optional*, defaults to `False`):
                    Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                    cached versions if they exist.
                proxies (`Dict[str, str]`, *optional*):
                    A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                    'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
                output_loading_info (`bool`, *optional*, defaults to `False`):
                    Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
                local_files_only(`bool`, *optional*, defaults to `False`):
                    Whether to only load local model weights and configuration files or not. If set to `True`, the model
                    won't be downloaded from the Hub.
                token (`str` or *bool*, *optional*):
                    The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                    `diffusers-cli login` (stored in `~/.huggingface`) is used.
                revision (`str`, *optional*, defaults to `"main"`):
                    The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                    allowed by Git.
                from_flax (`bool`, *optional*, defaults to `False`):
                    Load the model weights from a Flax checkpoint save file.
                subfolder (`str`, *optional*, defaults to `""`):
                    The subfolder location of a model file within a larger model repository on the Hub or locally.
                mirror (`str`, *optional*):
                    Mirror source to resolve accessibility issues if you're downloading a model in China. We do not
                    guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                    information.
                device_map (`str` or `Dict[str, Union[int, str, torch.device]]`, *optional*):
                    A map that specifies where each submodule should go. It doesn't need to be defined for each
                    parameter/buffer name; once a given module name is inside, every submodule of it will be sent to the
                    same device. Defaults to `None`, meaning that the model will be loaded on CPU.

                    Set `device_map="auto"` to have 🤗 Accelerate automatically compute the most optimized `device_map`. For
                    more information about each option see [designing a device
                    map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
                max_memory (`Dict`, *optional*):
                    A dictionary device identifier for the maximum memory. Will default to the maximum memory available for
                    each GPU and the available CPU RAM if unset.
                offload_folder (`str` or `os.PathLike`, *optional*):
                    The path to offload weights if `device_map` contains the value `"disk"`.
                offload_state_dict (`bool`, *optional*):
                    If `True`, temporarily offloads the CPU state dict to the hard drive to avoid running out of CPU RAM if
                    the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to `True`
                    when there is some disk offload.
                low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                    Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                    tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                    Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                    argument to `True` will raise an error.
                variant (`str`, *optional*):
                    Load weights from a specified `variant` filename such as `"fp16"` or `"ema"`. This is ignored when
                    loading `from_flax`.
                use_safetensors (`bool`, *optional*, defaults to `None`):
                    If set to `None`, the `safetensors` weights are downloaded if they're available **and** if the
                    `safetensors` library is installed. If set to `True`, the model is forcibly loaded from `safetensors`
                    weights. If set to `False`, `safetensors` weights are not loaded.
                disable_mmap ('bool', *optional*, defaults to 'False'):
                    Whether to disable mmap when loading a Safetensors model. This option can perform better when the model
                    is on a network mount or hard drive, which may not handle the seeky-ness of mmap very well.

            <Tip>

            To use private or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models), log-in with
            `huggingface-cli login`. You can also activate the special
            ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use this method in a
            firewalled environment.

            </Tip>

            Example:

            ```py
            from diffusers import UNet2DConditionModel

            unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
            ```

            If you get the error message below, you need to finetune the weights for your downstream task:

            ```bash
            Some weights of UNet2DConditionModel were not initialized from the model checkpoint at runwayml/stable-diffusion-v1-5 and are newly initialized because the shapes did not match:
            - conv_in.weight: found shape torch.Size([320, 4, 3, 3]) in the checkpoint and torch.Size([320, 9, 3, 3]) in the model instantiated
            You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
            ```
            """
            cache_dir = kwargs.pop("cache_dir", None)
            ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
            force_download = kwargs.pop("force_download", False)
            from_flax = kwargs.pop("from_flax", False)
            proxies = kwargs.pop("proxies", None)
            output_loading_info = kwargs.pop("output_loading_info", False)
            local_files_only = kwargs.pop("local_files_only", None)
            token = kwargs.pop("token", None)
            revision = kwargs.pop("revision", None)
            torch_dtype = kwargs.pop("torch_dtype", None)
            subfolder = kwargs.pop("subfolder", None)
            device_map = kwargs.pop("device_map", None)
            max_memory = kwargs.pop("max_memory", None)
            offload_folder = kwargs.pop("offload_folder", None)
            offload_state_dict = kwargs.pop("offload_state_dict", False)
            low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
            variant = kwargs.pop("variant", None)
            use_safetensors = kwargs.pop("use_safetensors", None)
            quantization_config = kwargs.pop("quantization_config", None)
            dduf_entries: Optional[Dict[str, DDUFEntry]] = kwargs.pop("dduf_entries", None)
            disable_mmap = kwargs.pop("disable_mmap", False)

            allow_pickle = False
            if use_safetensors is None:
                use_safetensors = True
                allow_pickle = True

            if low_cpu_mem_usage and not is_accelerate_available():
                low_cpu_mem_usage = False
                logger.warning(
                    "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                    " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                    " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                    " install accelerate\n```\n."
                )

            if device_map is not None and not is_accelerate_available():
                raise NotImplementedError(
                    "Loading and dispatching requires `accelerate`. Please make sure to install accelerate or set"
                    " `device_map=None`. You can install accelerate with `pip install accelerate`."
                )

            # Check if we can handle device_map and dispatching the weights
            if device_map is not None and not is_torch_version(">=", "1.9.0"):
                raise NotImplementedError(
                    "Loading and dispatching requires torch >= 1.9.0. Please either update your PyTorch version or set"
                    " `device_map=None`."
                )

            if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
                raise NotImplementedError(
                    "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                    " `low_cpu_mem_usage=False`."
                )

            if low_cpu_mem_usage is False and device_map is not None:
                raise ValueError(
                    f"You cannot set `low_cpu_mem_usage` to `False` while using device_map={device_map} for loading and"
                    " dispatching. Please make sure to set `low_cpu_mem_usage=True`."
                )

            # change device_map into a map if we passed an int, a str or a torch.device
            if isinstance(device_map, torch.device):
                device_map = {"": device_map}
            elif isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
                try:
                    device_map = {"": torch.device(device_map)}
                except RuntimeError:
                    raise ValueError(
                        "When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or "
                        f"'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."
                    )
            elif isinstance(device_map, int):
                if device_map < 0:
                    raise ValueError(
                        "You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "
                    )
                else:
                    device_map = {"": device_map}

            if device_map is not None:
                if low_cpu_mem_usage is None:
                    low_cpu_mem_usage = True
                elif not low_cpu_mem_usage:
                    raise ValueError("Passing along a `device_map` requires `low_cpu_mem_usage=True`")

            if low_cpu_mem_usage:
                if device_map is not None and not is_torch_version(">=", "1.10"):
                    # The max memory utils require PyTorch >= 1.10 to have torch.cuda.mem_get_info.
                    raise ValueError("`low_cpu_mem_usage` and `device_map` require PyTorch >= 1.10.")

            # Load config if we don't provide a configuration
            config_path = pretrained_model_name_or_path

            user_agent = {
                "diffusers": __version__,
                "file_type": "model",
                "framework": "pytorch",
            }

            # load config
            config, unused_kwargs, commit_hash = cls.load_config(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                return_commit_hash=True,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
                dduf_entries=dduf_entries,
                **kwargs,
            )
            # no in-place modification of the original config.
            config = copy.deepcopy(config)

            # determine initial quantization config.
            #######################################
            pre_quantized = "quantization_config" in config and config["quantization_config"] is not None
            if pre_quantized or quantization_config is not None:
                if pre_quantized:
                    config["quantization_config"] = DiffusersAutoQuantizer.merge_quantization_configs(
                        config["quantization_config"], quantization_config
                    )
                else:
                    config["quantization_config"] = quantization_config
                hf_quantizer = DiffusersAutoQuantizer.from_config(
                    config["quantization_config"], pre_quantized=pre_quantized
                )
            else:
                hf_quantizer = None

            if hf_quantizer is not None:
                if device_map is not None:
                    raise NotImplementedError(
                        "Currently, providing `device_map` is not supported for quantized models. Providing `device_map` as an input will be added in the future."
                    )

                hf_quantizer.validate_environment(torch_dtype=torch_dtype, from_flax=from_flax, device_map=device_map)
                torch_dtype = hf_quantizer.update_torch_dtype(torch_dtype)

                # In order to ensure popular quantization methods are supported. Can be disable with `disable_telemetry`
                user_agent["quant"] = hf_quantizer.quantization_config.quant_method.value

                # Force-set to `True` for more mem efficiency
                if low_cpu_mem_usage is None:
                    low_cpu_mem_usage = True
                    logger.info("Set `low_cpu_mem_usage` to True as `hf_quantizer` is not None.")
                elif not low_cpu_mem_usage:
                    raise ValueError("`low_cpu_mem_usage` cannot be False or None when using quantization.")

            # Check if `_keep_in_fp32_modules` is not None
            use_keep_in_fp32_modules = (cls._keep_in_fp32_modules is not None) and (
                (torch_dtype == torch.float16) or hasattr(hf_quantizer, "use_keep_in_fp32_modules")
            )
            if use_keep_in_fp32_modules:
                keep_in_fp32_modules = cls._keep_in_fp32_modules
                if not isinstance(keep_in_fp32_modules, list):
                    keep_in_fp32_modules = [keep_in_fp32_modules]

                if low_cpu_mem_usage is None:
                    low_cpu_mem_usage = True
                    logger.info("Set `low_cpu_mem_usage` to True as `_keep_in_fp32_modules` is not None.")
                elif not low_cpu_mem_usage:
                    raise ValueError("`low_cpu_mem_usage` cannot be False when `keep_in_fp32_modules` is True.")
            else:
                keep_in_fp32_modules = []
            #######################################

            # Determine if we're loading from a directory of sharded checkpoints.
            is_sharded = False
            index_file = None
            is_local = os.path.isdir(pretrained_model_name_or_path)
            index_file_kwargs = {
                "is_local": is_local,
                "pretrained_model_name_or_path": pretrained_model_name_or_path,
                "subfolder": subfolder or "",
                "use_safetensors": use_safetensors,
                "cache_dir": cache_dir,
                "variant": variant,
                "force_download": force_download,
                "proxies": proxies,
                "local_files_only": local_files_only,
                "token": token,
                "revision": revision,
                "user_agent": user_agent,
                "commit_hash": commit_hash,
                "dduf_entries": dduf_entries,
            }
            index_file = _fetch_index_file(**index_file_kwargs)
            # In case the index file was not found we still have to consider the legacy format.
            # this becomes applicable when the variant is not None.
            if variant is not None and (index_file is None or not os.path.exists(index_file)):
                index_file = _fetch_index_file_legacy(**index_file_kwargs)
            if index_file is not None and (dduf_entries or index_file.is_file()):
                is_sharded = True

            if is_sharded and from_flax:
                raise ValueError("Loading of sharded checkpoints is not supported when `from_flax=True`.")

            # load model
            model_file = None
            if from_flax:
                model_file = _get_model_file(
                    pretrained_model_name_or_path,
                    weights_name=FLAX_WEIGHTS_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                    commit_hash=commit_hash,
                )
                model = cls.from_config(config, **unused_kwargs)

                # Convert the weights
                from .modeling_pytorch_flax_utils import load_flax_checkpoint_in_pytorch_model

                model = load_flax_checkpoint_in_pytorch_model(model, model_file)
            else:
                # in the case it is sharded, we have already the index
                if is_sharded:
                    sharded_ckpt_cached_folder, sharded_metadata = _get_checkpoint_shard_files(
                        pretrained_model_name_or_path,
                        index_file,
                        cache_dir=cache_dir,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        token=token,
                        user_agent=user_agent,
                        revision=revision,
                        subfolder=subfolder or "",
                        dduf_entries=dduf_entries,
                    )
                    # TODO: https://github.com/huggingface/diffusers/issues/10013
                    if hf_quantizer is not None or dduf_entries:
                        model_file = _merge_sharded_checkpoints(
                            sharded_ckpt_cached_folder, sharded_metadata, dduf_entries=dduf_entries
                        )
                        logger.info("Merged sharded checkpoints as `hf_quantizer` is not None.")
                        is_sharded = False

                elif use_safetensors and not is_sharded:
                    try:
                        model_file = _get_model_file(
                            pretrained_model_name_or_path,
                            weights_name=_add_variant(SAFETENSORS_WEIGHTS_NAME, variant),
                            cache_dir=cache_dir,
                            force_download=force_download,
                            proxies=proxies,
                            local_files_only=local_files_only,
                            token=token,
                            revision=revision,
                            subfolder=subfolder,
                            user_agent=user_agent,
                            commit_hash=commit_hash,
                            dduf_entries=dduf_entries,
                        )

                    except IOError as e:
                        logger.error(f"An error occurred while trying to fetch {pretrained_model_name_or_path}: {e}")
                        if not allow_pickle:
                            raise
                        logger.warning(
                            "Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead."
                        )

                if model_file is None and not is_sharded:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path,
                        weights_name=_add_variant(WEIGHTS_NAME, variant),
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        token=token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                        commit_hash=commit_hash,
                        dduf_entries=dduf_entries,
                    )

                if low_cpu_mem_usage:
                    # Instantiate model with empty weights
                    with accelerate.init_empty_weights():
                        model = cls.from_config(config, **unused_kwargs)

                    if hf_quantizer is not None:
                        hf_quantizer.preprocess_model(
                            model=model, device_map=device_map, keep_in_fp32_modules=keep_in_fp32_modules
                        )

                    # if device_map is None, load the state dict and move the params from meta device to the cpu
                    if device_map is None and not is_sharded:
                        # `torch.cuda.current_device()` is fine here when `hf_quantizer` is not None.
                        # It would error out during the `validate_environment()` call above in the absence of cuda.
                        if hf_quantizer is None:
                            param_device = "cpu"
                        # TODO (sayakpaul,  SunMarc): remove this after model loading refactor
                        else:
                            param_device = torch.device(torch.cuda.current_device())
                        state_dict = load_state_dict(
                            model_file, variant=variant, dduf_entries=dduf_entries, disable_mmap=disable_mmap
                        )
                        model._convert_deprecated_attention_blocks(state_dict)

                        # move the params from meta device to cpu
                        missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
                        if hf_quantizer is not None:
                            missing_keys = hf_quantizer.update_missing_keys(model, missing_keys, prefix="")
                        if len(missing_keys) > 0:
                            raise ValueError(
                                f"Cannot load {cls} from {pretrained_model_name_or_path} because the following keys are"
                                f" missing: \n {', '.join(missing_keys)}. \n Please make sure to pass"
                                " `low_cpu_mem_usage=False` and `device_map=None` if you want to randomly initialize"
                                " those weights or else make sure your checkpoint file is correct."
                            )

                        named_buffers = model.named_buffers()

                        unexpected_keys = load_model_dict_into_meta(
                            model,
                            state_dict,
                            device=param_device,
                            dtype=torch_dtype,
                            model_name_or_path=pretrained_model_name_or_path,
                            hf_quantizer=hf_quantizer,
                            keep_in_fp32_modules=keep_in_fp32_modules,
                            named_buffers=named_buffers,
                        )

                        if cls._keys_to_ignore_on_load_unexpected is not None:
                            for pat in cls._keys_to_ignore_on_load_unexpected:
                                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

                        if len(unexpected_keys) > 0:
                            logger.warning(
                                f"Some weights of the model checkpoint were not used when initializing {cls.__name__}: \n {[', '.join(unexpected_keys)]}"
                            )

                    else:  # else let accelerate handle loading and dispatching.
                        # Load weights and dispatch according to the device_map
                        # by default the device_map is None and the weights are loaded on the CPU
                        device_map = _determine_device_map(
                            model, device_map, max_memory, torch_dtype, keep_in_fp32_modules, hf_quantizer
                        )
                        if device_map is None and is_sharded:
                            # we load the parameters on the cpu
                            device_map = {"": "cpu"}
                        try:
                            accelerate.load_checkpoint_and_dispatch(
                                model,
                                model_file if not is_sharded else index_file,
                                device_map,
                                max_memory=max_memory,
                                offload_folder=offload_folder,
                                offload_state_dict=offload_state_dict,
                                dtype=torch_dtype,
                                strict=False,
                            )
                            # 수정함 strict =True -> False
                        except AttributeError as e:
                            # When using accelerate loading, we do not have the ability to load the state
                            # dict and rename the weight names manually. Additionally, accelerate skips
                            # torch loading conventions and directly writes into `module.{_buffers, _parameters}`
                            # (which look like they should be private variables?), so we can't use the standard hooks
                            # to rename parameters on load. We need to mimic the original weight names so the correct
                            # attributes are available. After we have loaded the weights, we convert the deprecated
                            # names to the new non-deprecated names. Then we _greatly encourage_ the user to convert
                            # the weights so we don't have to do this again.

                            if "'Attention' object has no attribute" in str(e):
                                logger.warning(
                                    f"Taking `{str(e)}` while using `accelerate.load_checkpoint_and_dispatch` to mean {pretrained_model_name_or_path}"
                                    " was saved with deprecated attention block weight names. We will load it with the deprecated attention block"
                                    " names and convert them on the fly to the new attention block format. Please re-save the model after this conversion,"
                                    " so we don't have to do the on the fly renaming in the future. If the model is from a hub checkpoint,"
                                    " please also re-upload it or open a PR on the original repository."
                                )
                                model._temp_convert_self_to_deprecated_attention_blocks()
                                accelerate.load_checkpoint_and_dispatch(
                                    model,
                                    model_file if not is_sharded else index_file,
                                    device_map,
                                    max_memory=max_memory,
                                    offload_folder=offload_folder,
                                    offload_state_dict=offload_state_dict,
                                    dtype=torch_dtype,
                                    strict=True,
                                )
                                model._undo_temp_convert_self_to_deprecated_attention_blocks()
                            else:
                                raise e

                    loading_info = {
                        "missing_keys": [],
                        "unexpected_keys": [],
                        "mismatched_keys": [],
                        "error_msgs": [],
                    }
                else:
                    model = cls.from_config(config, **unused_kwargs)

                    state_dict = load_state_dict(
                        model_file, variant=variant, dduf_entries=dduf_entries, disable_mmap=disable_mmap
                    )
                    model._convert_deprecated_attention_blocks(state_dict)
                    model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
                        model,
                        state_dict,
                        model_file,
                        pretrained_model_name_or_path,
                        ignore_mismatched_sizes=ignore_mismatched_sizes,
                    )

                    loading_info = {
                        "missing_keys": missing_keys,
                        "unexpected_keys": unexpected_keys,
                        "mismatched_keys": mismatched_keys,
                        "error_msgs": error_msgs,
                    }

            if hf_quantizer is not None:
                hf_quantizer.postprocess_model(model)
                model.hf_quantizer = hf_quantizer

            if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
                raise ValueError(
                    f"{torch_dtype} needs to be of type `torch.dtype`, e.g. `torch.float16`, but is {type(torch_dtype)}."
                )
            # When using `use_keep_in_fp32_modules` if we do a global `to()` here, then we will
            # completely lose the effectivity of `use_keep_in_fp32_modules`.
            elif torch_dtype is not None and hf_quantizer is None and not use_keep_in_fp32_modules:
                model = model.to(torch_dtype)

            if hf_quantizer is not None:
                # We also make sure to purge `_pre_quantization_dtype` when we serialize
                # the model config because `_pre_quantization_dtype` is `torch.dtype`, not JSON serializable.
                model.register_to_config(_name_or_path=pretrained_model_name_or_path, _pre_quantization_dtype=torch_dtype)
            else:
                model.register_to_config(_name_or_path=pretrained_model_name_or_path)

            # Set model in evaluation mode to deactivate DropOut modules by default
            model.eval()
            if output_loading_info:
                return model, loading_info

            return model
        
    @override
    @wraps(torch.nn.Module.to)
    def to(self, *args, **kwargs):
        dtype_present_in_args = "dtype" in kwargs

        if not dtype_present_in_args:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    dtype_present_in_args = True
                    break

        if getattr(self, "is_quantized", False):
            if dtype_present_in_args:
                raise ValueError(
                    "Casting a quantized model to a new `dtype` is unsupported. To set the dtype of unquantized layers, please "
                    "use the `torch_dtype` argument when loading the model using `from_pretrained` or `from_single_file`"
                )

        if getattr(self, "quantization_method", None) == QuantizationMethod.BITS_AND_BYTES:
            if getattr(self, "is_loaded_in_8bit", False):
                raise ValueError(
                    "`.to` is not supported for `8-bit` bitsandbytes models. Please use the model as it is, since the"
                    " model has already been set to the correct devices and casted to the correct `dtype`."
                )
            elif is_bitsandbytes_version("<", "0.43.2"):
                raise ValueError(
                    "Calling `to()` is not supported for `4-bit` quantized models with the installed version of bitsandbytes. "
                    f"The current device is `{self.device}`. If you intended to move the model, please install bitsandbytes >= 0.43.2."
                )
        return super().to(*args, **kwargs)
        # try:
        #     return super().to(*args, **kwargs)
        # except:
        #     return super().to_empty(device='cpu')