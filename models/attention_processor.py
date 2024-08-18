# modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from diffusers.image_processor import IPAdapterMaskProcessor
from typing import Optional, Any, List
import math
from einops import rearrange, repeat

try:
    import xformers
    import xformers.ops
    xformers_available = True
except Exception as e:
    xformers_available = False


class AttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class IPAttnProcessor(nn.Module):
    r"""
    Attention processor for Multiple IP-Adapters.

    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        num_tokens (`int`, `Tuple[int]` or `List[int]`, defaults to `(4,)`):
            The context length of the image features.
        scale (`float` or List[`float`], defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, num_tokens=(4,), scale=1.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = [num_tokens]
        self.num_tokens = num_tokens

        if not isinstance(scale, list):
            scale = [scale] * len(num_tokens)
        if len(scale) != len(num_tokens):
            raise ValueError("`scale` should be a list of integers with the same length as `num_tokens`.")
        self.scale = scale

        self.to_k_ip = nn.ModuleList(
            [nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in range(len(num_tokens))]
        )
        self.to_v_ip = nn.ModuleList(
            [nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in range(len(num_tokens))]
        )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        ip_adapter_masks: Optional[torch.FloatTensor] = None,
    ):
        residual = hidden_states

        # separate ip_hidden_states from encoder_hidden_states
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states
            else:
                deprecation_message = (
                    "You have passed a tensor as `encoder_hidden_states`. This is deprecated and will be removed in a future release."
                    " Please make sure to update your script to pass `encoder_hidden_states` as a tuple to suppress this warning."
                )
                # deprecate("encoder_hidden_states not a tuple", "1.0.0", deprecation_message, standard_warn=False)
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    [encoder_hidden_states[:, end_pos:, :]],
                )

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        if ip_adapter_masks is not None:
            if not isinstance(ip_adapter_masks, List):
                # for backward compatibility, we accept `ip_adapter_mask` as a tensor of shape [num_ip_adapter, 1, height, width]
                ip_adapter_masks = list(ip_adapter_masks.unsqueeze(1))
            if not (len(ip_adapter_masks) == len(self.scale) == len(ip_hidden_states)):
                raise ValueError(
                    f"Length of ip_adapter_masks array ({len(ip_adapter_masks)}) must match "
                    f"length of self.scale array ({len(self.scale)}) and number of ip_hidden_states "
                    f"({len(ip_hidden_states)})"
                )
            else:
                for index, (mask, scale, ip_state) in enumerate(zip(ip_adapter_masks, self.scale, ip_hidden_states)):
                    if not isinstance(mask, torch.Tensor) or mask.ndim != 4:
                        raise ValueError(
                            "Each element of the ip_adapter_masks array should be a tensor with shape "
                            "[1, num_images_for_ip_adapter, height, width]."
                            " Please use `IPAdapterMaskProcessor` to preprocess your mask"
                        )
                    if mask.shape[1] != ip_state.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of ip images ({ip_state.shape[1]}) at index {index}"
                        )
                    if isinstance(scale, list) and not len(scale) == mask.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of scales ({len(scale)}) at index {index}"
                        )
        else:
            ip_adapter_masks = [None] * len(self.scale)

        # for ip-adapter
        for current_ip_hidden_states, scale, to_k_ip, to_v_ip, mask in zip(
            ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip, ip_adapter_masks
        ):
            if mask is not None:
                if not isinstance(scale, list):
                    scale = [scale]

                current_num_images = mask.shape[1]
                for i in range(current_num_images):
                    ip_key = to_k_ip(current_ip_hidden_states[:, i, :, :])
                    ip_value = to_v_ip(current_ip_hidden_states[:, i, :, :])

                    ip_key = attn.head_to_batch_dim(ip_key)
                    ip_value = attn.head_to_batch_dim(ip_value)

                    ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
                    _current_ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
                    _current_ip_hidden_states = attn.batch_to_head_dim(_current_ip_hidden_states)

                    mask_downsample = IPAdapterMaskProcessor.downsample(
                        mask[:, i, :, :],
                        batch_size,
                        _current_ip_hidden_states.shape[1],
                        _current_ip_hidden_states.shape[2],
                    )

                    mask_downsample = mask_downsample.to(dtype=query.dtype, device=query.device)

                    hidden_states = hidden_states + scale[i] * (_current_ip_hidden_states * mask_downsample)
            else:
                ip_key = to_k_ip(current_ip_hidden_states)
                ip_value = to_v_ip(current_ip_hidden_states)

                ip_key = attn.head_to_batch_dim(ip_key)
                ip_value = attn.head_to_batch_dim(ip_value)

                ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
                current_ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
                current_ip_hidden_states = attn.batch_to_head_dim(current_ip_hidden_states)

                hidden_states = hidden_states + scale * current_ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class AttnProcessor2_0(torch.nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            # deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

class IPAttnProcessor2_0(torch.nn.Module):
    r"""
    Attention processor for IP-Adapter for PyTorch 2.0.

    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        num_tokens (`int`, `Tuple[int]` or `List[int]`, defaults to `(4,)`):
            The context length of the image features.
        scale (`float` or `List[float]`, defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, num_tokens=(4,), scale=1.0):
        super().__init__()

        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = [num_tokens]
        self.num_tokens = num_tokens

        # if not isinstance(scale, list):
        #     scale = [scale] * len(num_tokens)
        # if len(scale) != len(num_tokens):
        #     raise ValueError("`scale` should be a list of integers with the same length as `num_tokens`.")
        self.scale = scale

        # self.to_k_ip = nn.ModuleList(
        #     [nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in range(len(num_tokens))]
        # )
        # self.to_v_ip = nn.ModuleList(
        #     [nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in range(len(num_tokens))]
        # )
        self.to_k_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(cross_attention_dim, hidden_size, bias=False)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        ip_adapter_masks: Optional[torch.FloatTensor] = None,
    ):
        residual = hidden_states

        # separate ip_hidden_states from encoder_hidden_states
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states
            else:
                deprecation_message = (
                    "You have passed a tensor as `encoder_hidden_states`. This is deprecated and will be removed in a future release."
                    " Please make sure to update your script to pass `encoder_hidden_states` as a tuple to suppress this warning."
                )
                # deprecate("encoder_hidden_states not a tuple", "1.0.0", deprecation_message, standard_warn=False)
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    [encoder_hidden_states[:, end_pos:, :]],
                )

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # adapter
        key_ip = self.to_k_ip(ip_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_ip = self.to_v_ip(ip_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        ip_attn_hidden_states = F.scaled_dot_product_attention(
            query, key_ip, value_ip, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
        )
        ip_attn_hidden_states = ip_attn_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_attn_hidden_states = ip_attn_hidden_states.to(query.dtype)  # (B, num_queries, attn_dim)
        
        if ip_adapter_masks is not None:
            ip_adapter_masks = batch_downsample_masks(ip_adapter_masks, ip_attn_hidden_states.shape[1], ip_attn_hidden_states.shape[2])
            ip_attn_hidden_states = ip_attn_hidden_states * ip_adapter_masks
        
        hidden_states = hidden_states + self.scale * ip_attn_hidden_states
        
        # if ip_adapter_masks is not None:
        #     if not isinstance(ip_adapter_masks, List):
        #         # for backward compatibility, we accept `ip_adapter_mask` as a tensor of shape [num_ip_adapter, 1, height, width]
        #         ip_adapter_masks = list(ip_adapter_masks.unsqueeze(1))
        #     if not (len(ip_adapter_masks) == len(self.scale) == len(ip_hidden_states)):
        #         raise ValueError(
        #             f"Length of ip_adapter_masks array ({len(ip_adapter_masks)}) must match "
        #             f"length of self.scale array ({len(self.scale)}) and number of ip_hidden_states "
        #             f"({len(ip_hidden_states)})"
        #         )
        #     else:
        #         for index, (mask, scale, ip_state) in enumerate(zip(ip_adapter_masks, self.scale, ip_hidden_states)):
        #             if not isinstance(mask, torch.Tensor) or mask.ndim != 4:
        #                 raise ValueError(
        #                     "Each element of the ip_adapter_masks array should be a tensor with shape "
        #                     "[1, num_images_for_ip_adapter, height, width]."
        #                     " Please use `IPAdapterMaskProcessor` to preprocess your mask"
        #                 )
        #             if mask.shape[1] != ip_state.shape[1]:
        #                 raise ValueError(
        #                     f"Number of masks ({mask.shape[1]}) does not match "
        #                     f"number of ip images ({ip_state.shape[1]}) at index {index}"
        #                 )
        #             if isinstance(scale, list) and not len(scale) == mask.shape[1]:
        #                 raise ValueError(
        #                     f"Number of masks ({mask.shape[1]}) does not match "
        #                     f"number of scales ({len(scale)}) at index {index}"
        #                 )
        # else:
        #     ip_adapter_masks = [None] * len(self.scale)

        # # for ip-adapter
        # for current_ip_hidden_states, scale, to_k_ip, to_v_ip, mask in zip(
        #     ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip, ip_adapter_masks
        # ):
        #     if mask is not None:
        #         if not isinstance(scale, list):
        #             scale = [scale]

        #         current_num_images = mask.shape[1]
        #         for i in range(current_num_images):
        #             ip_key = to_k_ip(current_ip_hidden_states[:, i, :, :])
        #             ip_value = to_v_ip(current_ip_hidden_states[:, i, :, :])

        #             ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        #             ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        #             # the output of sdp = (batch, num_heads, seq_len, head_dim)
        #             # TODO: add support for attn.scale when we move to Torch 2.1
        #             _current_ip_hidden_states = F.scaled_dot_product_attention(
        #                 query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
        #             )

        #             _current_ip_hidden_states = _current_ip_hidden_states.transpose(1, 2).reshape(
        #                 batch_size, -1, attn.heads * head_dim
        #             )
        #             _current_ip_hidden_states = _current_ip_hidden_states.to(query.dtype)

        #             mask_downsample = IPAdapterMaskProcessor.downsample(
        #                 mask[:, i, :, :],
        #                 batch_size,
        #                 _current_ip_hidden_states.shape[1],
        #                 _current_ip_hidden_states.shape[2],
        #             )

        #             mask_downsample = mask_downsample.to(dtype=query.dtype, device=query.device)
        #             hidden_states = hidden_states + scale[i] * (_current_ip_hidden_states * mask_downsample)
        #     else:
        #         ip_key = to_k_ip(current_ip_hidden_states)
        #         ip_value = to_v_ip(current_ip_hidden_states)

        #         ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        #         ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        #         # the output of sdp = (batch, num_heads, seq_len, head_dim)
        #         # TODO: add support for attn.scale when we move to Torch 2.1
        #         current_ip_hidden_states = F.scaled_dot_product_attention(
        #             query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
        #         )

        #         current_ip_hidden_states = current_ip_hidden_states.transpose(1, 2).reshape(
        #             batch_size, -1, attn.heads * head_dim
        #         )
        #         current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)

        #         hidden_states = hidden_states + scale * current_ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

# NOTE(ZSH): modify from IPAdapterMaskProcessor.downsample
def batch_downsample_masks(mask, num_queries, value_embed_dim):
    # mask: (B, 1, H, W)
    batch_size = mask.shape[0]
    o_h, o_w = mask.shape[2], mask.shape[3]
    ratio = o_w / o_h
    mask_h = int(math.sqrt(num_queries / ratio))
    mask_h = int(mask_h) + int((num_queries % int(mask_h)) != 0)
    mask_w = num_queries // mask_h
    
    mask_downsample = F.interpolate(mask, size=(mask_h, mask_w), mode="nearest")  # (B, 1, mask_h, mask_w)
    mask_downsample = mask_downsample.reshape(batch_size, -1)
    
    mask_downsample = mask_downsample.unsqueeze(-1).repeat(1, 1, value_embed_dim)
    return mask_downsample


class LatentAttnProcessor(nn.Module):
    def __init__(self, query_dim, cross_attention_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_q = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_q = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )
        
    def forward(self, hidden_states, encoder_hidden_states, attention_mask, latent_attn_mask, use_attn_mask=True):
        h = self.heads
        b = hidden_states.shape[0]  # B * hn
        
        q = self.to_q(hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)
        
        inner_dim = q.shape[-1]
        head_dim = inner_dim // h
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        
        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        if use_attn_mask:
            HW = sim.shape[1]
            H = W = int(math.sqrt(HW))
            latent_attn_mask = F.interpolate(latent_attn_mask, size=(H, W), mode="bilinear")            
            sim = sim.view(b, h, HW, head_dim) 
            latent_attn_mask = latent_attn_mask.view(b, 1, HW, 1)
            latent_attn_mask[latent_attn_mask == 1] = 5.0
            latent_attn_mask[latent_attn_mask == 0] = 0.1
            sim = sim * latent_attn_mask
            sim = sim.view(b * h, HW, head_dim)
        
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        
        return self.to_out(out)


class LayoutAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dropout=0., use_lora=False):
        super().__init__()
        dim_head = query_dim // heads
        inner_dim = query_dim
        context_dim = query_dim if context_dim is None else context_dim

        self.use_lora = use_lora
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, return_attn=False, need_softmax=True, guidance_mask=None):
        h = self.heads
        b = x.shape[0]

        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

        _, phase_num, H, W = guidance_mask.shape
        HW = H * W
        guidance_mask_o = guidance_mask.view(b * phase_num, HW, 1)
        guidance_mask_t = guidance_mask.view(b * phase_num, 1, HW)
        guidance_mask_sim = torch.bmm(guidance_mask_o, guidance_mask_t)  # (B * phase_num, HW, HW)
        guidance_mask_sim = guidance_mask_sim.view(b, phase_num, HW, HW).sum(dim=1)
        guidance_mask_sim[guidance_mask_sim > 1] = 1  # (B, HW, HW)
        guidance_mask_sim = guidance_mask_sim.view(b, 1, HW, HW)
        guidance_mask_sim = guidance_mask_sim.repeat(1, self.heads, 1, 1)
        guidance_mask_sim = guidance_mask_sim.view(b * self.heads, HW, HW)  # (B * head, HW, HW)

        sim[:, :, :HW][guidance_mask_sim == 0] = -torch.finfo(sim.dtype).max

        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of

        if need_softmax:
            attn = sim.softmax(dim=-1)
        else:
            attn = sim
            
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        if return_attn:
            attn = attn.view(b, h, attn.shape[-2], attn.shape[-1])
            return self.to_out(out), attn
        else:
            return self.to_out(out)
                                       