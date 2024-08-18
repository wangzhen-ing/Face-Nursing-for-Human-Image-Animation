# Adapted from https://github.com/magic-research/magic-animate/blob/main/magicanimate/models/mutual_self_attention.py
from typing import Any, Dict, Optional

import torch
from einops import rearrange
import numpy as np

from models.attention import TemporalBasicTransformerBlock

from .attention import BasicTransformerBlock
import torch.nn.functional as F
import math
# NOTE(wz) face latent injection both in spatial attention and cross attention with alignment(在attention操作后对齐)

def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result


class ReferenceAttentionControl:
    def __init__(
        self,
        unet,
        mode="write",
        do_classifier_free_guidance=False,
        attention_auto_machine_weight=float("inf"),
        gn_auto_machine_weight=1.0,
        style_fidelity=1.0,
        reference_attn=True,
        reference_adain=False,
        fusion_blocks="midup",
        batch_size=1,
    ) -> None:
        # 10. Modify self attention and group norm
        self.unet = unet
        assert mode in ["read", "write"]
        assert fusion_blocks in ["midup", "full"]
        self.reference_attn = reference_attn
        self.reference_adain = reference_adain
        self.fusion_blocks = fusion_blocks
        self.register_reference_hooks(
            mode,
            do_classifier_free_guidance,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            fusion_blocks,
            batch_size=batch_size,
        )

    def register_reference_hooks(
        self,
        mode,
        do_classifier_free_guidance,
        attention_auto_machine_weight,
        gn_auto_machine_weight,
        style_fidelity,
        reference_attn,
        reference_adain,
        dtype=torch.float16,
        batch_size=1,
        num_images_per_prompt=1,
        device=torch.device("cpu"),
        fusion_blocks="midup",
    ):
        MODE = mode
        do_classifier_free_guidance = do_classifier_free_guidance
        attention_auto_machine_weight = attention_auto_machine_weight
        gn_auto_machine_weight = gn_auto_machine_weight
        style_fidelity = style_fidelity
        reference_attn = reference_attn
        reference_adain = reference_adain
        fusion_blocks = fusion_blocks
        num_images_per_prompt = num_images_per_prompt
        dtype = dtype
        if do_classifier_free_guidance:
            uc_mask = (
                torch.Tensor(
                    [1] * batch_size * num_images_per_prompt * 16
                    + [0] * batch_size * num_images_per_prompt * 16
                )
                .to(device)
                .bool()
            )
        else:
            uc_mask = (
                torch.Tensor([0] * batch_size * num_images_per_prompt * 2)
                .to(device)
                .bool()
            )

        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            video_length=None,
            ref_idx=None,
            write_latent=True,
            do_latent_attention=False,
            latent_attn_masks=None,
            do_infer=False,
        ):
            if self.use_ada_layer_norm:  # False
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                (
                    norm_hidden_states,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                ) = self.norm1(
                    hidden_states,
                    timestep,
                    class_labels,
                    hidden_dtype=hidden_states.dtype,
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            print("hidden_states的大小为:")
            print(hidden_states.shape)
            print("norm_hidden_states的大小为:")
            print(norm_hidden_states.shape)

            # 1. Self-Attention
            # self.only_cross_attention = False
            cross_attention_kwargs = (
                cross_attention_kwargs if cross_attention_kwargs is not None else {}
            )
            if self.only_cross_attention:
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=(
                        encoder_hidden_states if self.only_cross_attention else None
                    ),
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:
                if MODE == "write":
                    self.bank.append(norm_hidden_states.clone())
                    print("bank的大小为:")
                    print(len(self.bank), len(self.bank[0]))
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=(
                            encoder_hidden_states if self.only_cross_attention else None
                        ),
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                    print("经过attention后的大小为:")
                    print(attn_output.shape)
                if MODE == "read":
                    bank_fea = [
                        rearrange(
                            d.unsqueeze(1).repeat(1, video_length, 1, 1),
                            "b t l c -> (b t) l c",
                        )
                        for d in self.bank
                    ]
                    print("bank_fea的大小为:")
                    print(len(bank_fea), len(bank_fea[0]))
                    if not ref_idx:
                        modify_norm_hidden_states = torch.cat(
                            [norm_hidden_states] + bank_fea, dim=1
                        )
                    else:
                        modify_norm_hidden_states = torch.cat(
                            [norm_hidden_states] + [bank_fea[i] for i in ref_idx], dim=1
                        )
                    print("modify_norm_hidden_states的大小为:")
                    print(modify_norm_hidden_states.shape)
                    
                    # if write_latent:
                    #     self.latent_bank.append(norm_hidden_states)
                        
                    # if do_latent_attention:
                    #     modify_norm_hidden_states = torch.cat([modify_norm_hidden_states] + self.latent_bank, dim=1)
                    #     self.latent_bank.clear()
                    
                    hidden_states_uc = (
                        self.attn1(
                            norm_hidden_states,
                            encoder_hidden_states=modify_norm_hidden_states,
                            attention_mask=attention_mask,
                        )
                        + hidden_states
                    )
                    print("hidden_states_uc的大小为:")
                    print(hidden_states_uc.shape)
                    

                    B, N, D = norm_hidden_states.shape
                    # print(norm_hidden_states.shape)
                    if latent_attn_masks is not None:
                        h = w = int(N**0.5)
                        latent_attn_masks = F.interpolate(latent_attn_masks, size=(h, w), mode="nearest")
                        batch, channel, height, weight = latent_attn_masks.shape
                        latent_attn_masks = latent_attn_masks.permute(0, 2, 3, 1)
                        # print(latent_attn_masks.shape, '====')
                        latent_attn_masks = latent_attn_masks.reshape(batch, height * weight , channel)
                        # latent_attn_masks = latent_attn_masks.permute(0, 2, 3, 1).reshape(B, N, 1)
                        # if main_process:
                        #     print(latent_attn_masks.shape)
                    
                    if write_latent:
                        self.latent_bank.append(norm_hidden_states)
                        print("latent_bank的大小为:")
                        print(len(self.latent_bank), len(self.latent_bank[0]))        
                                    
                    if do_latent_attention:
                        latent_hidden_states_uc = (
                            self.attn_latent[0](
                                norm_hidden_states,
                                encoder_hidden_states=torch.cat(
                                    [norm_hidden_states, self.latent_bank[0]], dim=1 
                                ),
                                attention_mask=attention_mask,
                            ) + hidden_states
                        )
                        # self.latent_bank.clear()
                        latent_hidden_states_uc = latent_hidden_states_uc * latent_attn_masks.to(latent_hidden_states_uc.dtype)
                        
                        if do_classifier_free_guidance:
                            latent_hidden_states_c = latent_hidden_states_uc.clone()
                            _uc_mask = uc_mask.clone()
                            if hidden_states.shape[0] != _uc_mask.shape[0]:
                                _uc_mask = (
                                    torch.Tensor(
                                        [1] * (hidden_states.shape[0] // 2)
                                        + [0] * (hidden_states.shape[0] // 2)
                                    )
                                    .to(device)
                                    .bool()
                                )
                            latent_hidden_states_c[_uc_mask] = (
                                self.attn_latent[0](
                                    norm_hidden_states[_uc_mask],
                                    encoder_hidden_states=norm_hidden_states[_uc_mask],
                                    attention_mask=attention_mask,
                                ) + hidden_states[_uc_mask]
                            )
                            latent_hidden_states = latent_hidden_states_c.clone()
                        else:
                            latent_hidden_states = latent_hidden_states_uc
                    
                    if do_classifier_free_guidance:
                        hidden_states_c = hidden_states_uc.clone()
                        _uc_mask = uc_mask.clone()
                        if hidden_states.shape[0] != _uc_mask.shape[0]:
                            _uc_mask = (
                                torch.Tensor(
                                    [1] * (hidden_states.shape[0] // 2)
                                    + [0] * (hidden_states.shape[0] // 2)
                                )
                                .to(device)
                                .bool()
                            )
                        hidden_states_c[_uc_mask] = (
                            self.attn1(
                                norm_hidden_states[_uc_mask],
                                encoder_hidden_states=norm_hidden_states[_uc_mask],
                                attention_mask=attention_mask,
                            )
                            + hidden_states[_uc_mask]
                        )
                        hidden_states = hidden_states_c.clone()
                    else:
                        hidden_states = hidden_states_uc
                        
                    if do_latent_attention:
                        hidden_states = hidden_states + latent_hidden_states
                                                        
                    # self.bank.clear()
                    if self.attn2 is not None:
                        # Cross-Attention
                        norm_hidden_states = (
                            self.norm2(hidden_states, timestep)
                            if self.use_ada_layer_norm
                            else self.norm2(hidden_states)
                        )

                        if do_latent_attention:
                            latent_cross_hidden_states = (
                                self.attn_cross_latent[0](
                                    norm_hidden_states,
                                    encoder_hidden_states=self.latent_bank[0],
                                    attention_mask=attention_mask,
                                ) + hidden_states
                            )
                            latent_cross_hidden_states = latent_cross_hidden_states * latent_attn_masks.to(latent_cross_hidden_states.dtype)
                            self.latent_bank.clear()
                        
                        # if do_classifier_free_guidance:
                        #     latent_hidden_states_c = latent_hidden_states_uc.clone()
                        #     _uc_mask = uc_mask.clone()
                        #     if hidden_states.shape[0] != _uc_mask.shape[0]:
                        #         _uc_mask = (
                        #             torch.Tensor(
                        #                 [1] * (hidden_states.shape[0] // 2)
                        #                 + [0] * (hidden_states.shape[0] // 2)
                        #             )
                        #             .to(device)
                        #             .bool()
                        #         )
                        #     latent_hidden_states_c[_uc_mask] = (
                        #         self.attn_latent[0](
                        #             norm_hidden_states[_uc_mask],
                        #             encoder_hidden_states=norm_hidden_states[_uc_mask],
                        #             attention_mask=attention_mask,
                        #         ) + hidden_states[_uc_mask]
                        #     )
                        #     latent_hidden_states = latent_hidden_states_c.clone()
                        # else:
                        #     latent_hidden_states = latent_hidden_states_uc

                        hidden_states = (
                            self.attn2(
                                norm_hidden_states,
                                encoder_hidden_states=encoder_hidden_states,
                                attention_mask=attention_mask,
                                **cross_attention_kwargs,
                            )
                            + hidden_states
                        )

                        if do_latent_attention:
                            hidden_states = hidden_states + latent_cross_hidden_states 


                    # if write_latent:
                    #     self.latent_bank.append(hidden_states.clone())
                    
                    # if do_latent_attention:
                    #     assert not write_latent
                    #     if do_classifier_free_guidance and do_infer:
                    #         hidden_states_c = hidden_states[hidden_states.shape[0] // 2:, ...]
                    #         # hidden_states_c = hidden_states
                    #     else:
                    #         hidden_states_c = hidden_states
                    #     # latent_attn_masks: b*t, hie_num, h, w
                    #     hidden_states_c = self.norm_latent[0](hidden_states_c)
                        
                    #     # latent_hidden_states_lst = [self.norm_latent[0](
                    #     #     h for h in self.latent_bank
                    #     # )]
                    #     # latent_hidden_states = torch.cat([hidden_states_c] + latent_hidden_states_lst, dim=1)
                    #     # hidden_states_c = self.latent_attn[0](
                    #     #     hidden_states_c,
                    #     #     encoder_hidden_states=latent_hidden_states,
                    #     #     attention_mask=attention_mask,
                    #     # ) + hidden_states_c
                    #     # self.latent_bank.clear()
                        
                    #     assert latent_attn_masks.shape[1] == len(self.latent_bank)
                    #     # hidden_states_c = self.norm_latent(hidden_states_c)
                    #     hierarchy_num = len(self.latent_bank)
                    #     # hidden states: b*t, l, c                        
                    #     # hidden_states_c = hidden_states_c.unsqueeze(1).repeat(1, hierarchy_num+1, 1, 1)
                    #     hsize = int(math.sqrt(hidden_states_c.shape[-2]))
                        
                    #     B, mask_hn, mask_H, mask_W = latent_attn_masks.shape
                    #     assert mask_hn == hierarchy_num

                    #     hie_hidden_states_lst = []
                    #     latent_attn_masks = F.interpolate(latent_attn_masks, size=(hsize, hsize), mode="bilinear")              
                    #     for latent_idx, latent_hidden_states in enumerate(self.latent_bank): # b, hn, l, c
                    #         # latent_hidden_states = torch.stake(self.latent_bank, dim=1)  # b, hn-1, l, c
                    #         if do_classifier_free_guidance and do_infer:
                    #             latent_hidden_states = latent_hidden_states[latent_hidden_states.shape[0] // 2:, ...]
                    #         _, latent_HW, latent_C = latent_hidden_states.shape
                            
                    #         hie_attn_mask = latent_attn_masks[:, latent_idx, ...]
                    #         # hie_attn_mask = F.interpolate(hie_attn_mask.unsqueeze(1), size=(hsize, hsize), mode="bilinear")
                    #         hie_attn_mask = hie_attn_mask.reshape(B, hsize**2, 1)
                            
                    #         hie_hidden_states = hidden_states_c.clone()
                    #         latent_hidden_states = self.norm_latent[latent_idx+1](latent_hidden_states)
                    #         hie_hidden_states = self.attn_latent[latent_idx](
                    #             hidden_states=hie_hidden_states,
                    #             encoder_hidden_states=torch.cat([hie_hidden_states, latent_hidden_states], dim=1),
                    #             attention_mask=attention_mask,
                    #             # latent_attn_mask=latent_attn_masks,
                    #         )

                    #         hie_hidden_states = hie_hidden_states * hie_attn_mask
                    #         # hie_hidden_states = hie_hidden_states + hidden_states_c * hie_attn_mask
                    #         hie_hidden_states_lst.append(hie_hidden_states)
                        
                    #     self.latent_bank.clear() # clear latent bank
                        
                    #     # background feature
                    #     supp_mask = torch.prod(1 - latent_attn_masks, dim=1, keepdim=False)
                    #     supp_mask = F.interpolate(supp_mask.unsqueeze(1), size=(hsize, hsize), mode="bilinear")
                    #     ori_supp_mask = supp_mask.clone()
                    #     supp_mask = supp_mask.reshape(B, hsize**2, 1)
                    #     supp_hidden_states = hidden_states_c * supp_mask                        
                    #     # template = self.la(x=hidden_states_c, guidance_mask=torch.cat(
                    #     #     [ori_supp_mask, latent_attn_masks], dim=1
                    #     # ))
                    #     template = hidden_states_c
                        
                    #     # hidden_states_stacked = torch.stack([supp_hidden_states] + hie_hidden_states_lst + [template], dim=1) # B, hie_num+2, l, c
                    #     hidden_states_stacked = torch.stack([supp_hidden_states] + hie_hidden_states_lst, dim=1) # B, hie_num+2, l, c

                    #     # guidance_mask = torch.cat([
                    #     #     ori_supp_mask, latent_attn_masks, torch.ones(B, 1, hsize, hsize).to(ori_supp_mask.device),
                    #     # ], dim=1)
                    #     guidance_mask = torch.cat([
                    #         ori_supp_mask, latent_attn_masks,#, torch.ones(B, 1, hsize, hsize).to(ori_supp_mask.device),
                    #     ], dim=1)
                                                
                    #     hidden_states_c, sac_scale = self.sac(hidden_states_stacked, guidance_mask)
                    #     hidden_states_c = hidden_states_c.squeeze(1)
                    #     if do_classifier_free_guidance and do_infer:
                    #         hidden_states = torch.cat([hidden_states[:hidden_states.shape[0] // 2], hidden_states_c], dim=0)
                    #         # hidden_states = hidden_states_c
                    #     else:
                    #         hidden_states = hidden_states_c
                        
                    # Feed-forward
                    hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

                    # Temporal-Attention
                    if self.unet_use_temporal_attention:
                        d = hidden_states.shape[1]
                        hidden_states = rearrange(
                            hidden_states, "(b f) d c -> (b d) f c", f=video_length
                        )
                        norm_hidden_states = (
                            self.norm_temp(hidden_states, timestep)
                            if self.use_ada_layer_norm
                            else self.norm_temp(hidden_states)
                        )
                        hidden_states = (
                            self.attn_temp(norm_hidden_states) + hidden_states
                        )
                        hidden_states = rearrange(
                            hidden_states, "(b d) f c -> (b f) d c", d=d
                        )

                    return hidden_states

            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep)
                    if self.use_ada_layer_norm
                    else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states
                
            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = (
                    norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                )

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states

        if self.reference_attn:
            if self.fusion_blocks == "midup":
                attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            attn_modules = sorted(
                attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                if isinstance(module, BasicTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module, BasicTransformerBlock
                    )
                if isinstance(module, TemporalBasicTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module, TemporalBasicTransformerBlock
                    )

                module.bank = []
                module.attn_weight = float(i) / float(len(attn_modules))
                module.latent_bank = []
                
    def update(self, writer, dtype=torch.float16):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, TemporalBasicTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in (
                        torch_dfs(writer.unet.mid_block)
                        + torch_dfs(writer.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, TemporalBasicTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in torch_dfs(writer.unet)
                    if isinstance(module, BasicTransformerBlock)
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            writer_attn_modules = sorted(
                writer_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            for r, w in zip(reader_attn_modules, writer_attn_modules):
                r.bank = [v.clone().to(dtype) for v in w.bank]
                # r.bank.append([v.clone().to(dtype) for v in bank])
                # w.bank.clear()

    def clear(self):
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) + torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, BasicTransformerBlock)
                    or isinstance(module, TemporalBasicTransformerBlock)
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            for r in reader_attn_modules:
                r.bank.clear()
