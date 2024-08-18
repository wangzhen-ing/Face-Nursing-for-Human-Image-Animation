import torch
import torch.nn as nn
from models.unet_2d_condition import UNet2DConditionModel
from models.unet_3d import UNet3DConditionModel
from models.resnet import InflatedConv3d, InflatedGroupNorm
from models.transformer_3d import FusionTransformer3DModel
import torch.nn.functional as F

class ChampFaceModel(nn.Module):
    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        reference_control_writer,
        reference_control_reader,
        guidance_encoder_group,
        image_proj_model,
        adapter_modules,
        adapter_ckpt_path=None,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet

        self.reference_control_writer = reference_control_writer
        self.reference_control_reader = reference_control_reader

        self.guidance_types = []
        self.guidance_input_channels = []

        for guidance_type, guidance_module in guidance_encoder_group.items():
            setattr(self, f"guidance_encoder_{guidance_type}", guidance_module)
            self.guidance_types.append(guidance_type)
            self.guidance_input_channels.append(guidance_module.guidance_input_channels)

        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if adapter_ckpt_path is not None:
            self.load_vanilla_adapter(adapter_ckpt_path)

        # self.fusion_module = FusionModule()
        # self.fusion_module.initialize_from_unet(denoising_unet)

    def forward(
        self,
        noisy_latents,
        face_noisy_latents,
        timesteps,
        ref_image_latents,
        region_image_latents,
        clip_image_embeds,
        multi_guidance_cond,
        region_multi_guidance_cond,
        region_images_embeds,
        id_embeds,
        # region_masks,
        ref_idx,
        uncond_fwd,
        write_latent,
        do_latent_attention,
        latent_attn_masks=None,
    ):
        guidance_cond_group = torch.split(
            multi_guidance_cond, self.guidance_input_channels, dim=1
        )
        # print(len(guidance_cond_group))
        # print(region_multi_guidance_cond.shape)
        # print(self.guidance_input_channels)
        # torch.Size([4, 12, 1, 768, 768])
        # torch.Size([4, 3, 1, 256, 256])
        # [3, 3, 3, 3]
        # region_multi_guidance_cond = region_multi_guidance_cond.repeat(1,4,1,1,1)

 
        # face_guidance_cond_group = torch.split(
        #     region_multi_guidance_cond, self.guidance_input_channels, dim=1
        # )

        guidance_fea_lst = []
        face_guidance_fea_lst = []
        for guidance_idx, guidance_cond in enumerate(guidance_cond_group):
            guidance_encoder = getattr(
                self, f"guidance_encoder_{self.guidance_types[guidance_idx]}"
            )
            guidance_fea = guidance_encoder(guidance_cond)
            guidance_fea_lst += [guidance_fea]

            # NOTE(ZSH): face guidance should follow the order of body guidance
            # face_guidance_cond = face_guidance_cond_group[guidance_idx]
            # face_guidance_fea = guidance_encoder(face_guidance_cond)
            # face_guidance_fea_lst += [face_guidance_fea]
        # face_guidance_cond = face_guidance_cond_group[guidance_idx]
        face_guidance_encoder = getattr(
                self, f"guidance_encoder_DWpose"
            )

        face_guidance_fea = face_guidance_encoder(region_multi_guidance_cond)

        guidance_fea = torch.stack(guidance_fea_lst, dim=0).sum(0)
        # print("11233")
        # print(face_guidance_fea.shape)
        # print(guidance_fea.shape)
        # torch.Size([4, 320, 1, 32, 32])
        # torch.Size([4, 320, 1, 96, 96])
        # print("00990")



        # print(guidance_fea.shape)
        # torch.Size([4, 320, 1, 96, 96])
        # face_guidance_fea = torch.stack(face_guidance_fea_lst, dim=0).sum(0)
        # print("face_guidance shape")

        # torch.Size([8, 320, 1, 32, 32])
        # torch.Size([8, 320, 1, 32, 32])
        # face_guidance shape
        # torch.Size([8, 320, 1, 32, 32])
        # face_guidance shape
        # torch.Size([8, 320, 1, 32, 32])
        # face_guidance shape
        # torch.Size([8, 320, 1, 32, 32])
        # face_guidance shape
        # torch.Size([8, 320, 1, 32, 32])
        # face_guidance shape
        # torch.Size([8, 320, 1, 32, 32])
        # face_guidance shape
        # torch.Size([8, 320, 1, 32, 32])

        # print(face_guidance_fea.shape)

        #NOTE(WZ): operate face motion by GaussianSmoothing
        # if smooth_attentions:
        #     smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
        #     input = F.pad(face_guidance_fea.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
        #     face_guidance_fea = smoothing(input).squeeze(0).squeeze(0)

        ip_tokens = self.image_proj_model(id_embeds, region_images_embeds)
        # print("--090989090909")
        # print(ip_tokens.shape)
        # print(ip_tokens.shape)
        # print(ip_tokens)
        # torch.Size([8, 4, 768])


        # ip_tokens_lst = [ip_tokens[i].unsqueeze(0) for i in range(ip_tokens.shape[0])]
        # encoder_hidden_states = (clip_image_embeds, ip_tokens)

        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            # self.reference_control_reader.update(self.reference_control_writer)
            # self.reference_unet(
            #     region_image_latents,
            #     ref_timesteps,
            #     encoder_hidden_states=ip_tokens,
            #     return_dict=False,
            # )
            self.reference_control_reader.update(self.reference_control_writer)

        # if self.adapter_modules is not None:
        #     model_encoder_hidden_states = encoder_hidden_states
        # else:
        model_encoder_hidden_states = clip_image_embeds
        face_ref_idx=(1,) if not uncond_fwd else None
        # face_timesteps = torch.zeros_like(timesteps)
        face_timesteps = timesteps
        # face_pred, face_latents = self.denoising_unet(
        #     face_noisy_latents,
        #     face_timesteps,
        #     guidance_fea=face_guidance_fea,
        #     encoder_hidden_states=ip_tokens,
        #     ref_idx=face_ref_idx,
        #     write_latent=write_latent,
        #     do_latent_attention=False,
        # ).sample
        
        ref_idx=(0,) if not uncond_fwd else None
        model_pred, model_latents = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=guidance_fea,
            encoder_hidden_states=model_encoder_hidden_states,
            # cross_attention_kwargs={"ip_adapter_masks": region_masks}
            ref_idx=ref_idx,
            write_latent=False,
            do_latent_attention=do_latent_attention,
            ip_tokens=ip_tokens,
            latent_attn_masks=latent_attn_masks,
        ).sample
        
        # ref_idx = (ref_idx,) if isinstance(ref_idx, int) and not uncond_fwd else None


        # fusion_pred = self.fusion_module(model_latents, face_latents)

        return model_pred
    
    def load_vanilla_adapter(self, ckpt_path: str):
        load_adapter = self.adapter_modules is not None
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        if load_adapter:
            orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        strict_load_image_proj_model = True
        if "latents" in state_dict["image_proj"] and "latents" in self.image_proj_model.state_dict():
            # Check if the shapes are mismatched
            if state_dict["image_proj"]["latents"].shape != self.image_proj_model.state_dict()["latents"].shape:
                print(f"Shapes of 'image_proj.latents' in checkpoint {ckpt_path} and current model do not match.")
                print("Removing 'latents' from checkpoint and loading the rest of the weights.")
                del state_dict["image_proj"]["latents"]
                strict_load_image_proj_model = False
        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=strict_load_image_proj_model)
        if load_adapter:
            self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=False)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        if load_adapter:
            new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))
            assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")


class UpSampler(nn.Module):
    def __init__(
        self,
        channels,
        out_channels=None,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        
        self.conv = InflatedConv3d(self.channels, self.out_channels, kernel_size=3, padding=1)
        
    def forward(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels
        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()
            
        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(
                hidden_states, scale_factor=[1.0, 2.0, 2.0], mode="nearest"
            )
        else:
            hidden_states = F.interpolate(
                hidden_states, size=output_size, mode="nearest"
            )        
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)
            
        hidden_states = self.conv(hidden_states)

        return hidden_states                        

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)


    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)
        
        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1) # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)

class FusionModule(nn.Module):
    def __init__(
        self,
        channels=320,
        num_attention_heads=8,
        norm_num_groups=32,
        block_out_channels=320,
        norm_eps=1e-5,
        out_channels=4,
        use_inflated_groupnorm=False,
    ):
        super().__init__()
        # self.upsampler = UpSampler(channels=channels, out_channels=channels)
        self.fusion_attn = FusionTransformer3DModel(
            num_attention_heads=num_attention_heads,
            attention_head_dim=channels // num_attention_heads,
            in_channels=channels,
            norm_num_groups=norm_num_groups,
            unet_use_cross_frame_attention=False,
            unet_use_temporal_attention=False,
        )
        
        
        if use_inflated_groupnorm:
            self.conv_norm_out = InflatedGroupNorm(
                num_channels=block_out_channels,
                num_groups=norm_num_groups,
                eps=norm_eps,
            )
        else:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels,
                num_groups=norm_num_groups,
                eps=norm_eps,
            )        
        self.conv_act = nn.SiLU()
        self.conv_out = InflatedConv3d(
            block_out_channels, out_channels, kernel_size=3, padding=1
        )
        
    def forward(self, upsampled_unet_latents, face_latents):
        # upsampled_unet_latents = self.upsampler(unet_latents)
        fusion_latents = self.fusion_attn(
            hidden_states=upsampled_unet_latents,
            face_hidden_states=face_latents,
        ).sample

        fusion_latents = self.conv_norm_out(fusion_latents)
        fusion_latents = self.conv_act(fusion_latents)
        fusion_latents = self.conv_out(fusion_latents)

        return fusion_latents

    def initialize_from_unet(self, unet):
        self.conv_norm_out.weight.data = unet.conv_norm_out.weight.data.clone()
        self.conv_norm_out.bias.data = unet.conv_norm_out.bias.data.clone()
        self.conv_out.weight.data = unet.conv_out.weight.data.clone()
        self.conv_out.bias.data = unet.conv_out.bias.data.clone()

    def to(self, *args, **kwargs):
        super_result = super(FusionModule, self).to(*args, **kwargs)
        if 'dtype' in kwargs:
            self.dtype = kwargs['dtype']
        else:
            for arg in args:
                if isinstance(arg, torch.dtype):
                    self.dtype = arg
                    break
        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            for arg in args:
                if isinstance(arg, (torch.device, str)):
                    self.dtype = arg
                    break
        return super_result
