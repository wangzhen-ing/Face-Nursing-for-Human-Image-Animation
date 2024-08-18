import torch
import torch.nn as nn
from models.unet_2d_condition import UNet2DConditionModel
from models.unet_3d import UNet3DConditionModel


class ChampIPModel(nn.Module):
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

    def forward(
        self,
        noisy_latents,
        timesteps,
        ref_image_latents,
        region_image_latents,
        clip_image_embeds,
        multi_guidance_cond,
        region_images_embeds,
        id_embeds,
        region_masks,
        uncond_fwd: bool = False,
    ):
        guidance_cond_group = torch.split(
            multi_guidance_cond, self.guidance_input_channels, dim=1
        )
        guidance_fea_lst = []
        for guidance_idx, guidance_cond in enumerate(guidance_cond_group):
            guidance_encoder = getattr(
                self, f"guidance_encoder_{self.guidance_types[guidance_idx]}"
            )
            guidance_fea = guidance_encoder(guidance_cond)
            guidance_fea_lst += [guidance_fea]

        guidance_fea = torch.stack(guidance_fea_lst, dim=0).sum(0)

        ip_tokens = self.image_proj_model(id_embeds, region_images_embeds)
        # ip_tokens_lst = [ip_tokens[i].unsqueeze(0) for i in range(ip_tokens.shape[0])]
        encoder_hidden_states = (clip_image_embeds, ip_tokens)

        if not uncond_fwd:
            ref_timesteps = torch.zeros_like(timesteps)
            self.reference_unet(
                ref_image_latents,
                ref_timesteps,
                encoder_hidden_states=clip_image_embeds,
                return_dict=False,
            )
            # self.reference_control_reader.update(self.reference_control_writer)
            self.reference_unet(
                region_image_latents,
                ref_timesteps,
                encoder_hidden_states=ip_tokens,
                return_dict=False,
            )
            self.reference_control_reader.update(self.reference_control_writer)

        if self.adapter_modules is not None:
            model_encoder_hidden_states = encoder_hidden_states
        else:
            model_encoder_hidden_states = clip_image_embeds
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
            guidance_fea=guidance_fea,
            encoder_hidden_states=model_encoder_hidden_states,
            cross_attention_kwargs={"ip_adapter_masks": region_masks}
        ).sample

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
