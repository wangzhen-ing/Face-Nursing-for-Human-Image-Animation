import inspect
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np
import cv2
import torch
from diffusers import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.utils import BaseOutput, is_accelerate_available
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from tqdm import tqdm
from transformers import CLIPImageProcessor
# from models.multimutual_self_attention import ReferenceAttentionControl
from models.paramutual_cross_attention_fine3 import ReferenceAttentionControl
from insightface.utils import face_align
from PIL import Image
import torchvision.transforms as transforms
#NOTE(wz) pipeline for target image and target face image. 


@dataclass
class MultiGuidance2ImagePipelineOutput(BaseOutput):
    images: Union[torch.Tensor, np.ndarray]


class MultiGuidanceIPFine2ImagePipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae,
        image_encoder,
        reference_unet,
        denoising_unet,
        # guidance_encoder_depth,
        # guidance_encoder_normal,
        # guidance_encoder_semantic_map,
        guidance_encoder_DWpose,
        image_encoder_ip,
        image_proj_model,
        # fusion_module,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            image_encoder=image_encoder,
            reference_unet=reference_unet,
            denoising_unet=denoising_unet,
            # guidance_encoder_depth=guidance_encoder_depth,
            # guidance_encoder_normal=guidance_encoder_normal,            
            # guidance_encoder_semantic_map=guidance_encoder_semantic_map,
            guidance_encoder_DWpose=guidance_encoder_DWpose,
            image_encoder_ip=image_encoder_ip,
            image_proj_model=image_proj_model,
            # fusion_module=fusion_module,
            scheduler=scheduler,
        )
        # self.vae=vae.to(dtype=torch.float32),
        # self.image_encoder=image_encoder.to(dtype=torch.float32),
        # self.reference_unet=reference_unet,
        # self.denoising_unet=denoising_unet,
        #     # guidance_encoder_depth=guidance_encoder_depth,
        #     # guidance_encoder_normal=guidance_encoder_normal,            
        #     # guidance_encoder_semantic_map=guidance_encoder_semantic_map,
        # self.guidance_encoder_DWpose=guidance_encoder_DWpose,
        # self.image_encoder_ip=image_encoder_ip.to(dtype=torch.float32),
        # self.image_proj_model=image_proj_model.to(dtype=torch.float32),
        #     # fusion_module=fusion_module,
        # self.scheduler=scheduler,



        
        # guidance_encoder_DWpose.update({'DWpose':guidance_encoder_DWpose.pop("guidance_encoder_DWpose")})

        self.guidance_input_channels = []
        self.guidance_types = []
        # for guidance_type, guidance_module in guidance_encoder_DWpose.items():
        #     setattr(self, f"guidance_encoder_{guidance_type}", guidance_module)
        #     self.guidance_types.append(guidance_type)
        #     self.guidance_input_channels.append(guidance_module.guidance_input_channels)
        # setattr(self, "guidance_encoder_DWpose", guidance_encoder_DWpose["guidance_encoder_DWpose"])
        self.guidance_types.append("DWpose")
        self.guidance_input_channels.append(guidance_encoder_DWpose.guidance_input_channels)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.clip_image_processor = CLIPImageProcessor()
        self.ref_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.ipadapter_mask_processor = IPAdapterMaskProcessor()

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        # video = self.vae.decode(latents).sample
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx : frame_idx + 1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        video = video.cpu().float().numpy()
        return video

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        width,
        height,
        video_length,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def __call__(
        self,
        ref_image_latents,
        region_image_latents,
        clip_image_embeds,
        image_prompt_embeds,
        clip_image,
        multi_guidance_cond,
        region_multi_guidance_cond,
        region_images_embeds,
        region_clip_images,
        id_embeds,
        # ref_idx,
        uncond_fwd,
        write_latent,
        do_latent_attention,
        batch_size,
        width,
        height,
        video_length,
        face_image_size,
        num_inference_steps,
        guidance_scale,
        num_images_per_prompt=1,
        latent_attn_masks=None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "tensor",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):  
        
        # Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # batch_size = 1

        # Prepare clip image embeds
        # clip_image = self.clip_image_processor.preprocess(
        #     ref_image.resize((224, 224)), return_tensors="pt"
        # ).pixel_values
        # clip_image_embeds = self.image_encoder(
        #     clip_image.to(device, dtype=self.image_encoder.dtype)
        # ).image_embeds
        # image_prompt_embeds = clip_image_embeds.unsqueeze(1)

        uncond_image_prompt_embeds = self.image_encoder(
            torch.zeros_like(clip_image).to(device, dtype=self.image_encoder.dtype)
        ).image_embeds
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.unsqueeze(1)
        

        # id_image = cv2.imread(ref_img_path)
        # faces = face_app.get(id_image)
        # id_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0).to(device, dtype=self.image_encoder_ip.dtype)
        # id_embeds = id_embeds.to(device, dtype=self.image_encoder_ip.dtype)
        uncond_id_embeds = torch.zeros_like(id_embeds).to(device, dtype=self.image_encoder_ip.dtype)
        # print(self.image_encoder_ip.dtype)

        # face_image = face_align.norm_crop(id_image, landmark=faces[0].kps, image_size=224)
        # region_clip_image = self.clip_image_processor.preprocess(
        #     images=face_image, return_tensors="pt"
        # ).pixel_values
        # region_image_embeds = self.image_encoder_ip(
        #     region_clip_image.to(device, dtype=self.image_encoder_ip.dtype),
        #     output_hidden_states=True,
        # ).hidden_states[-2]
        uncond_region_images_embeds = self.image_encoder_ip(
            torch.zeros_like(region_clip_images).to(device, dtype=self.image_encoder_ip.dtype),
            output_hidden_states=True,
        ).hidden_states[-2]
        # print(region_images_embeds.shape)
        # torch.Size([4, 257, 1280])
        
        if do_classifier_free_guidance:
            image_prompt_embeds = torch.cat(
                [uncond_image_prompt_embeds, image_prompt_embeds], dim=0
            )
            region_images_embeds = torch.cat(
                [uncond_region_images_embeds, region_images_embeds], dim=0
            )
            id_embeds = torch.stack(
                [uncond_id_embeds, id_embeds], dim=0
            )
        
        # print("id_embeds---------")
        # print(id_embeds.shape)
        # torch.Size([2, 4, 1, 512])


        ip_tokens = self.image_proj_model(id_embeds, region_images_embeds)
        # print(ip_tokens.shape)
        # # torch.Size([8, 4, 768])
        # print(region_images_embeds.shape)
        # # torch.Size([8, 257, 1280])

        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
        )

        num_channels_latents = self.denoising_unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            width,
            height,
            video_length,
            clip_image_embeds.dtype,
            device,
            generator,
        )
        # latents = latents.unsqueeze(2)  # (bs, c, 1, h', w')
        latents_dtype = latents.dtype

        face_latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            face_image_size,
            face_image_size,
            video_length,
            clip_image_embeds.dtype,
            device,
            generator,
        )
        # face_latents = face_latents.unsqueeze(2)
        # print("qwerty")
        # print(face_latents.size())
        
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # # Prepare ref image latents
        # ref_image_tensor = self.ref_image_processor.preprocess(
        #     ref_image, height=height, width=width
        # )  # (bs, c, width, height)
        # ref_image_tensor = ref_image_tensor.to(
        #     dtype=self.vae.dtype, device=self.vae.device
        # )
        # ref_image_latents = self.vae.encode(ref_image_tensor).latent_dist.mean
        # ref_image_latents = ref_image_latents * 0.18215  # (b, 4, h, w)

        # region_image_tensor = self.ref_image_processor.preprocess(
        #     Image.fromarray(face_image[..., ::-1]), height=height, width=width,
        # )
        # region_image_tensor = region_image_tensor.to(
        #     dtype=self.vae.dtype, device=self.vae.device
        # )
        # region_image_latents = self.vae.encode(region_image_tensor).latent_dist.mean
        # region_image_latents = region_image_latents * 0.18215

         #--------------------------guidance feature--------------
        guidance_cond_group = torch.split(
            multi_guidance_cond, self.guidance_input_channels, dim=1
        )
        face_guidance_cond_group = torch.split(
            region_multi_guidance_cond, self.guidance_input_channels, dim=1
        )
        guidance_fea_lst = []
        face_guidance_fea_lst = []
        for guidance_idx, guidance_cond in enumerate(guidance_cond_group):
            guidance_encoder = getattr(
                self, f"guidance_encoder_{self.guidance_types[guidance_idx]}"
            )
            guidance_fea = guidance_encoder(guidance_cond)
            guidance_fea_lst += [guidance_fea]

            # NOTE(ZSH): face guidance should follow the order of body guidance
            face_guidance_cond = face_guidance_cond_group[guidance_idx]
            face_guidance_fea = guidance_encoder(face_guidance_cond)
            face_guidance_fea_lst += [face_guidance_fea]

        guidance_fea = torch.stack(guidance_fea_lst, dim=0).sum(0)
        guidance_fea = torch.cat([guidance_fea] * 2) if do_classifier_free_guidance else guidance_fea
        # print("00000000000000")
        # print(face_guidance_fea.shape)
        # print("zzzzzzzzzzzzzzzzz")

        
        face_guidance_fea = torch.stack(face_guidance_fea_lst, dim=0).sum(0)
        face_guidance_fea = torch.cat([face_guidance_fea] * 2) if do_classifier_free_guidance else face_guidance_fea
        #--------------------------------------------------------------
        

        # face_mask = transforms.ToTensor()(face_mask).to(device=device, dtype=self.vae.dtype)
        # # face_mask = torch.stack([face_mask * 2]) if do_classifier_free_guidance else face_mask
        # latent_attn_masks = torch.stack([face_mask], dim=1)
        # denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # 1. Forward reference image
                if i == 0:
                    print(ref_image_latents.dtype, image_prompt_embeds.dtype)
                    self.reference_unet(
                        ref_image_latents.repeat(
                            (2 if do_classifier_free_guidance else 1), 1, 1, 1
                        ),
                        torch.zeros_like(t),
                        encoder_hidden_states=image_prompt_embeds,
                        return_dict=False,
                    )
                    
                    self.reference_unet(
                        region_image_latents.repeat(
                            (2 if do_classifier_free_guidance else 1), 1, 1, 1
                        ),
                        torch.zeros_like(t),
                        encoder_hidden_states=ip_tokens,
                        return_dict=False,
                    )

                    # 2. Update reference unet feature into denosing net
                    reference_control_reader.update(reference_control_writer)

                # 3.1 expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                face_latent_input = (
                    torch.cat([face_latents] * 2) if do_classifier_free_guidance else face_latents
                )
                face_latent_input = self.scheduler.scale_model_input(
                    face_latent_input, t
                )
                # print("111111111111111111111111111234567890")
                # print(face_latent_input.shape)
                # print(face_guidance_fea.shape)
                # print(face_latents.size())

                # print(face_latent_input.dtype, ip_tokens.dtype, face_guidance_fea.dtype)
                face_pred, face_latents_pred = self.denoising_unet(
                    face_latent_input,
                    t,
                    encoder_hidden_states=ip_tokens,
                    guidance_fea=face_guidance_fea,
                    return_dict=False,
                    ref_idx=(1,),
                    # cross_attention_kwargs={"ip_adapter_masks": region_mask}
                    additional_upsample=False,
                    write_latent=False,
                    do_latent_attention=False,
                )

                noise_pred, model_latents_pred = self.denoising_unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=image_prompt_embeds,
                    guidance_fea=guidance_fea,
                    return_dict=False,
                    ref_idx=(0,),
                    # cross_attention_kwargs={"ip_adapter_masks": region_mask}
                    additional_upsample=False,
                    write_latent=False,
                    do_latent_attention=False,
                    latent_attn_masks=None,
                    do_infer=False,
                )

                # fusion_pred = self.fusion_module(model_latents_pred, face_latents_pred)

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    face_pred_uncond, face_pred_text = face_pred.chunk(2)
                    face_pred = face_pred_uncond + guidance_scale * (
                        face_pred_text - face_pred_uncond
                    )
                    # fusion_pred_uncond, fusion_pred_text = fusion_pred.chunk(2)
                    # fusion_pred = fusion_pred_uncond + guidance_scale * (
                    #     fusion_pred_text - fusion_pred_uncond
                    # )                    

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]                
                face_latents = self.scheduler.step(
                    face_pred, t, face_latents, **extra_step_kwargs, return_dict=False
                )[0] 
                # upsampled_latents = self.scheduler.step(
                #     fusion_pred, t, upsampled_latents, **extra_step_kwargs, return_dict=False
                # )[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)
            reference_control_reader.clear()
            reference_control_writer.clear()


        # Post-processing
        image = self.decode_latents(latents)  # (b, c, f, h, w)
        face_image = self.decode_latents(face_latents)
        # upsampled_image = self.decode_latents(upsampled_latents)
        # Convert to tensor
        if output_type == "tensor":
            image = torch.from_numpy(image)
            face_image = torch.from_numpy(face_image)
            # upsampled_image = torch.from_numpy(upsampled_image)

        if not return_dict:
            return (image, face_image)#, upsampled_image)

        return MultiGuidance2ImagePipelineOutput(images=upsampled_image)

