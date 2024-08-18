import argparse
import logging
import math
import os
import os.path as osp
import random
import warnings
from pathlib import Path
import cv2

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from torchvision.utils import save_image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from omegaconf import OmegaConf
from PIL import Image, ImageOps
from tqdm.auto import tqdm
from transformers import CLIPVisionModelWithProjection
from transformers import CLIPImageProcessor

# from models.champ_model_face import ChampIPModel
from models.champ_model_add_face_fine import ChampFaceModel
from models.guidance_encoder import GuidanceEncoder
from models.unet_2d_condition import UNet2DConditionModel
from models.unet_3d_face import UNet3DConditionModel
# from models.multimutual_self_attention import ReferenceAttentionControl
# from models.paramutual_self_attention import ReferenceAttentionControl
# from models.paramutual_self_cross_attention import ReferenceAttentionControl
from models.paramutual_cross_attention_fine import ReferenceAttentionControl
from models.resampler import Resampler, ProjPlusModel, ImageProjModel

from datasets.image_face_dataset_v2_fine import ImageFaceFineDataset
from datasets.data_utils import mask_to_bkgd
from utils.tb_tracker import TbTracker
from utils.util import seed_everything, delete_additional_ckpt, compute_snr, is_torch2_available
from insightface.app import FaceAnalysis

from pipelines.pipeline_addface_inference_fine import MultiGuidanceIPFine2ImagePipeline
from pipelines.pipeline_addface_train_fine import MultiGuidanceIPFine2ImagePipeline_train


if is_torch2_available():
    from models.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
# else:
#     from models.attention_processor import IPAttnProcessor, AttnProcessor

#NOTE(wz) inject fine feature extract from target face image.

warnings.filterwarnings("ignore")
check_min_version("0.10.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def torch_dfs(model: torch.nn.Module):
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

def padding_pil(img_pil, img_size):
    # resize a PIL image and zero padding the short edge
    W, H = img_pil.size
    resize_ratio = img_size / max(W, H)
    new_W, new_H = int(W * resize_ratio), int(H * resize_ratio)
    img_pil = img_pil.resize((new_W, new_H))
    
    left = (img_size - new_W) // 2
    right = img_size - new_W - left
    top = (img_size - new_H) // 2
    bottom = img_size - new_H - top
    
    padding_border = (left, top, right, bottom)
    img_pil = ImageOps.expand(img_pil, border=padding_border, fill=0)
    
    return img_pil

def concat_pil(img_pil_lst):
    # horizontally concat PIL images 
    # NOTE(ZSH): assume all images are of same size
    W, H = img_pil_lst[0].size
    num_img = len(img_pil_lst)
    new_width = num_img * W
    new_image = Image.new("RGB", (new_width, H), color=0)
    for img_idx, img in enumerate(img_pil_lst):
        new_image.paste(img, (W * img_idx, 0))  
    
    return new_image

def validate(
    ref_img_path,
    guid_folder,
    guid_types,
    face_guid_types,
    guid_idx,
    region_img_path,
    # region_mask_path,
    width, height,
    face_image_size,
    pipe,
    face_app,
    generator,
    denoising_steps=20,
    guidance_scale=3.5,
    aug_type="Padding",
):
    ref_img_pil = Image.open(ref_img_path)
    guid_folder = Path(guid_folder)
    guid_img_pil_lst = []
    for guid_type in guid_types:
        guid_img_lst = sorted((guid_folder / guid_type).iterdir())
        guid_img_path = guid_img_lst[guid_idx]
        if guid_type == "semantic_map":
            mask_img_path = guid_folder / "mask" / guid_img_path.name
            guid_img_pil = mask_to_bkgd(guid_img_path, mask_img_path)
        else:
            guid_img_pil = Image.open(guid_img_path).convert("RGB")
        if aug_type == "Padding":
            guid_img_pil = padding_pil(guid_img_pil, height)
        guid_img_pil_lst += [guid_img_pil]
    region_img_pil = Image.fromarray(np.array(Image.open(region_img_path)))
    face_guid_img_pil_lst = []
    for face_guid_type in face_guid_types:
        face_guid_img_lst = sorted((guid_folder / face_guid_type).iterdir())
        face_guid_img_path = face_guid_img_lst[guid_idx]
        face_guid_pil = Image.open(face_guid_img_path)
        face_guid_img_pil_lst += [face_guid_pil]

    region_mask_path = guid_folder / "face_masks" / guid_img_path.name
    region_mask_pil = Image.open(region_mask_path).convert("L") if region_mask_path.exists() else Image.fromarray(np.zeros_like(np.array(ref_img_pil))[..., 0]).convert("L")
    
    if aug_type == "Padding":
        ref_img_pil = padding_pil(ref_img_pil, height)
        region_mask_pil = padding_pil(region_mask_pil, height)

    with torch.autocast("cuda"):
        val_images, face_val_images = pipe(
            ref_img_pil,
            guid_img_pil_lst,
            face_guid_img_pil_lst,
            guid_types,
            face_guid_types,
            region_img_pil,
            region_mask_pil,
            ref_img_path,
            face_app,
            width,
            height,
            face_image_size,
            denoising_steps,
            guidance_scale,
            generator=generator,
            output_type="tensor",
            # weight_dtype=torch.float16,
            return_dict=False,
        )
    
    
    # val_images, face_val_images = pipe(
    #     ref_img_pil,
    #     guid_img_pil_lst,
    #     face_guid_img_pil_lst,
    #     guid_types,
    #     face_guid_types,
    #     region_img_pil,
    #     region_mask_pil,
    #     ref_img_path,
    #     face_app,
    #     width,
    #     height,
    #     face_image_size,
    #     denoising_steps,
    #     guidance_scale,
    #     generator=generator,
    #     output_type="tensor",
    #     weight_dtype=torch.float16,
    #     return_dict=False,
    # )
    
    return val_images, ref_img_pil, guid_img_pil_lst, face_val_images#, upsampled_val_images

def log_validation(
    cfg,
    vae,
    image_enc,
    image_enc_ip,
    model,
    scheduler,
    accelerator,
    width,
    height,
    face_image_size,
    face_app,
    seed=42,
    dtype=torch.float32,
):
    logger.info("Running validation ...")
    unwrap_model = accelerator.unwrap_model(model)
    reference_unet = unwrap_model.reference_unet
    denoising_unet = unwrap_model.denoising_unet
    image_proj_model = unwrap_model.image_proj_model
    image_proj_model_fine = unwrap_model.image_proj_model_fine

    # fusion_module = unwrap_model.fusion_module
    guid_types = unwrap_model.guidance_types
    guidance_encoder_group = {
        f"guidance_encoder_{g}": getattr(unwrap_model, f"guidance_encoder_{g}") for g in guid_types
    }
    
    generator = torch.manual_seed(seed)
    vae = vae.to(dtype=dtype)
    image_enc = image_enc.to(dtype=dtype)
    image_enc_ip = image_enc_ip.to(dtype=dtype)
    
    pipeline = MultiGuidanceIPFine2ImagePipeline_train(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        **guidance_encoder_group,
        image_encoder_ip=image_enc_ip,
        image_proj_model=image_proj_model,
        image_proj_model_fine=image_proj_model_fine,
        # fusion_module=fusion_module,
        scheduler=scheduler,
    )
    pipeline = pipeline.to(device=accelerator.device)
    
    ref_img_lst = cfg.validation.ref_images
    guid_folder_lst = cfg.validation.guidance_folders
    guid_idxes = cfg.validation.guidance_indexes
    region_img_lst = cfg.validation.region_images
    
    # region_mask_lst = cfg.validation.region_masks
    
    val_results = []
    for val_idx, (ref_img_path, guid_folder, guid_idx, region_img_path) in enumerate(
        zip(ref_img_lst, guid_folder_lst, guid_idxes, region_img_lst)):
        
        image_tensor, ref_img_pil, guid_img_pil_lst, face_image_tensor = validate(
            ref_img_path=ref_img_path,
            guid_folder=guid_folder,
            guid_types=guid_types,
            face_guid_types=cfg.data.face_guids,
            guid_idx=guid_idx,
            region_img_path=region_img_path,
            # region_mask_path=region_mask_path,
            width=width,
            height=height,
            face_image_size=cfg.data.face_image_size,
            pipe=pipeline,
            face_app=face_app,
            generator=generator,
            aug_type=cfg.validation.aug_type,
        )
        
        image_tensor = image_tensor[0, :, 0].permute(1, 2, 0).cpu().numpy()
        face_tensor = face_image_tensor[0, :, 0].permute(1, 2, 0).cpu().numpy()
        # upsampled_tensor = upsampled_image_tensor[0, :, 0].permute(1, 2, 0).cpu().numpy()
        W, H = ref_img_pil.size
        result_img_pil = Image.fromarray((image_tensor * 255).astype(np.uint8))
        result_img_pil = result_img_pil.resize((W, H))
        face_img_pil = Image.fromarray((face_tensor * 255).astype(np.uint8))
        face_img_pil = face_img_pil.resize((W, H))
        # up_img_pil = Image.fromarray((upsampled_tensor * 255).astype(np.uint8))
        # up_img_pil = up_img_pil.resize((W, H))
        guid_img_pil_lst = [img.resize((W, H)) for img in guid_img_pil_lst]
        result_pil_lst = [result_img_pil, face_img_pil, ref_img_pil, *guid_img_pil_lst]
        concated_pil = concat_pil(result_pil_lst)
        
        val_results.append({"name": f"val_{val_idx}", "img": concated_pil})
    
    vae = vae.to(dtype=torch.float16)
    image_enc = image_enc.to(dtype=torch.float16)
    image_enc_ip = image_enc_ip.to(dtype=torch.float16)

    del pipeline
    torch.cuda.empty_cache()

    return val_results

def setup_guidance_encoder(cfg):
    guidance_encoder_group = dict()

    for guidance_type in cfg.data.guids:
        guidance_encoder_group[guidance_type] = GuidanceEncoder(
            guidance_embedding_channels=cfg.guidance_encoder_kwargs.guidance_embedding_channels,
            guidance_input_channels=cfg.guidance_encoder_kwargs.guidance_input_channels,
            block_out_channels=cfg.guidance_encoder_kwargs.block_out_channels,
        )

    return guidance_encoder_group

def load_stage1_state_dict(
    denoising_unet,
    reference_unet,
    guidance_encoder_group,
    image_proj_model,
    image_proj_model_fine,
    stage1_ckpt_dir, stage1_ckpt_step="latest",
):
    if stage1_ckpt_step == "latest":
        ckpt_files = sorted(os.listdir(stage1_ckpt_dir), key=lambda x: int(x.split("-")[-1].split(".")[0]))
        latest_pth_name = (Path(stage1_ckpt_dir) / ckpt_files[-1]).stem
        stage1_ckpt_step = int(latest_pth_name.split("-")[-1])
    
    denoising_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"denoising_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"reference_unet-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        ),
        strict=False,
    )
    for k, module in guidance_encoder_group.items():
        module.load_state_dict(
            torch.load(
                osp.join(stage1_ckpt_dir, f"guidance_encoder_{k}-{stage1_ckpt_step}.pth"),
                map_location="cpu",
            ),
            strict=False,
        )
    image_proj_model.load_state_dict(
        torch.load(
            os.path.join(stage1_ckpt_dir, f"image_proj_model-{stage1_ckpt_step}.pth"),
            map_location="cpu",
        )
    )
    # image_proj_model_fine.load_state_dict(
    #     torch.load(
    #         os.path.join(stage1_ckpt_dir, f"image_proj_model-{stage1_ckpt_step}.pth"),
    #         map_location="cpu",
    #     )
    # )
    
    logger.info(f"Loaded stage1 models from {stage1_ckpt_dir}, step={stage1_ckpt_step}")

def set_requires_grad(model, module_name):
    module_wgrad = []
    for name, module in model.named_modules():
        if any(m in name for m in module_name):
            for params in module.parameters():
                params.requires_grad = True
            module_wgrad.append(name)
    logger.info(f"{module_wgrad} setted require grad")

def main(cfg):
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    tb_tracker = TbTracker(cfg.exp_name, cfg.output_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
        mixed_precision=cfg.solver.mixed_precision,
        log_with=tb_tracker,
        project_dir=f'{cfg.output_dir}/{cfg.exp_name}',
        kwargs_handlers=[kwargs],
    )
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
        
    if cfg.seed is not None:
        seed_everything(cfg.seed)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training"
        )

    sched_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        sched_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**sched_kwargs)
    sched_kwargs.update({"beta_schedule": "scaled_linear"})
    train_noise_scheduler = DDIMScheduler(**sched_kwargs)
    
    vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(
        "cuda", dtype=weight_dtype
    )

    reference_unet = UNet2DConditionModel.from_pretrained(
        cfg.base_model_path,
        subfolder="unet",
    ).to(device="cuda")

    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        cfg.base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs={
            "use_motion_module": False,
            "unet_use_temporal_attention": False,
            "use_latent_attention": True,
            "num_hie_latent": 1,
        },
    ).to(device="cuda")


    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_path,
    ).to(dtype=weight_dtype, device="cuda")    
    
    guidance_encoder_group = setup_guidance_encoder(cfg)
    # print(guidance_encoder_group)

    # IP-Adapter
    # attn_procs = {}
    # unet = denoising_unet
    # for name in unet.attn_processors.keys():
    #     cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
    #     if name.startswith("mid_block"):
    #         hidden_size = unet.config.block_out_channels[-1]
    #     elif name.startswith("up_blocks"):
    #         block_id = int(name[len("up_blocks.")])
    #         hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
    #     elif name.startswith("down_blocks"):
    #         block_id = int(name[len("down_blocks.")])
    #         hidden_size = unet.config.block_out_channels[block_id]
    #     if cross_attention_dim is None:
    #         attn_procs[name] = AttnProcessor().to(unet.device, dtype=unet.dtype)
    #     else:
    #         attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, 
    #                                             cross_attention_dim=cross_attention_dim, 
    #                                             num_tokens=[4]).to(unet.device, dtype=unet.dtype)
    # denoising_unet.set_attn_processor(attn_procs)
    # adapter_modules = torch.nn.ModuleList(denoising_unet.attn_processors.values())
    
    image_enc_ip = CLIPVisionModelWithProjection.from_pretrained(
        cfg.image_encoder_ip_path,
    ).to(dtype=weight_dtype, device="cuda") 
    
    image_proj_model = ProjPlusModel(
        cross_attention_dim=denoising_unet.config.cross_attention_dim,
        id_embeddings_dim=512,
        clip_embeddings_dim=1280,
        num_tokens=4,
    ).to(device="cuda")
    
    #NOTE(wz) ip-adapter used for get feature from the generated target face
    image_proj_model_fine = ImageProjModel(
        cross_attention_dim=denoising_unet.config.cross_attention_dim,
        clip_embeddings_dim=1024,
        clip_extra_context_tokens=4,
    ).to(device="cuda") 

    load_stage1_state_dict(
        denoising_unet,
        reference_unet,
        guidance_encoder_group,
        image_proj_model,
        image_proj_model_fine,
        cfg.stage1_ckpt_dir,
        cfg.stage1_ckpt_step,        
    )
    
    # Freeze some modules
    vae.requires_grad_(False)
    image_enc.requires_grad_(False)
    #TODO: denoising_unet True or False?
    denoising_unet.requires_grad_(False)
    reference_unet.requires_grad_(False)
            
    for module in guidance_encoder_group.values():
        module.requires_grad_(False)
    
    image_proj_model_fine.requires_grad_(True)
    image_proj_model.requires_grad_(False)
    image_enc_ip.requires_grad_(False)
    
    # set_requires_grad(denoising_unet, ["attn_latent","attn_cross_latent","norm_latent"])
    set_requires_grad(denoising_unet, ["attn_cross_latent"])


    reference_control_writer = ReferenceAttentionControl(
        reference_unet,
        do_classifier_free_guidance=False,
        mode="write",
        fusion_blocks="full",
    )
    reference_control_reader = ReferenceAttentionControl(
        denoising_unet,
        do_classifier_free_guidance=False,
        mode="read",
        fusion_blocks="full",
    )
    
    model = ChampFaceModel(
        reference_unet,
        denoising_unet,
        reference_control_writer,
        reference_control_reader,
        guidance_encoder_group,
        image_proj_model,
        image_proj_model_fine,
        adapter_modules=None,
        adapter_ckpt_path=None,
    )

    app = FaceAnalysis(name="buffalo_l",root="./pretrained_models", providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    if cfg.solver.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            reference_unet.enable_xformers_memory_efficient_attention()
            denoising_unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    if cfg.solver.gradient_checkpointing:
        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

    if cfg.solver.scale_lr:
        learning_rate = (
            cfg.solver.learning_rate
            * cfg.solver.gradient_accumulation_steps
            * cfg.data.train_bs
            * accelerator.num_processes
        )
    else:
        learning_rate = cfg.solver.learning_rate
        
    if cfg.solver.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps,
    )            
    
    train_dataset = ImageFaceFineDataset(
        video_folder=cfg.data.video_folder,
        image_size=cfg.data.image_size,
        sample_margin=cfg.data.sample_margin,
        data_parts=cfg.data.data_parts,
        guids=cfg.data.guids,
        extra_region=None,
        bbox_crop=cfg.data.bbox_crop,
        bbox_resize_ratio=tuple(cfg.data.bbox_resize_ratio),
        select_face=cfg.data.select_face,
        face_folder=cfg.data.face_folder,
        face_image_size=cfg.data.face_image_size,
        face_guids=cfg.data.face_guids,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.data.train_bs, shuffle=True, num_workers=16
    )
    
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.solver.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(
        cfg.solver.max_train_steps / num_update_steps_per_epoch
    )
    # guidance_encoder_group.update({'guidance_encoder_DWpose':guidance_encoder_group.pop("DWpose")})
    # print(guidance_encoder_group)
    guidance_encoder_group = {
        "guidance_encoder_DWpose":guidance_encoder_group["DWpose"]
    }


    #NOTE:(wz) use this pipeline to acquire target people image and target face image.
    pipeline_inference = MultiGuidanceIPFine2ImagePipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        **guidance_encoder_group,
        image_encoder_ip=image_enc_ip,
        image_proj_model=image_proj_model,
        # fusion_module=fusion_module,
        scheduler=val_noise_scheduler,
    )
    pipeline_inference = pipeline_inference.to(device=accelerator.device)
    clip_image_processor = CLIPImageProcessor()
    
    logger.info("Start training ...")
    logger.info(f"Num Samples: {len(train_dataset)}")
    logger.info(f"Train Batchsize: {cfg.data.train_bs}")
    logger.info(f"Num Epochs: {num_train_epochs}")
    logger.info(f"Total Steps: {cfg.solver.max_train_steps}")
    
    global_step, first_epoch = 0, 0
    
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = f"{cfg.output_dir}/{cfg.exp_name}/checkpoints"
        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        
    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")
    generator = torch.manual_seed(42)

    # Training Loop
    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.
        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                pixel_values = batch["tgt_img"].to(weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents.unsqueeze(2)  # (b, c, 1, h, w)
                    latents = latents * 0.18215
                    
                noise = torch.randn_like(latents)
                if cfg.noise_offset > 0.0:                 
                    noise += cfg.noise_offset * torch.randn(
                        (noise.shape[0], noise.shape[1], 1, 1, 1),
                        device=noise.device,
                    )
                    
                face_pixel_values = batch["tgt_face_img"].to(weight_dtype)
                with torch.no_grad():
                    face_latents = vae.encode(face_pixel_values).latent_dist.sample()
                    face_latents = face_latents.unsqueeze(2)
                    face_latents = face_latents * 0.18215
                    
                face_noise = torch.randn_like(face_latents)
                if cfg.noise_offset > 0.0:                 
                   face_noise += cfg.noise_offset * torch.randn(
                        (face_noise.shape[0], face_noise.shape[1], 1, 1, 1),
                        device=face_noise.device,
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(
                    0,
                    train_noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )   
                timesteps = timesteps.long()
            
                tgt_guid_imgs = batch["tgt_guid"]
                tgt_guid_imgs = tgt_guid_imgs.unsqueeze(2)
                
                tgt_face_guid_imgs = batch["tgt_face_guid"]
                tgt_face_guid_imgs = tgt_face_guid_imgs.unsqueeze(2)

                uncond_fwd = random.random() < cfg.uncond_ratio
                uncond_region_clip_fwd = random.random() < 0.1
              
                if uncond_fwd:
                    clip_img = torch.zeros_like(
                        batch["clip_img"], dtype=image_enc.dtype, device=image_enc.device
                    )
                    id_embeds = torch.zeros_like(
                        batch["face_embedding"], dtype=image_enc_ip.dtype, device=image_enc_ip.device
                    )
                else:
                    clip_img = batch["clip_img"]
                    id_embeds = batch["face_embedding"]
                
                if uncond_region_clip_fwd:
                    region_clip_images = torch.zeros_like(
                        batch["face_clip_img"], dtype=image_enc.dtype, device=image_enc.device
                    )
                else:
                    region_clip_images = batch["face_clip_img"]

                with torch.no_grad():
                    # print("batch[ref_image]")
                    # torch.Size([4, 3, 1024, 1024])
                    

                    ref_image_latents = vae.encode(
                        batch["ref_img"].to(
                        dtype=vae.dtype, device=vae.device
                    )).latent_dist.sample()  # (bs, d, 64, 64)
                    ref_image_latents = ref_image_latents * 0.18215
                    
                   
                    region_image_latents = vae.encode(
                        batch["face_vae_img"].to(
                        dtype=vae.dtype, device=vae.device
                    )).latent_dist.sample()  # (bs, d, 64, 64)
                    region_image_latents = region_image_latents * 0.18215
                    # print("batch[clip_img]")
                    # print(batch["clip_img"].shape)
                    # print("clip-------------")
                    # print(clip_img.shape)
                    # batch[clip_img]
                    # torch.Size([4, 3, 224, 224])
                    # clip-------------
                    # torch.Size([4, 3, 224, 224])

                 
                    clip_image_embeds = image_enc(
                        clip_img.to("cuda", dtype=weight_dtype)
                    ).image_embeds
                    # print("1111111111111")
                    # print(clip_image_embeds.shape)

                    region_image_embeds = image_enc_ip(
                        region_clip_images.to("cuda", dtype=weight_dtype),
                        output_hidden_states=True,
                    ).hidden_states[-2]  # NOTE(ZSH): use the second last hidden states
                    # print("22222222222222222")
                    # print(region_image_embeds.shape)
                    # 1111111111111
                    # torch.Size([4, 768])
                    # 22222222222222222
                    # torch.Size([4, 257, 1280])

                  
                    image_prompt_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)
                  
                    id_embeds = id_embeds.unsqueeze(1).to(
                        dtype=clip_image_embeds.dtype, device=clip_image_embeds.device
                    )
                # print(id_embeds)
                # type(id_embeds)

                noisy_latents = train_noise_scheduler.add_noise(
                    latents, noise, timesteps
                )
                face_noisy_latents = train_noise_scheduler.add_noise(
                    face_latents, face_noise, timesteps
                )

                if train_noise_scheduler.prediction_type == "epsilon":
                    target = noise
                    face_target = face_noise
                elif train_noise_scheduler.prediction_type == "v_prediction":
                    target = train_noise_scheduler.get_velocity(
                        latents, noise, timesteps
                    )
                    face_target = train_noise_scheduler.get_velocity(
                        face_latents, face_noise, timesteps
                    )
                  
                else:
                    raise ValueError(
                        f"Unknown prediction type {train_noise_scheduler.prediction_type}"
                    )

                with torch.autocast("cuda"):
                    #NOTE(wz) happen in training to get blurry target image and target face image.
                    target_images1, target_face_images1 = pipeline_inference(
                        ref_image_latents,
                        region_image_latents,
                        clip_image_embeds,
                        image_prompt_embeds,
                        clip_img,
                        tgt_guid_imgs,
                        tgt_face_guid_imgs,
                        region_image_embeds,
                        region_clip_images,
                        id_embeds,
                        # face_ref_idx,
                        uncond_fwd,
                        write_latent=False,
                        do_latent_attention=False,
                        latent_attn_masks=batch["tgt_attn_mask"],
                        batch_size=cfg.data.train_bs,
                        width=cfg.data.image_size,
                        height=cfg.data.image_size,
                        face_image_size=cfg.data.face_image_size,
                        num_inference_steps=20,
                        guidance_scale=3.5,
                        generator=generator,
                        return_dict=False,
                    )
                
                # #NOTE(wz) happen in training to get blurry target image and target face image.
                # target_images1, target_face_images1 = pipeline_inference(
                #     ref_image_latents,
                #     region_image_latents,
                #     clip_image_embeds,
                #     image_prompt_embeds,
                #     clip_img,
                #     tgt_guid_imgs,
                #     tgt_face_guid_imgs,
                #     region_image_embeds,
                #     region_clip_images,
                #     id_embeds,
                #     # face_ref_idx,
                #     uncond_fwd,
                #     write_latent=False,
                #     do_latent_attention=False,
                #     latent_attn_masks=batch["tgt_attn_mask"],
                #     batch_size=cfg.data.train_bs,
                #     width=cfg.data.image_size,
                #     height=cfg.data.image_size,
                #     face_image_size=cfg.data.face_image_size,
                #     num_inference_steps=20,
                #     guidance_scale=3.5,
                #     generator=generator,
                #     return_dict=False,
                # )
                # print("---------target_image---------")
                # print(target_images1.shape)
                # print(target_face_images1.shape)
                # torch.Size([4, 3, 1, 1024, 1024])
                # torch.Size([4, 3, 1, 256, 256])
                target_images1 = target_images1.squeeze(2)
                target_face_images1 = target_face_images1.squeeze(2)
                # print(target_images1.shape)
                # print(target_face_images1.shape) 

                if uncond_fwd:
                    clip_img = torch.zeros_like(
                        target_images1, dtype=image_enc.dtype, device=image_enc.device
                    )
                else:
                    clip_img = target_images1
                
                if uncond_region_clip_fwd:
                    region_clip_images = torch.zeros_like(
                        target_face_images1, dtype=image_enc.dtype, device=image_enc.device
                    )
                else:
                    region_clip_images = target_face_images1
                # print("clip_img2")
                # print(clip_img.shape)
                # torch.Size([4, 3, 1024, 1024])

                with torch.no_grad():

                    ref_image_latents = vae.encode(
                        target_images1.to(
                        dtype=vae.dtype, device=vae.device
                    )).latent_dist.sample()  # (bs, d, 64, 64)
                    ref_image_latents = ref_image_latents * 0.18215

                    clip_img = clip_image_processor.preprocess(
                        images=clip_img, return_tensors="pt"
                    ).pixel_values

                    clip_image_embeds = image_enc(
                        clip_img.to("cuda", dtype=weight_dtype)
                    ).image_embeds
                    # print("333333333333333")
                    # print(clip_image_embeds.shape)
                    image_prompt_embeds = clip_image_embeds.unsqueeze(1)  # (bs, 1, d)

                    # region_clip_images = clip_image_processor.preprocess(
                    #     region_clip_images.resize((224, 224)), return_tensors="pt"
                    # ).pixel_values

                    region_clip_images = clip_image_processor.preprocess(
                        images=region_clip_images, return_tensors="pt"
                    ).pixel_values

                    # region_image_embeds = image_enc_ip(
                    #     region_clip_images.to("cuda", dtype=weight_dtype),
                    #     output_hidden_states=True,
                    # ).hidden_states[-2]  # NOTE(ZSH): use the second last hidden states


                    region_image_embeds = image_enc_ip(
                        region_clip_images.to("cuda", dtype=weight_dtype),
                        output_hidden_states=True,
                    ).image_embeds
                    region_image_embeds = region_image_embeds.unsqueeze(1)
                    
                    
                    # print("4444444444")
                    # print(region_image_embeds.shape)
                    # 333333333333333                                                                                                                                                 | 0/4 [00:00<?, ?it/s]
                    # torch.Size([4, 768])
                    # 4444444444
                    # torch.Size([4, 1, 1024])

                    # ip_tokens = image_proj_model_fine(region_image_embeds)
                    # print("region_image_embeds---------")
                    # print(region_image_embeds.shape)
                    # print(ip_tokens.shape)
                    # print(image_prompt_embeds.shape)
                    # torch.Size([4, 257, 1280])
                    # torch.Size([1028, 4, 768])
                    # torch.Size([4, 1, 768])
                    # ip_tokens = ip_tokens.transpose(0,1)  # 交换第二与第三维度

                    # encoder_hidden_states = torch.cat([image_prompt_embeds, ip_tokens], dim=1)
                   

                face_ref_idx=1
                # print(noisy_latents.shape)
                # torch.Size([4, 4, 1, 128, 128])

                model_pred = model(
                    noisy_latents,
                    timesteps,
                    ref_image_latents,
                    image_prompt_embeds,
                    region_image_embeds,
                    # ip_tokens,
                    tgt_guid_imgs,
                    face_ref_idx,
                    uncond_fwd,
                    write_latent=False,
                    do_latent_attention=True,
                    latent_attn_masks=batch["tgt_attn_mask"],
                )
                
                if cfg.snr_gamma == 0:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    snr = compute_snr(train_noise_scheduler, timesteps)
                    if train_noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [snr, cfg.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none",
                    )

                    # face region additional loss
                    down_sampled_mask = F.interpolate(
                        batch["tgt_attn_mask"][:, 0:1, ...], size=(model_pred.shape[-2], model_pred.shape[-1]), mode="bilinear")
                    down_sampled_mask = down_sampled_mask.unsqueeze(1).repeat(1, model_pred.shape[1], 1, 1, 1)                      # NOTE(ZSH): IMPORTANT note potential error in broadcast
                    face_model_pred = torch.masked_select(model_pred, down_sampled_mask > 0)
                    face_target = torch.masked_select(target, down_sampled_mask > 0)
                    face_loss = F.mse_loss(
                        face_model_pred.float(), face_target.float(), reduction="none"
                    )
                    face_loss = (
                        face_loss.mean(dim=list(range(1, len(face_loss.shape))))
                        * mse_loss_weights
                    )
                    
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
            
                    loss = loss.mean() + cfg.face_loss_weight * face_loss.mean()
                    
                avg_loss = accelerator.gather(loss.repeat(cfg.data.train_bs)).mean()
                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params,
                        cfg.solver.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Logging
            save_dir = f"{cfg.output_dir}/{cfg.exp_name}"
            if accelerator.sync_gradients:
                reference_control_reader.clear()
                reference_control_writer.clear()
                progress_bar.update(1)
                global_step += 1
                tb_tracker.add_scalar(tag='train loss', scalar_value=train_loss, global_step=global_step)
                tb_tracker.add_scalar(tag='face_train loss', scalar_value=face_loss.mean().detach().item(), global_step=global_step)

                train_loss = 0.0
                #　save checkpoints
                if global_step % cfg.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
                        delete_additional_ckpt(save_dir, 6)
                        accelerator.save_state(save_path)                
                # check data
                if global_step % cfg.checkpointing_steps == 0 or global_step == 1:
                    img_forcheck = batch['tgt_img'] * 0.5 + 0.5
                    ref_forcheck = batch['ref_img'] * 0.5 + 0.5
                    guid_forcheck = list(torch.chunk(batch['tgt_guid'], batch['tgt_guid'].shape[1]//3, dim=1))
                    region_forcheck = batch['face_img']
                    batch_forcheck = torch.cat([ref_forcheck, img_forcheck] + guid_forcheck + [region_forcheck], dim=0)
                    save_image(batch_forcheck, f'{cfg.output_dir}/{cfg.exp_name}/sanity_check/data-{global_step:06d}-rank{accelerator.device.index}.png', nrows=4)
                # log validation                      
                if global_step % cfg.validation.validation_steps == 0 or global_step == 1:
                    if accelerator.is_main_process:
                        
                        sample_dicts = log_validation(
                            cfg=cfg,
                            vae=vae,
                            image_enc=image_enc,
                            image_enc_ip=image_enc_ip,
                            model=model,
                            scheduler=val_noise_scheduler,
                            accelerator=accelerator,
                            width=cfg.data.image_size,
                            height=cfg.data.image_size,
                            face_image_size=cfg.data.face_image_size,
                            face_app=app,
                            seed=cfg.seed
                        )
                        
                        for sample_dict in sample_dicts:
                            sample_name = sample_dict["name"]
                            img = sample_dict["img"]
                            img.save(f"{save_dir}/validation/{global_step:06d}-{sample_name}.png")
                            
            logs = {
                "step_loss": loss.detach().item(),
                "face_loss": cfg.face_loss_weight * face_loss.mean().detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "stage": 1.5,
            }
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps:
                break
            
        # save model after each epoch
        if (
            epoch
        ) % cfg.save_model_epoch_interval == 0 and accelerator.is_main_process:
            unwrap_model = accelerator.unwrap_model(model)
            save_checkpoint(
                unwrap_model.reference_unet,
                f"{save_dir}/saved_models",
                "reference_unet",
                global_step,
                total_limit=None,
            )
            save_checkpoint(
                unwrap_model.denoising_unet,
                f"{save_dir}/saved_models",
                "denoising_unet",
                global_step,
                total_limit=None,
            )
            for guid_type in unwrap_model.guidance_types:
                save_checkpoint(
                    getattr(unwrap_model, f"guidance_encoder_{guid_type}"),
                    f"{save_dir}/saved_models",
                    f"guidance_encoder_{guid_type}",
                    global_step,
                    total_limit=None,
                )
            save_checkpoint(
                unwrap_model.image_proj_model,
                f"{save_dir}/saved_models",
                "image_proj_model",
                global_step,
                total_limit=None,
            )
            save_checkpoint(
                unwrap_model.image_proj_model_fine,
                f"{save_dir}/saved_models",
                "image_proj_model_fine",
                global_step,
                total_limit=None,
            )       
            # save_checkpoint(
            #     unwrap_model.adapter_modules,
            #     f"{save_dir}/saved_models",
            #     "adapter_modules",
            #     global_step,
            #     total_limit=None,
            # )
            # save_checkpoint(
            #     unwrap_model.fusion_module,
            #     f"{save_dir}/saved_models",
            #     "fusion_module",
            #     global_step,
            #     total_limit=None,
            # )

    accelerator.wait_for_everyone()
    accelerator.end_training()                                    
                            
def save_checkpoint(model, save_dir, prefix, ckpt_num, total_limit=None):
    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit is not None:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(save_dir, removing_checkpoint)
                os.remove(removing_checkpoint)

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)           
    
    
if __name__ == "__main__":
    import shutil
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/champ/workspaces/wangzhen/code/champ-train-dev/configs/train/stage1.5_x10.3.yaml")
    args = parser.parse_args()

    if args.config[-5:] == ".yaml":
        config = OmegaConf.load(args.config)
    else:
        raise ValueError("Do not support this format config file")
    
    save_dir = os.path.join(config.output_dir, config.exp_name)
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'sanity_check'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'saved_models'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'validation'), exist_ok=True)
    # save config, script
    shutil.copy(args.config, os.path.join(save_dir, 'sanity_check', f'{config.exp_name}.yaml'))
    shutil.copy(os.path.abspath(__file__), os.path.join(save_dir, 'sanity_check'))
    
    main(config)             
    