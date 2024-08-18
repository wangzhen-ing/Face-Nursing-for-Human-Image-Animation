import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
from tqdm import tqdm
from datasets.data_utils import process_bbox, crop_bbox, mask_to_bbox, mask_to_bkgd
from diffusers.image_processor import IPAdapterMaskProcessor


class VideoDataset(Dataset):
    def __init__(
        self,
        video_folder: str,
        image_size: int = 512,
        sample_frames: int = 24,
        sample_rate: int = 4,
        data_parts: list = ["all"],
        guids: list = ["depth", "normal", "semantic_map", "dwpose"],
        extra_region: list = [],
        bbox_crop: bool = True,
        bbox_resize_ratio: tuple = (0.8, 1.2),
        aug_type: str = "Resize",
        select_face: bool = False,
        face_folder=None,
        face_image_size=256,
        face_guids=["lmk_images"],
    ):
        super().__init__()
        self.video_folder = video_folder
        self.image_size = image_size
        self.sample_frames = sample_frames
        self.sample_rate = sample_rate
        self.data_parts = data_parts
        self.guids = guids
        self.extra_region = extra_region
        self.bbox_crop = bbox_crop
        self.bbox_resize_ratio = bbox_resize_ratio
        self.aug_type = aug_type        
        self.select_face = select_face
        self.face_folder = face_folder if face_folder else video_folder
        self.face_image_size = face_image_size
        self.face_guids = face_guids
        
        # self.data_lst = self.generate_data_lst()
        self.data_lst, self.face_data_lst = self.generate_data_lst()
        
        self.clip_image_processor = CLIPImageProcessor()
        self.pixel_transform, self.guid_transform = self.setup_transform()
        self.ipadapter_mask_processor = IPAdapterMaskProcessor()

        self.face_pixel_transform = transforms.Compose(
            [
                    transforms.Resize((self.face_image_size, self.face_image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.face_guid_transform = transforms.Compose(
            [
                    transforms.Resize((self.face_image_size, self.face_image_size)),
                    transforms.ToTensor(),
            ]
        )
        self.face_ref_transform = transforms.Compose(
            [
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
            ]
        )
        
    # def generate_data_lst(self):
    #     video_folder = Path(self.video_folder)
    #     if "all" in self.data_parts:
    #         data_parts = sorted(video_folder.iterdir())
    #     else:
    #         data_parts = [(video_folder / p) for p in self.data_parts]
    #     data_lst = []
    #     for data_part in data_parts:
    #         for video_dir in tqdm(sorted(data_part.iterdir())):
    #             if self.is_valid(video_dir):
    #                 data_lst += [video_dir]
    #     return data_lst


    def generate_data_lst(self):
        video_folder = Path(self.video_folder)
        face_video_folder = Path(self.face_folder)
        if "all" in self.data_parts:
            data_parts = sorted(video_folder.iterdir())
        else:
            data_parts = [(video_folder / p) for p in self.data_parts]
        data_lst = []
        face_data_lst = []
        for data_part in data_parts:
            for video_dir in tqdm(sorted(data_part.iterdir())):
                if self.is_valid(video_dir):
                    part_name = data_part.name
                    video_dir_name = video_dir.name
                    face_video_dir = face_video_folder / part_name / video_dir_name
                    length = len(list((video_dir / "images").glob("*.png")))
                    if face_video_dir.exists() and self.is_face_valid(face_video_dir, length):
                        data_lst += [video_dir]
                        face_data_lst += [face_video_dir]

        print(f"Finish loading {len(data_lst)} data, {len(face_data_lst)} face data")
        return data_lst, face_data_lst
    
    # def is_valid(self, video_dir: Path):
    #     video_length = len(list((video_dir / "images").iterdir()))
    #     for guid in self.guids:
    #         guid_length = len(list((video_dir / guid).iterdir()))
    #         if guid_length != video_length:
    #             return False
    #     if self.select_face:
    #         if not (video_dir / "face_images").is_dir():
    #             return False
    #         else:
    #             face_img_length = len(list((video_dir / "face_images").iterdir()))
    #             if face_img_length == 0:
    #                 return False
    #     return True

    def is_valid(self, video_dir: Path):
        video_length = len(list((video_dir / "images").glob("*.png")))
        for guid in self.guids:
            if not (video_dir / guid).exists():
                return False
            guid_length = len(list((video_dir / guid).glob("*.png")))
            if guid_length == 0 or guid_length != video_length:
                return False
        if self.select_face:
            if not (video_dir / "face_images_app").is_dir():
                return False
            else:
                face_img_length = len(list((video_dir / "face_images_app").glob("*.png")))
                if face_img_length < 5:
                    return False
        return True

    def is_face_valid(self, video_dir: Path, length):
        # video_length = len(list((video_dir / "face_images").glob("*.png")))
        for guid in self.face_guids:
            if not (video_dir / guid).exists():
                return False
            guid_length = len(list((video_dir / guid).glob("*.png")))
            if guid_length == 0 or guid_length != length:
                return False
        return True
    
    def resize_long_edge(self, img):
        img_W, img_H = img.size
        long_edge = max(img_W, img_H)
        scale = self.image_size / long_edge
        new_W, new_H = int(img_W * scale), int(img_H * scale)
        
        img = F.resize(img, (new_H, new_W))
        return img

    def padding_short_edge(self, img):
        img_W, img_H = img.size
        width, height = self.image_size, self.image_size
        padding_left = (width - img_W) // 2
        padding_right = width - img_W - padding_left
        padding_top = (height - img_H) // 2
        padding_bottom = height - img_H - padding_top
        
        img = F.pad(img, (padding_left, padding_top, padding_right, padding_bottom), 0, "constant")
        return img
    
    def setup_transform(self):
        if self.bbox_crop:
            if self.aug_type == "Resize":
                pixel_transform = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                guid_transform = transforms.Compose([
                    transforms.Resize((self.image_size, self.image_size)),
                    transforms.ToTensor(),
                ])
                
            elif self.aug_type == "Padding":
                pixel_transform = transforms.Compose([
                    transforms.Lambda(self.resize_long_edge),
                    transforms.Lambda(self.padding_short_edge),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ])
                guid_transform = transforms.Compose([
                    transforms.Lambda(self.resize_long_edge),
                    transforms.Lambda(self.padding_short_edge),
                    transforms.ToTensor(),
                ])
            else:
                raise NotImplementedError("Do not support this augmentation")
        
        else:
            pixel_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=self.image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            guid_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=self.image_size, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
                transforms.ToTensor(),
            ])
        
        return pixel_transform, guid_transform
    
    def set_clip_idx(self, video_length):
        clip_length = min(video_length, (self.sample_frames - 1) * self.sample_rate + 1)
        start_idx = random.randint(0, video_length - clip_length)
        clip_idxes = np.linspace(
            start_idx, start_idx + clip_length - 1, self.sample_frames, dtype=int
        ).tolist()
        return clip_idxes
    
    def get_mean_bbox(self, clip_idxes, bboxes):
        clip_bbox_lst = []
        for c_idx in clip_idxes:
            clip_bbox = bboxes[c_idx]
            clip_bbox_lst.append(np.array(clip_bbox))
        clip_bbox_mean = np.stack(clip_bbox_lst, axis=0).mean(0, keepdims=False)
        return clip_bbox_mean
        
    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, list):
            ret_lst = []
            for img in images:
                if isinstance(img, list):
                    transformed_sub_images = [transform(sub_img) for sub_img in img]
                    sub_ret_tensor = torch.cat(transformed_sub_images, dim=0)  # (c*n, h, w)
                    ret_lst.append(sub_ret_tensor)
                else:
                    transformed_images = transform(img)
                    ret_lst.append(transformed_images)  # (c*1, h, w)
            ret_tensor = torch.stack(ret_lst, dim=0)  # (f, c*n, h, w)     
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor
    
    def __len__(self):
        return len(self.data_lst)
    
    def __getitem__(self, idx):
        video_dir = self.data_lst[idx]
        face_video_dir = self.face_data_lst[idx]
        
        # reference image
        img_dir = video_dir / "images"
        img_face_dir = face_video_dir / "face_images"
        if self.select_face:
            face_img_dir = video_dir / "face_images_app"
            face_img_lst = [img.name for img in face_img_dir.glob("*.png")]
            ref_img_name = random.choice(face_img_lst)
        else:
            ref_img_name = random.choice([img.name for img in img_dir.glob("*.png")])
        ref_img_path = img_dir / ref_img_name
        ref_img_pil = Image.open(ref_img_path)

        face_meta_path = video_dir / "face_info" / (ref_img_path.stem + ".json")
        with open(str(face_meta_path), "r") as fp:
            face_meta = json.load(fp)
        
        face_embedding = torch.from_numpy(np.array(face_meta["normed_embedding"]))
        
        # tgt frames indexes
        video_length = len(list(img_dir.iterdir()))
        clip_idxes = self.set_clip_idx(video_length)
        
        # calculate bbox first
        if self.bbox_crop:
            human_bbox_json_path = video_dir / "human_bbox.json"
            with open(human_bbox_json_path) as bbox_fp:
                human_bboxes = json.load(bbox_fp)
            mean_bbox = self.get_mean_bbox(clip_idxes, human_bboxes)
            resize_scale = random.uniform(*self.bbox_resize_ratio)
            ref_W, ref_H = ref_img_pil.size
            tgt_bbox = process_bbox(mean_bbox, ref_H, ref_W, resize_scale)
        
        img_path_lst = sorted([img.name for img in img_dir.glob("*.png")])
        tgt_vidpil_lst = []
        tgt_guid_vidpil_lst = []

        tgt_face_vidpil_lst = []
        tgt_face_mask_vidpil_lst = []
        tgt_face_guid_vidpil_lst = []
        
        # tgt frames
        # guid frames: [[frame0: n_type x pil], [frame1: n x pil], [frame2: n x pil], ...]
        for c_idx in clip_idxes:
            tgt_img_path = img_dir / img_path_lst[c_idx]
            tgt_img_pil = Image.open(tgt_img_path)
            tgt_img_pil = crop_bbox(tgt_img_pil, tgt_bbox)
            tgt_vidpil_lst.append(tgt_img_pil)

            tgt_face_img_path = img_face_dir / img_path_lst[c_idx]
            tgt_face_img_pil = Image.open(tgt_face_img_path)
            # tgt_face_img_pil = Image.fromarray(np.array(tgt_face_img_pil))
            tgt_face_vidpil_lst.append(tgt_face_img_pil)

            tgt_face_mask_path = face_video_dir / "face_masks" / img_path_lst[c_idx]
            tgt_face_mask_pil = Image.open(tgt_face_mask_path).convert("L")
            tgt_face_mask_vidpil_lst.append(tgt_face_mask_pil)
            
            tgt_img_name = tgt_img_path.name
            tgt_guid_pil_lst = []
            tgt_face_guid_pil_lst = []
            for guid in self.guids:
                guid_img_path = video_dir / guid / tgt_img_name
                if guid == "semantic_map":
                    mask_img_path = video_dir / "mask" / tgt_img_name            
                    guid_img_pil = mask_to_bkgd(guid_img_path, mask_img_path)
                else:
                    guid_img_pil = Image.open(guid_img_path).convert("RGB")
                if self.bbox_crop:
                    guid_img_pil = crop_bbox(guid_img_pil, tgt_bbox)
                tgt_guid_pil_lst.append(guid_img_pil)
            tgt_guid_vidpil_lst.append(tgt_guid_pil_lst)

            for face_guid in self.face_guids:
                face_guid_img_path = face_video_dir / face_guid / tgt_img_name
                face_guid_img_pil = Image.open(face_guid_img_path).convert("RGB")
                tgt_face_guid_pil_lst += [face_guid_img_pil]
            tgt_face_guid_vidpil_lst.append(tgt_face_guid_pil_lst)

        
        ref_img_idx = img_path_lst.index(ref_img_name)
        if self.bbox_crop:
            ref_bbox = process_bbox(human_bboxes[ref_img_idx], ref_H, ref_W, resize_scale)
            ref_img_pil = crop_bbox(ref_img_pil, ref_bbox)
        
        state = torch.get_rng_state()
        tgt_vid = self.augmentation(tgt_vidpil_lst, self.pixel_transform, state)
        tgt_guid_vid = self.augmentation(tgt_guid_vidpil_lst, self.guid_transform, state)
        ref_img_vae = self.augmentation(ref_img_pil, self.pixel_transform, state)
        clip_img = self.clip_image_processor(
            images=ref_img_pil, return_tensor="pt"
        ).pixel_values[0]


        tgt_face_mask = self.augmentation(tgt_face_mask_vidpil_lst, self.guid_transform, state)
        tgt_attn_mask = torch.stack([tgt_face_mask], dim=0).squeeze(1)
        
        # NOTE(ZSH): Face image and Adapter mask; Not support bbox crop now! Face images are in BGR mode!
        # face images
        assert self.select_face
        face_img_pil = Image.open(face_img_dir / ref_img_name)
        face_img_vae_pil = Image.fromarray(np.array(face_img_pil)[..., ::-1])
        face_clip_img = self.clip_image_processor(
            images=face_img_pil, return_tensor="pt",
        ).pixel_values[0]
        face_vae_img = self.augmentation(face_img_vae_pil, self.face_ref_transform, state)
        tgt_face_vid = self.augmentation(tgt_face_vidpil_lst, self.face_pixel_transform, state)
        tgt_face_guid_vid = self.augmentation(tgt_face_guid_vidpil_lst, self.face_guid_transform, state)


        face_img = transforms.ToTensor()(transforms.Resize((self.image_size, self.image_size))(face_img_pil))
        sample = dict(
            tgt_vid=tgt_vid,
            tgt_guid_vid=tgt_guid_vid,
            tgt_face_vid=tgt_face_vid,
            tgt_face_guid_vid=tgt_face_guid_vid,
            tgt_attn_mask=tgt_attn_mask,
            ref_img=ref_img_vae,
            clip_img=clip_img,
            face_embedding=face_embedding,
            face_clip_img=face_clip_img,
            face_vae_img=face_vae_img,
            face_img=face_img,
            ref_img_path=str(ref_img_path),
        )
        
        return sample
    
