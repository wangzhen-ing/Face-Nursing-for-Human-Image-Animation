import concurrent.futures
import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import sys
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from dwpose import DWposeDetector
import json
import cv2
from tqdm import tqdm

def process_single_image(image_path, detector, face_app):
    img_name = Path(image_path).name
    root_dir = Path(image_path).parent.parent
    save_dir = root_dir.joinpath('DWpose')
    out_path = save_dir.joinpath(img_name)
    # if os.path.exists(out_path):
    #     return

    dirs = ["DWpose", "dwpose_meta", "face_info", "face_images_app"]
    for dir in dirs:
        path = root_dir.joinpath(dir)
        path.mkdir(parents=True, exist_ok=True)

    frame_pil = Image.open(image_path)
    dwpose, _, extracted_data, candidates, subset = detector(
        frame_pil, image_resolution=min(*(frame_pil.size))
    )
    dwpose = dwpose.resize(frame_pil.size)
    dwpose.save(out_path)
    print(f'save dwpose to {out_path}')
    dwpose_meta = {
        "coords": candidates.tolist(),
        "scores": subset.tolist(),
    }
    meta_path = root_dir / "dwpose_meta" / (image_path.stem + ".json")
    with open(str(meta_path), "w") as meta_fp:
        json.dump(dwpose_meta, meta_fp, indent=4)

    id_image = cv2.imread(str(image_path))
    faces = app.get(id_image)
    
    if len(faces) > 0:
        face_info = faces[0]
        face_info_path = root_dir / "face_info" / (image_path.stem + ".json")
        face_info_dict = {
            "normed_embedding": face_info.normed_embedding.tolist(),
            "pose": face_info["pose"].tolist(),
            "kps": face_info["kps"].tolist(),
            "bbox": face_info["bbox"].tolist(),
            "gender": face_info["gender"].item(),
            "landmark_3d_68": face_info["landmark_3d_68"].tolist(),
            "landmark_2d_106": face_info["landmark_2d_106"].tolist(),
            "embedding": face_info["embedding"].tolist(),
            "embedding_norm": face_info.embedding_norm.item(),
        }
        with open(str(face_info_path), "w") as fp:
            json.dump(face_info_dict, fp, indent=4)
        face_image = face_align.norm_crop(id_image, landmark=face_info.kps, image_size=224)
        face_image_pil = Image.fromarray(face_image)
        face_image_path = root_dir / "face_images_app" / (image_path.stem + ".png")
        face_image_pil.save(face_image_path)

def process_batch_images(image_list, detector, face_app):
    for i, image_path in enumerate(image_list):
        print(f"Process {i + 1}/{len(image_list)} image")
        process_single_image(image_path, detector, face_app)
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=str, default='out_03rmpdwjx')
    parser.add_argument("-j", type=int, default=4, help="Num workers")
    args = parser.parse_args()

    num_workers = args.j

    root_dir = '/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/champ/datasets/new_dataset'
    part = args.part
    # collect all image paths
    # image_paths = [os.path.join(root, name)
    #                for root, dirs, files in os.walk(args.image_root)
    #                for name in files if name.endswith((".jpg", ".png"))]
    image_paths = []
    part = [part]
    if part == ["all"]:
        part = os.listdir(root_dir)
    # 遍历数据集根目录下的所有子目录
    for p in sorted(part):
        for subdir in tqdm(Path(os.path.join(root_dir, p)).iterdir()):
            if subdir.is_dir():  # 确保是目录
                # 遍历子目录下的所有文件
                # for cond_dir in subdir.iterdir():
                #     if cond_dir.name == 'images':
                cond_dir = subdir / "images"
                output_dir = subdir / "DWpose"
                for image_path in cond_dir.iterdir():
                    if image_path.is_file() and image_path.suffix in ['.jpg', '.png', '.jpeg']:# and not (output_dir / image_path.name).exists():  # 确保是文件且为图片
                        image_paths.append(image_path)
    # detector = DWposeDetector().to('cuda:0')
    # process_single_image(image_paths[0], detector)
    batch_size = (len(image_paths) + num_workers - 1) // num_workers
    image_chunks = [
        image_paths[i: i + batch_size]
        for i in range(0, len(image_paths), batch_size)
    ]
    # cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    # gpu_ids = [int(id) for id in range(len(cuda_visible_devices.split(",")))]
    gpu_ids = list(range(torch.cuda.device_count()))
    print(f"avaliable gpu ids: {gpu_ids}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Initialize DWposeDetector per thread due to potential GPU memory issues
        futures = []
        for i, chunk in enumerate(image_chunks):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            detector = DWposeDetector()
            detector = detector.to(f"cuda:{gpu_id}")
            app = FaceAnalysis(name="buffalo_l",root="/cpfs01/projects-HDD/cfff-6f3a36a0cd1e_HDD/zsy_43187/champ/workspaces/zhushenhao/code/champ-train-dev/pretrained_models")
            app.prepare(ctx_id=0, det_size=(640, 640))
            futures.append(
                executor.submit(
                    process_batch_images, chunk, detector, app
                )
            )
        # futures = [executor.submit(process_batch_images, image_paths[i::num_workers], DWposeDetector().to(f"cuda:{i % torch.cuda.device_count()}"))
        #            for i in range(num_workers)]
            
        for future in concurrent.futures.as_completed(futures):
            future.result()
    print('finished')


