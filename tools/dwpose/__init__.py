# https://github.com/IDEA-Research/DWPose
# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import copy
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import torch
from controlnet_aux.util import HWC3, resize_image
from PIL import Image

from . import util
from .wholebody import Wholebody


def draw_pose(pose, H, W):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    canvas = util.draw_facepose(canvas, faces)
    # canvas = util.draw_facepose_instantid(canvas, faces)

    return canvas
    
def crop_region(img, coords, expansion_factor=1.0):
    # coords: (n, 2)
    # ensure half of the points are available
    if (coords > 0).sum() < coords.size // 2:
        return None, None
    
    coords_x = coords[..., 0][coords[..., 0] >= 0]
    coords_y = coords[..., 1][coords[..., 1] >= 0]
    x_min = np.min(coords_x)
    x_max = np.max(coords_x)
    y_min = np.min(coords_y)
    y_max = np.max(coords_y)

    w = x_max - x_min
    h = y_max - y_min

    dx = (w * expansion_factor - w) / 2
    dy = (h * expansion_factor - h) / 2

    # Apply expansion
    x_min = max(x_min - dx, 0)
    x_max = min(x_max + dx, 1)
    y_min = max(y_min - dy, 0)
    y_max = min(y_max + dy, 1)
    
    H, W, C = img.shape
    
    x_min = int(x_min*W)
    x_max = int(x_max*W)
    y_min = int(y_min*H)
    y_max = int(y_max*H)

    def handle_equal(v_min, v_max):
        if v_min == v_max:
            if v_min >= 1:
                v_min -= 1
            else:
                v_max += 1
        return v_min, v_max

    x_min, x_max = handle_equal(x_min, x_max)
    y_min, y_max = handle_equal(y_min, y_max)

    cropped_img = img[y_min: y_max, x_min: x_max, :]   
    cropped_mask = [x_min, y_min, x_max, y_max]

    return cropped_img[..., ::-1], cropped_mask


class DWposeDetector:
    def __init__(self):
        pass

    def to(self, device):
        self.pose_estimation = Wholebody(device)
        return self

    def cal_height(self, input_image):
        input_image = cv2.cvtColor(
            np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR
        )

        input_image = HWC3(input_image)
        H, W, C = input_image.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(input_image)
            nums, keys, locs = candidate.shape
            # candidate[..., 0] /= float(W)
            # candidate[..., 1] /= float(H)
            body = candidate
        return body[0, ..., 1].min(), body[..., 1].max() - body[..., 1].min()

    def __call__(
        self,
        input_image,
        detect_resolution=512,
        image_resolution=512,
        output_type="pil",
        **kwargs,
    ):
        input_image = cv2.cvtColor(
            np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR
        )

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        H, W, C = input_image.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(input_image)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            score = subset[:, :18]
            max_ind = np.mean(score, axis=-1).argmax(axis=0)
            score = score[[max_ind]]
            body = candidate[:, :18].copy()
            body = body[[max_ind]]
            nums = 1
            body = body.reshape(nums * 18, locs)
            body_score = copy.deepcopy(score)
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18 * i + j)
                    else:
                        score[i][j] = -1

            un_visible = subset < 0.3
            un_face = subset < 0.75
            un_hand = subset < 0.75
            face_candidate = copy.deepcopy(candidate)
            hand_candidate = copy.deepcopy(candidate)
            candidate[un_visible] = -1
            face_candidate[un_face] = -1
            hand_candidate[un_hand] = -1 

            foot = candidate[:, 18:24]

            faces = candidate[[max_ind], 24:92]

            hands = candidate[[max_ind], 92:113]
            hands = np.vstack([hands, candidate[[max_ind], 113:]])

            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            detected_map = draw_pose(pose, H, W)
            detected_map = HWC3(detected_map)

            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape

            detected_map = cv2.resize(
                detected_map, (W, H), interpolation=cv2.INTER_LINEAR
            )

            if output_type == "pil":
                detected_map = Image.fromarray(detected_map)

            face_crop = face_candidate[[max_ind], 24:92][0]
            hand_crop_left = hand_candidate[[max_ind], 92:113][0]
            hand_crop_right = hand_candidate[[max_ind], 113:][0]
            
            face_img, face_mask = crop_region(img, face_crop, expansion_factor=2)
            hand_img_left, hand_mask_left = crop_region(img, hand_crop_left, expansion_factor=2)
            hand_img_right, hand_mask_right = crop_region(img, hand_crop_right, expansion_factor=2)
            
            if output_type == "pil":
                face_img = Image.fromarray(face_img) if face_img is not None else None
                hand_img_left = Image.fromarray(hand_img_left) if hand_img_left is not None else None
                hand_img_right = Image.fromarray(hand_img_right) if hand_img_right is not None else None

            extracted_data = dict(
                face_images=face_img, face_masks=face_mask,
                hand_left_images=hand_img_left, hand_left_masks=hand_mask_left,
                hand_right_images=hand_img_right, hand_right_masks=hand_mask_right,
            )

            candidate_res = candidate[max_ind, :]
            candidate_res[..., 0] *= float(W)
            candidate_res[..., 1] *= float(H)
            
            return detected_map, body_score, extracted_data, candidate_res, subset.T
