import numpy as np
import base64
import io
import PIL.Image
from math import pi,atan2,asin
import cv2

def img_data_to_pil(img_data):
    f = io.BytesIO()
    f.write(img_data)
    img_pil = PIL.Image.open(f)
    return img_pil

def img_data_to_arr(img_data):
    img_pil = img_data_to_pil(img_data)
    img_arr = np.array(img_pil)
    return img_arr

def img_b64_to_arr(img_b64):
    img_data = base64.b64decode(img_b64)
    img_arr = img_data_to_arr(img_data)
    return img_arr

def in_bbox(rect,pts):
    logic = (rect[0] < pts[:,0]) & (pts[:,0]< rect[2]) & (rect[1] < pts[:,1]) & (pts[:,1] < rect[3])
    return logic

def in_mask(mask,pts):
    pts = pts.astype(int)
    pts_mask = np.zeros_like(mask)
    pts_mask[pts[:,1],pts[:,0]] = 1
    logic = pts_mask & mask
    return logic

def mask_iou(mask1, mask2, eps=1e-7):
    """
    Calculate masks IoU.

    Args:
        mask1 (torch.Tensor): A tensor of shape (N, n) where N is the number of ground truth objects and n is the
                        product of image width and height.
        mask2 (torch.Tensor): A tensor of shape (M, n) where M is the number of predicted objects and n is the
                        product of image width and height.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): A tensor of shape (N, M) representing masks IoU.
    """
    mask1 = mask1.reshape(mask1.shape[0],-1).astype(float)
    mask2 = mask2.reshape(mask2.shape[0],-1).astype(float)
    intersection = np.clip(np.matmul(mask1, mask2.T),0,None)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)

