from typing import Tuple
from PIL import Image
import torchvision.transforms as transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import os
import numpy as np

class PadToSquare:
    def __init__(self, background_color:Tuple[float, float, float]):
        """
        pad an image to squre (borrowed from LLAVA, thx)
        :param background_color: rgb values for padded pixels, normalized to [0, 1]
        """
        self.bg_color = tuple(int(x*255) for x in background_color)

    def __call__(self, img: Image.Image):
        width, height = img.size
        if width == height:
            return img
        elif width > height:
            result = Image.new(img.mode, (width, width), self.bg_color)
            result.paste(img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(img.mode, (height, height), self.bg_color)
            result.paste(img, ((height - width) // 2, 0))
            return result

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + f"(bg_color={self.bg_color})"
        return format_string


def T_random_resized_crop(size=224):
    t = transforms.Compose([
        transforms.RandomResizedCrop(size=(size, size), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=BICUBIC,
                                     antialias=None),  # 3 is bicubic
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    return t


def T_resized_center_crop(size=224):
    t = transforms.Compose([
        transforms.Resize(
            size, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    return t


def T_padded_resize(size=224):
    t = transforms.Compose([
        PadToSquare(background_color=(0.48145466, 0.4578275, 0.40821073)),
        transforms.Resize(
            size, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    return t


def get_transform(transform_type: str, size=224):
    if transform_type == "random_resized_crop":
        transform = T_random_resized_crop(size)
    elif transform_type == "resized_center_crop":
        transform = T_resized_center_crop(size)
    elif transform_type == "padded_resize":
        transform = T_padded_resize(size)
    else:
        raise ValueError("unknown transform type: transform_type")
    return transform


def pc_norm(pc):
    """ pc: NxC, return NxC """
    xyz = pc[:, :3]
    rgb_feature = pc[:, 3:]

    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    xyz = xyz / m
    
    if np.max(rgb_feature) > 1:
        # Normalize RGB values to [0, 1]
        rgb_feature = rgb_feature / 255.0
        
    rgb_feature[rgb_feature < 0] = 0
    rgb_feature[rgb_feature > 1] = 1

    pc = np.concatenate((xyz, rgb_feature), axis=1)
    return pc

def load_objaverse_point_cloud(filename, pointnum=8192, use_color=True):
    """
    Load point cloud from file, return Nx6 array
    """
    point_cloud = np.load(filename)
    
    # if 9 dims, extract 6 dims
    if point_cloud.shape[1] == 9:
        point_cloud = point_cloud[:, [0, 1, 2, 6, 7, 8]]

    # * normalize
    # point_cloud = pc_norm(point_cloud)
    # normalize the rgb channel
    if np.max(point_cloud[:, 3:]) > 1:
        point_cloud[:, 3:] = point_cloud[:, 3:] / 255.0
    
    if point_cloud.shape[0] < pointnum:
        duplicate_num = pointnum // point_cloud.shape[0] + 1
        point_cloud = np.tile(point_cloud, (duplicate_num, 1))[:pointnum]
    
    if point_cloud.shape[0] > pointnum:
        choice = np.random.choice(point_cloud.shape[0], pointnum, replace=False)
        point_cloud = point_cloud[choice, :]

    if not use_color:
        point_cloud = point_cloud[:, :3]
        assert point_cloud.shape[1] == 3
    else:
         assert point_cloud.shape[1] == 6

    return point_cloud

def load_partnet_point_cloud_ply(filename, pointnum=8192, use_color=True):
    pass

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point