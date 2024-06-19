"""Preprocess the multiscan data, majorly parsing the .mp4 file into images."""

import h5py
import os
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from utils import AxisBBox3D


def preprocess_multiscan(multi_scan_dir, multi_scan_art_file, data_id):
    # Parse the .mp4 file into images
    video_file = os.path.join(multi_scan_dir, data_id, data_id + ".mp4")
    video_dir = os.path.join(multi_scan_dir, data_id, "video")
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    command = f"ffmpeg -i {video_file} {video_dir}/%04d.png"
    print(command)
    os.system(command)


if __name__ == "__main__":
    multi_scan_dir = "/home/harvey/Data/multi_scan/output"
    multi_scan_art_file = "/home/harvey/Data/multi_scan_art/articulated_dataset/articulated_objects.train.h5"
    data_id = "scene_00000_01"
    preprocess_multiscan(multi_scan_dir, multi_scan_art_file, data_id)
