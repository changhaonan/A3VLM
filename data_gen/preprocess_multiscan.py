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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi_scan_dir", type=str, default="/home/harvey/Data/multi_scan/output")
    parser.add_argument("--multi_scan_art_file", type=str, default="/home/harvey/Data/multi_scan_art/articulated_dataset/articulated_objects.train.h5")
    parser.add_argument("--data_id", type=str, default="scene_00010_01")
    args = parser.parse_args()

    multi_scan_dir = args.multi_scan_dir
    multi_scan_art_file = args.multi_scan_art_file
    data_id = args.data_id
    preprocess_multiscan(multi_scan_dir, multi_scan_art_file, data_id)
