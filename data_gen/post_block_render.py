"""Post-block rendering. Replacing the background and etc."""

import imageio
import os
import cv2
import numpy as np


def replace_background(data_dir, data_id, video_length):
    # Load data
    data_path = os.path.join(data_dir, data_id)
    datas = os.listdir(data_path)

    data_dict = {}
    color_datas = [data for data in datas if "color_imgs" in data]
    depth_datas = [data for data in datas if "depth_imgs" in data]
    mask_datas = [data for data in datas if "mask_imgs" in data]

    for color_data in color_datas:
        joint_select_idx = int(color_data.split("_")[-1].split(".")[0])
        if joint_select_idx not in data_dict:
            data_dict[joint_select_idx] = {"bg": []}
        color_imgs = np.load(os.path.join(data_path, color_data))["images"]
        data_dict[joint_select_idx]["color"] = color_imgs

    for depth_data in depth_datas:
        joint_select_idx = int(depth_data.split("_")[-1].split(".")[0])
        if joint_select_idx not in data_dict:
            data_dict[joint_select_idx] = {"bg": []}
        depth_imgs = np.load(os.path.join(data_path, depth_data))["images"]
        data_dict[joint_select_idx]["depth"] = depth_imgs

    for mask_data in mask_datas:
        joint_select_idx = int(mask_data.split("_")[-1].split(".")[0])
        if joint_select_idx not in data_dict:
            data_dict[joint_select_idx] = {"bg": []}
        mask_imgs = np.load(os.path.join(data_path, mask_data))["images"]
        data_dict[joint_select_idx]["mask"] = mask_imgs

    video_datas = os.listdir(os.path.join(data_path, "video"))
    bg_datas = [data for data in video_datas if "bg_sd" in data]
    bg_datas = sorted(bg_datas, key=lambda x: int(x.split("_")[3].split(".")[0]))
    for bg_data in bg_datas:
        joint_select_idx = int(bg_data.split("_")[2].split(".")[0])
        if joint_select_idx not in data_dict:
            data_dict[joint_select_idx] = {"bg": []}
        bg_image = cv2.imread(os.path.join(data_path, "video", bg_data))
        data_dict[joint_select_idx]["bg"].append(bg_image)
    # Generate video
    for joint_select_idx, data in data_dict.items():
        color_imgs = data["color"]
        depth_imgs = data["depth"]
        mask_imgs = data["mask"]
        bg_imgs = data["bg"]

        num_videos = len(bg_imgs)
        # Replace background
        for video_idx in range(num_videos):
            color_new_imgs = []
            for frame_idx in range(video_length):
                color_img = color_imgs[video_idx * video_length + frame_idx]
                depth_img = depth_imgs[video_idx * video_length + frame_idx]
                mask_img = mask_imgs[video_idx * video_length + frame_idx]
                bg_img = bg_imgs[video_idx]
                bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)
                bg_img = cv2.resize(bg_img, (color_img.shape[1], color_img.shape[0]))
                color_new = np.where(mask_img[..., None] == 0, bg_img, color_img)
                color_new_imgs.append(color_new)
            imageio.mimsave(
                os.path.join(
                    data_path, f"video/color_{joint_select_idx}_{video_idx}.gif"
                ),
                color_new_imgs,
                duration=0.1,
            )


if __name__ == "__main__":
    output_dir = "/home/harvey/Data/partnet-mobility-v0/output_v2"
    data_id = "3596"
    video_length = 20
    replace_background(output_dir, data_id, video_length)
