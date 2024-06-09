import os
import csv
import json
import random
import numpy as np

render_result_path = "/mnt/petrelfs/huangsiyuan/data/partnet_pyrender_960_v6_fix_angle"
texture_prompts_path = "/mnt/petrelfs/huangsiyuan/data/texture_prompts"

import numpy as np

palette = np.asarray(
    [
        [0, 0, 0],
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ]
)


def load_all_texture_prompts(texture_prompts_path):
    cato_texture_prompts = {}
    all_texture_prompts = os.listdir(texture_prompts_path)
    all_texture_prompts = [texture_prompt for texture_prompt in all_texture_prompts if texture_prompt.endswith(".text")]
    print(f"Texture prompts length: {len(all_texture_prompts)}")
    for texture_prompt in all_texture_prompts:
        with open(os.path.join(texture_prompts_path, texture_prompt), "r") as f:
            texture_prompt_content = f.readlines()  # the list
        cato = texture_prompt.split(".")[0]
        cato_texture_prompts[cato.lower()] = texture_prompt_content

    return cato_texture_prompts


def padding_prompts(texture_prompts, image_num):
    if len(texture_prompts) < image_num:
        texture_prompts = texture_prompts * (image_num // len(texture_prompts) + 1)
    else:
        texture_prompts = random.sample(texture_prompts, image_num)
    return texture_prompts


def construct_dataset_index(idx, render_result_path, cato_texture_prompts):
    dataset_tasks = []
    meta_json = os.path.join(render_result_path, str(idx), "meta.json")
    meta_info = json.load(open(meta_json, "r"))
    model_cat = meta_info["model_cat"]
    texture_descriptions = cato_texture_prompts[model_cat.lower()]

    raw_img_path = os.path.join(render_result_path, str(idx), "raw_images")
    img_list = os.listdir(raw_img_path)
    img_list = [img for img in img_list if img.endswith(".png")]
    img_list = sorted(img_list, key=lambda x: int(x.split(".")[0]))
    texture_descriptions = padding_prompts(texture_descriptions, len(img_list))

    depth_img_path = os.path.join(render_result_path, str(idx), "depth_images")
    depth_img_list = os.listdir(depth_img_path)
    depth_img_list = [img for img in depth_img_list if img.endswith(".png")]
    depth_img_list = sorted(depth_img_list, key=lambda x: int(x.split(".")[0]))

    mask_img_path = os.path.join(render_result_path, str(idx), "mask")
    mask_img_list = os.listdir(mask_img_path)
    mask_img_list = [img for img in mask_img_list if img.endswith(".png")]
    mask_img_list = sorted(mask_img_list, key=lambda x: int(x.split(".")[0]))

    assert len(img_list) == len(depth_img_list), f"Image list length: {len(img_list)}, Depth image list length: {len(depth_img_list)}"

    for i in range(len(img_list)):
        img_name = img_list[i]
        img_index = int(img_name.split(".")[0])
        depth_img_name = depth_img_list[i]
        mask_img_name = mask_img_list[i]
        img_index_depth = int(depth_img_name.split(".")[0])
        assert img_index == img_index_depth, f"Image index: {img_index}, Depth image index: {img_index_depth}"
        img_path = os.path.join(raw_img_path, img_name)
        depth_img_item_path = os.path.join(depth_img_path, depth_img_name)
        mask_img_item_path = os.path.join(mask_img_path, mask_img_name)

        img_info = {
            "dataset_id": idx,
            "img_index": img_index,
            "img_name": img_name,
            "img_path": img_path,
            "mask_img_path": mask_img_item_path,
            "depth_img_path": depth_img_item_path,
            "texture_description": texture_descriptions[i].replace("\n", ""),
            "category": model_cat,
        }

        dataset_tasks.append(img_info)

    return dataset_tasks


def construct_dataset_csv(render_result_path, texture_prompts_path):
    all_dataset_ids = os.listdir(render_result_path)
    all_dataset_ids = [int(dataset_id) for dataset_id in all_dataset_ids if dataset_id.isdigit() and os.path.exists(os.path.join(render_result_path, dataset_id, "annotations.json"))]
    print(f"Dataset length: {len(all_dataset_ids)}")

    cato_texture_prompts = load_all_texture_prompts(texture_prompts_path)

    all_dataset_info = []
    for idx in all_dataset_ids:
        all_dataset_info.extend(construct_dataset_index(idx, render_result_path, cato_texture_prompts))

    print(f"Total image-level dataset length: {len(all_dataset_info)}")

    # save to json
    with open("partnet_pyrender_dataset_v5.json", "w") as f:
        json.dump(all_dataset_info, f)

    # save to csv
    with open("partnet_pyrender_dataset_v5.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(all_dataset_info[0].keys())
        for dataset_info in all_dataset_info:
            writer.writerow(dataset_info.values())


def construct_data_index_classname(render_result_path):
    all_dataset_ids = os.listdir(render_result_path)
    all_dataset_ids = [int(dataset_id) for dataset_id in all_dataset_ids if dataset_id.isdigit() and os.path.exists(os.path.join(render_result_path, dataset_id, "annotations.json"))]
    print(f"Dataset length: {len(all_dataset_ids)}")

    class_name_index = {}
    for idx in all_dataset_ids:
        meta_json = os.path.join(render_result_path, str(idx), "meta.json")
        meta_info = json.load(open(meta_json, "r"))
        model_cat = meta_info["model_cat"]
        if model_cat not in class_name_index:
            class_name_index[model_cat] = []
            class_name_index[model_cat].append(idx)
        else:
            class_name_index[model_cat].append(idx)

    with open("partnet_pyrender_dataset_v4_classname.json", "w") as f:
        json.dump(class_name_index, f)

    print(class_name_index.keys())


if __name__ == "__main__":
    construct_dataset_csv(render_result_path=render_result_path, texture_prompts_path=texture_prompts_path)
    # construct_data_index_classname(render_result_path=render_result_path)
