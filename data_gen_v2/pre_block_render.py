"""Prepare for block rendering. The main goal is to compute static bbox.
Generate meta information.
"""

import tqdm
import trimesh
import numpy as np
import os
import json
from urchin import URDF
from block_object_render import generate_robot_cfg, generate_robot_mesh


def pre_block_render(data_dir, output_dir, on_cat_list, under_cat_list, block_list):
    data_names = os.listdir(data_dir)
    data_names = [name for name in data_names if name.isdigit()]
    meta_infos = {
        "on": [],
        "under": [],
        "other": [],
        "all": [],
        "meta": {},
    }
    rng = np.random.RandomState(0)
    for data_name in tqdm.tqdm(data_names):
        # Try loading first
        data_file = f"{data_dir}/{data_name}/mobility.urdf"
        try:
            robot = URDF.load(data_file)
        except Exception as e:
            print(f"Error in loading {data_file}: {e}")
            continue

        joint_cfg, link_cfg = generate_robot_cfg(robot, rng, is_bg_obj=True)
        robot_link_mesh_map = {}
        for link, link_pose in robot.link_fk(cfg=joint_cfg).items():
            link_mesh = link.collision_mesh
            link_name = link.name
            if link_mesh is not None:
                robot_link_mesh_map[link_mesh] = (link_pose, link_name)
                link_mesh.apply_transform(link_pose)
        # Merged mesh
        robot_mesh_list = []
        for mesh, (pose, name) in robot_link_mesh_map.items():
            # mesh.apply_transform(pose)
            robot_mesh_list.append(mesh)
        robot_mesh = trimesh.util.concatenate(robot_mesh_list)
        robot_bbox = robot_mesh.bounds
        meta_infos["meta"][data_name] = {
            "bbox": robot_bbox.tolist(),
        }
        # Make stats for only valid data
        meta_file = os.path.join(data_dir, data_name, "meta.json")
        with open(meta_file, "r") as f:
            meta_info = json.load(f)
        model_cat = meta_info["model_cat"]
        if model_cat in block_list:
            continue
        elif model_cat in on_cat_list:
            meta_infos["on"].append(data_name)
        elif model_cat in under_cat_list:
            meta_infos["under"].append(data_name)
        else:
            meta_infos["other"].append(data_name)
        assert (
            (model_cat in on_cat_list)
            or (model_cat in under_cat_list)
            or (model_cat in block_list)
            or (model_cat in other_list)
        ), f"Unknown model category: {model_cat}"
        meta_infos["all"].append(data_name)  # All valid data
        
    # Export meta info
    meta_file = os.path.join(data_dir, "meta.json")
    with open(meta_file, "w") as f:
        json.dump(meta_infos, f, indent=4)


if __name__ == "__main__":
    data_dir = "/home/harvey/Data/partnet-mobility-v0/dataset"
    output_dir = "/home/harvey/Data/partnet-mobility-v0/output_v2"
    os.makedirs(output_dir, exist_ok=True)

    on_cat_list = [
        "Safe",
        "Display",
        "Laptop",
        "Lighter",
        "Microwave",
        "Mouse",
        "Box",
        "Bottle",
        "Bucket",
        "Camera",
        "CoffeeMachine",
        "Display",
        "TrashCan",
        "KitchenPot",
        "Pliers",
        "Remote",
        "Bottle",
        "Toaster",
        "Lamp",
        "Dispenser",
        "Scissors",
        "Stapler",
        "Kettle",
        "Keyboard",
        "USB",
        "Faucet",
        "Phone",
        "Eyeglasses",
        "Clock",
        "Globe",
        "Knife",
        "Oven",
        "Pen",
        "Printer",
        "Refrigerator",
        "Switch",
    ]
    under_cat_list = ["StorageFurniture", "Table", "Dishwasher"]
    other_list = [
        "Door",
        "Suitcase",
        "FoldingChair",
        "Toilet",
        "WashingMachine",
        "Cart",
        "Chair",
        "Fan",
        "Window",
    ]
    block_list = [
        "Phone",
        "Keyboard",
        "Remote",
    ]
    pre_block_render(data_dir, output_dir, on_cat_list, under_cat_list, block_list)
