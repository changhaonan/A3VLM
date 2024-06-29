"""Generate mesh for a given configuration."""

import numpy as np
from packaging import version

# Check NumPy version
if version.parse(np.__version__) >= version.parse("1.20.0"):
    # Create an alias for np.int
    np.int = int
    np.float = float

import argparse
from urdfpy import URDF
import trimesh
import time
import os
import json
import tqdm
import cv2
import copy
from render_tools import render_parts

from multiprocessing import Pool, cpu_count
from functools import partial


def compute_kinematic_level(robot):
    kinematic_dict = {}
    kinematic_dict["base"] = 0
    acutated_joint_names = [joint.name for joint in robot.actuated_joints]
    joint_list = copy.deepcopy(robot.joints)
    for i in range(len(joint_list)):
        for joint in joint_list:
            if joint.parent in kinematic_dict and joint.child not in kinematic_dict:
                if joint.name in acutated_joint_names and not (joint.parent.endswith("helper")):
                    kinematic_dict[joint.child] = kinematic_dict[joint.parent] + 1
                else:
                    kinematic_dict[joint.child] = kinematic_dict[joint.parent]
                joint_list.remove(joint)
    kinematic_level = max(kinematic_dict.values())
    return kinematic_level


def render_data_item_with_idx(
    data_name, data_dir, output_dir, num_poses, camera_info, cam_radius_max, cam_radius_min, num_joint_value, only_front=False, load_camera_pose=False, load_joint_value=False
):
    data_file = f"{data_dir}/{data_name}/mobility.urdf"
    output_dir = f"{output_dir}/{data_name}"
    os.makedirs(output_dir, exist_ok=True)
    image_dir = os.path.join(output_dir, "raw_images")
    os.makedirs(image_dir, exist_ok=True)
    depth_dir = os.path.join(output_dir, "depth_images")
    os.makedirs(depth_dir, exist_ok=True)
    real_depth_dir = os.path.join(output_dir, "real_depth_images")
    os.makedirs(real_depth_dir, exist_ok=True)
    mask_dir = os.path.join(output_dir, "mask")
    os.makedirs(mask_dir, exist_ok=True)
    info = {}
    annotations = []

    info["camera_info"] = camera_info
    meta_file = f"{data_dir}/{data_name}/meta.json"
    with open(meta_file, "r") as f:
        meta = json.load(f)
    info["model_cat"] = meta["model_cat"]

    if load_camera_pose:
        existing_info_file = f"{output_dir}/info.json"
        with open(existing_info_file, "r") as f:
            existing_info = json.load(f)
        predefine_camera_poses = np.array(existing_info["camera_poses"])
    else:
        predefine_camera_poses = None

    if load_joint_value:
        existing_info_file = f"{output_dir}/info.json"
        with open(existing_info_file, "r") as f:
            existing_info = json.load(f)

    try:
        for i in range(num_joint_value):
            # Load URDF
            robot = URDF.load(data_file)
            # Compute joint kinematic level
            kinematic_level = compute_kinematic_level(robot)

            # Actuation links
            actuaion_joints = robot.actuated_joints
            joint_cfg = {}
            link_cfg = {}
            for joint in actuaion_joints:
                # joint_value_sample = np.random.rand()
                joint_value_sample = 0.5
                if joint.limit is not None:
                    limit_low = joint.limit.lower
                    limit_high = joint.limit.upper
                    if not load_joint_value:
                        if joint_value_sample < 0.1:
                            joint_value = limit_low
                        elif joint_value_sample > 0.9:
                            joint_value = limit_high
                        else:
                            joint_value = joint_value_sample * (limit_high - limit_low) + limit_low
                    else:
                        joint_value_normalized = existing_info[joint.child][i * num_poses]
                        joint_value = joint_value_normalized * (limit_high - limit_low) + limit_low
                    if kinematic_level > 1:
                        # Existing hierarchical joint, disable the joint
                        joint_value = 0.0
                    joint_cfg[joint.name] = joint_value
                    link_cfg[joint.child] = (joint_value - limit_low) / (limit_high - limit_low + 1e-6)  # normalize

            # Save current robot mesh as obj
            robot_link_mesh_map = {}
            for link, link_pose in robot.link_fk(cfg=joint_cfg).items():
                link_mesh = link.collision_mesh
                link_name = link.name
                if link_mesh is not None:
                    robot_link_mesh_map[link_mesh] = (link_pose, link_name)
            robot_visual_map = {}
            for mesh, pose in robot.visual_trimesh_fk(cfg=joint_cfg).items():
                robot_visual_map[mesh] = (pose, "visual")

            # Render parts
            # 1. Render on link level
            _annotations, camera_poses, _, depth_imgs, mask_imgs = render_parts(
                robot_link_mesh_map,
                num_poses,
                camera_info,
                cam_radius_max,
                cam_radius_min,
                image_idx_offset=i * num_poses,
                only_front=only_front,
                camera_sample_method="xy",
                predefine_camera_poses=predefine_camera_poses,
            )
            # 2. Render on visual level
            _, _, color_imgs, _, _ = render_parts(
                robot_visual_map,
                num_poses,
                camera_info,
                cam_radius_max,
                cam_radius_min,
                image_idx_offset=0,
                only_front=only_front,
                camera_sample_method="xy",
                predefine_camera_poses=camera_poses,
                is_link_map=False,
            )
            robot_mesh_list = []
            for mesh, (pose, name) in robot_link_mesh_map.items():
                mesh.apply_transform(pose)
                robot_mesh_list.append(mesh)
            robot_mesh = trimesh.util.concatenate(robot_mesh_list)

            # Swap y and z axis
            transform = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
            robot_mesh.apply_transform(transform)

            # Save mesh
            output_path = f"{output_dir}/mesh_{i}.obj"
            robot_mesh.export(output_path)

            # Save joint value
            for link_name in link_cfg:
                if link_name not in info:
                    info[link_name] = []
                info[link_name] += [link_cfg[link_name]] * num_poses

            # Save camera poses
            if "camera_poses" not in info:
                info["camera_poses"] = []
            info["camera_poses"] += camera_poses

            # Save annotations
            annotations += _annotations

            # Save color images
            for j, img in enumerate(color_imgs):
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_path = f"{image_dir}/{i * num_poses + j:06d}.png"
                cv2.imwrite(img_path, img)

            # Save depth images
            for j, depth_img in enumerate(depth_imgs):
                # Normalize depth image
                zero_mask = depth_img == 0
                depth_min = np.min(depth_img[~zero_mask])
                depth_max = np.max(depth_img[~zero_mask])
                norm_depth_img = (depth_img - depth_min) / (depth_max - depth_min + 1e-6)
                norm_depth_img[zero_mask] = 0
                norm_depth_img[~zero_mask] = 0.9 * (1 - norm_depth_img[~zero_mask]) + 0.1
                norm_depth_img_path = f"{depth_dir}/{i * num_poses + j:06d}.png"
                cv2.imwrite(norm_depth_img_path, (norm_depth_img * 255).astype(np.uint8))
                depth_img_path = f"{real_depth_dir}/{i * num_poses + j:06d}.png"
                cv2.imwrite(depth_img_path, (depth_img * 1000).astype(np.uint16))

            # Save mask images
            for j, img in enumerate(mask_imgs):
                img_path = f"{mask_dir}/{i * num_poses + j:06d}.png"
                cv2.imwrite(img_path, img)

    except Exception as e:
        print(f"Error in {data_name}: {e}")
        return False

    # Save poses info
    if (not load_camera_pose) and (not load_joint_value):
        with open(f"{output_dir}/info.json", "w") as f:
            json.dump(info, f)

    # Save annotations
    with open(f"{output_dir}/annotations.json", "w") as f:
        json.dump(annotations, f)

    # Copy other files
    os.system(f"cp {data_dir}/{data_name}/semantics.txt {output_dir}/semantics.txt")
    os.system(f"cp {data_dir}/{data_name}/mobility.urdf {output_dir}/mobility.urdf")
    os.system(f"cp {data_dir}/{data_name}/meta.json {output_dir}/meta.json")
    os.system(f"cp {data_dir}/{data_name}/mobility_v2.json {output_dir}/mobility_v2.json")
    return True


if __name__ == "__main__":
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=0, help="Random seed")
    argparser.add_argument("--version", type=int, default=0, help="Version of the dataset")
    argparser.add_argument("--data_dir", type=str, default="test_data")
    argparser.add_argument("--data_name", type=str, default="all")
    argparser.add_argument("--output_dir", type=str, default="output")
    argparser.add_argument("--num_poses", type=int, default=5)
    argparser.add_argument("--light_radius_min", type=float, default=3.0)
    argparser.add_argument("--light_radius_max", type=float, default=5.0)
    argparser.add_argument("--cam_radius_min", type=float, default=2.5)
    argparser.add_argument("--cam_radius_max", type=float, default=3.0)
    argparser.add_argument("--img_width", type=int, default=960)
    argparser.add_argument("--img_height", type=int, default=960)
    argparser.add_argument("--intrinsic", type=str, default="1000,1000,480,480")
    argparser.add_argument("--only_front", action="store_true", help="Only render the front view")
    argparser.add_argument("--num_joint_value", type=int, default=8, help="Generate num_joint_value different joint values")
    argparser.add_argument("--load_camera_pose", action="store_true", help="Load camera poses from file")
    argparser.add_argument("--load_joint_value", action="store_true", help="Load joint values from file")
    args = argparser.parse_args()

    # Set parameters for rendering
    np.random.seed(args.seed)

    cam_radius_max = args.cam_radius_max
    cam_radius_min = args.cam_radius_min
    intrinsic = np.array([float(x) for x in args.intrinsic.split(",")])
    camera_info = {"fx": intrinsic[0], "fy": intrinsic[1], "cx": intrinsic[2], "cy": intrinsic[3], "width": args.img_width, "height": args.img_height}
    assert camera_info["cx"] == 0.5 * camera_info["width"], "cx should be 0.5 * width"
    assert camera_info["cy"] == 0.5 * camera_info["height"], "cy should be 0.5 * height"
    assert camera_info["fx"] > camera_info["width"], "fx should be larger than width"
    assert camera_info["fy"] > camera_info["height"], "fy should be larger than height"

    data_dir = args.data_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_name = args.data_name
    num_poses = args.num_poses
    num_joint_value = args.num_joint_value
    load_camera_pose = args.load_camera_pose
    load_joint_value = args.load_joint_value
    # only_front = args.only_front
    # load_camera_pose = True
    # load_joint_value = True
    only_front = True

    if data_name is not None or data_name is not "all":
        start_time = time.time()
        render_data_item_with_idx(data_name, data_dir, output_dir, num_poses, camera_info, cam_radius_max, cam_radius_min, num_joint_value, only_front, load_camera_pose, load_joint_value)
        end_time = time.time()
        print(f"Rendering {data_name} takes {end_time - start_time:.2f}s")
    else:
        data_names = os.listdir(data_dir)
        data_names = [x for x in data_names if os.path.isdir(os.path.join(data_dir, x))]
        # skip the ones with "raw_imgs" folder existing
        data_names_ = []
        for data_name in data_names:
            if not os.path.exists(os.path.join(data_dir, data_name, "raw_images")):
                data_names_.append(data_name)
            elif len(os.listdir(os.path.join(data_dir, data_name, "raw_images"))) < num_poses * num_joint_value:
                data_names_.append(data_name)
        data_names = data_names_

        print(f"Rendering {len(data_names)} datasets")
        # the data indexing specified
        render_function = partial(
            render_data_item_with_idx,
            data_dir=data_dir,
            output_dir=output_dir,
            num_poses=num_poses,
            camera_info=camera_info,
            cam_radius_max=cam_radius_max,
            cam_radius_min=cam_radius_min,
            num_joint_value=num_joint_value,
            only_front=only_front,
            load_camera_pose=load_camera_pose,
            load_joint_value=load_joint_value,
        )

        workers = min(cpu_count(), len(data_names))
        with Pool(cpu_count()) as p:
            status = list(tqdm.tqdm(p.imap(render_function, data_names), total=len(data_names)))

        for i, data_name in enumerate(data_names):
            if not status[i]:
                print(f"Error in {data_name}")
                continue
            # print(f"Finish {data_name}")
