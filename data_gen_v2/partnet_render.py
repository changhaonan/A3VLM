"""Do object render as building blocks."""

import shutil
import imageio
import json
import os
import cv2
import copy
import pyrender
import trimesh
import numpy as np
from urchin import URDF
from matplotlib import pyplot as plt
from data_utils import AxisBBox3D, get_arrow, check_annotations_3d, check_annotations_2d
from tqdm import tqdm
from partnet_modifier import limit_modifier
import multiprocessing
import logging

logger = logging.getLogger(__name__)


################################ Utility functions ################################
def get_pointcloud(
    color,
    depth,
    mask,
    intrinsic,
    sample_size=-1,
    flip_x: bool = False,
    flip_y: bool = False,
):
    """Get pointcloud from perspective depth image and color image.

    Args:
      color: HxWx3 uint8 array of RGB images.
      depth: HxW float array of perspective depth in meters.
      mask: HxW uint8 array of mask images.
      intrinsics: 3x3 float array of camera intrinsics matrix.

    Returns:
      points: HxWx3 float array of 3D points in camera coordinates.
    """

    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    if flip_x:
        px = width - 1 - px
    if flip_y:
        py = height - 1 - py
    px = (px - intrinsic[0, 2]) * (depth / intrinsic[0, 0])
    py = (py - intrinsic[1, 2]) * (depth / intrinsic[1, 1])
    # Stack the coordinates and reshape
    points = np.float32([px, py, depth]).transpose(1, 2, 0).reshape(-1, 3)

    # Assuming color image is in the format height x width x 3 (RGB)
    # Reshape color image to align with points
    colors = color.reshape(-1, 3)
    masks = mask.reshape(-1, 1)
    pcolors = np.hstack((points, colors, masks))
    pcolors = pcolors[pcolors[:, 0] != 0.0, :]
    if pcolors.shape[0] == 0:
        return None, 0

    points = pcolors[:, :3]
    colors = pcolors[:, 3:6]
    masks = pcolors[:, 6]

    # Sample points
    if points.shape[0] > sample_size and sample_size > 0:
        num_points = points.shape[0]
        indices = np.random.choice(num_points, sample_size, replace=False)
        points = points[indices]
        colors = colors[indices]
        masks = masks[indices]

    # return pcd_with_color
    return points, colors, masks


def sample_camera_pose(
    cam_radius_min: float,
    cam_radius_max: float,
    look_at: np.ndarray,
    up: np.ndarray,
    only_front: bool = False,
    rng: np.random.RandomState = np.random.RandomState(0),
):
    # Sample a radius within the given range
    radius = rng.uniform(cam_radius_min, cam_radius_max)

    # Sample spherical coordinates
    theta = rng.uniform(0, 2 * np.pi)
    phi = rng.uniform(0, np.pi)

    # Convert spherical coordinates to Cartesian coordinates for the camera position
    if only_front:
        x = -np.abs(radius * np.sin(phi) * np.cos(theta)) + look_at[0]
        y = np.abs(radius * np.sin(phi) * np.sin(theta)) + look_at[1]
        z = np.abs(radius * np.cos(phi)) + look_at[2]
    else:
        x = radius * np.sin(phi) * np.cos(theta) + look_at[0]
        y = radius * np.sin(phi) * np.sin(theta) + look_at[1]
        z = radius * np.cos(phi) + look_at[2]

    # Calculate camera position
    cam_position = look_at + np.array([x, y, z])

    # Calculate camera orientation vectors
    z_axis = -(look_at - cam_position)
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # Construct the camera-to-world transformation matrix
    camera_to_world_matrix = np.eye(4)
    camera_to_world_matrix[0:3, 0] = x_axis
    camera_to_world_matrix[0:3, 1] = y_axis
    camera_to_world_matrix[0:3, 2] = z_axis
    camera_to_world_matrix[0:3, 3] = cam_position

    return camera_to_world_matrix


def sample_camera_pose_xy(
    cam_radius_min: float,
    cam_radius_max: float,
    look_at: np.ndarray,
    up: np.ndarray,
    only_front: bool = False,
    rng: np.random.RandomState = np.random.RandomState(0),
):
    """Sample a camera pose given the look-at point and up vector at x-y plane."""
    # Sample a radius within the given range
    radius = rng.uniform(cam_radius_min, cam_radius_max)

    # Convert spherical coordinates to Cartesian coordinates for the camera position
    if only_front:
        theta = rng.uniform(np.pi * 0.6, np.pi * 1.4)
        phi = rng.uniform(0.23 * np.pi, 0.26 * np.pi)
        x = radius * np.cos(theta) * np.cos(phi) + look_at[0]
        y = radius * np.sin(theta) * np.cos(phi) + look_at[1]
        z = radius * np.sin(phi) + look_at[2]
    else:
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(-0.25 * np.pi, 0.25 * np.pi)
        x = radius * np.cos(theta) * np.cos(phi) + look_at[0]
        y = radius * np.sin(theta) * np.cos(phi) + look_at[1]
        z = radius * np.sin(phi) + look_at[2]

    # Calculate camera position
    cam_position = look_at + np.array([x, y, z])

    # Calculate camera orientation vectors
    z_axis = -(look_at - cam_position)
    z_axis = z_axis / np.linalg.norm(z_axis)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # Construct the camera-to-world transformation matrix
    camera_to_world_matrix = np.eye(4)
    camera_to_world_matrix[0:3, 0] = x_axis
    camera_to_world_matrix[0:3, 1] = y_axis
    camera_to_world_matrix[0:3, 2] = z_axis
    camera_to_world_matrix[0:3, 3] = cam_position

    return camera_to_world_matrix


def compute_kinematic_level(robot):
    kinematic_dict = {}
    kinematic_dict["base"] = 0
    acutated_joint_names = [joint.name for joint in robot.actuated_joints]
    joint_list = copy.deepcopy(robot.joints)
    for i in range(len(joint_list)):
        for joint in joint_list:
            if joint.parent in kinematic_dict and joint.child not in kinematic_dict:
                if joint.name in acutated_joint_names and not (
                    joint.parent.endswith("helper")
                ):
                    kinematic_dict[joint.child] = kinematic_dict[joint.parent] + 1
                else:
                    kinematic_dict[joint.child] = kinematic_dict[joint.parent]
                joint_list.remove(joint)
    kinematic_level = max(kinematic_dict.values())
    return kinematic_level


def generate_robot_cfg(
    robot: URDF, rng: np.random.RandomState, is_bg_obj=False, **kwargs
):
    # Compute joint kinematic level
    kinematic_level = compute_kinematic_level(robot)

    # Parameter for video
    video_mode = kwargs.get("video_mode", False)
    joint_value_idx = kwargs.get("joint_value_idx", 0)
    joint_speed = kwargs.get("joint_speed", 0.0)
    joint_select_idx = kwargs.get("joint_select_idx", 0)

    actuaion_joints = robot.actuated_joints
    joint_cfg = {}  # Save in joint's name, raw value
    link_cfg = {}  # Save in link's name, normalized
    for joint_idx, joint in enumerate(actuaion_joints):
        if not video_mode:
            joint_value_sample = rng.rand() if not is_bg_obj else 0.0
        else:
            # In video mode, we want to have a continous motion
            if joint_idx != (joint_select_idx % len(actuaion_joints)) or is_bg_obj:
                # Not the selected joint, keep the previous value
                joint_value_sample = 0.0
            else:
                # The selected joint, move the joint
                if joint_speed > 0:
                    # -pi/2 ~ pi/2
                    joint_value_sample = (
                        np.sin(-np.pi / 2 + joint_speed * joint_value_idx * np.pi) / 2
                        + 0.5
                    )
                else:
                    # pi/2 ~ -pi/2
                    joint_value_sample = (
                        np.sin(np.pi / 2 - joint_speed * joint_value_idx * np.pi) / 2
                        + 0.5
                    )
        if joint.limit is None:
            joint_limit = [-np.pi / 2, np.pi / 2]
        else:
            joint_limit = [joint.limit.lower, joint.limit.upper]
        limit_low, limit_high = limit_modifier(
            kwargs.get("model_cat", None),
            joint.joint_type,
            [joint_limit[0], joint_limit[1]],
        )
        joint_value = joint_value_sample * (limit_high - limit_low) + limit_low
        if kinematic_level > 1:
            # Existing hierarchical joint, disable the joint
            joint_value = 0.0
        joint_cfg[joint.name] = joint_value
        link_cfg[joint.child] = (joint_value - limit_low) / (
            limit_high - limit_low + 1e-6
        )  # normalize
    return joint_cfg, link_cfg


def generate_block_updown_setup(
    data_id,
    on_list,
    under_list,
    other_list,
    meta_info,
    solo_prob,
    rng=np.random.RandomState(0),
):
    """
    On list is the list of objects that are placed upon other objects.
    Under list is the list of objects that are placed under other objects.
    Return [goal_bbox_list, align_direction_list]
    """
    goal_bbox_list = []
    align_direction_list = []
    data_ids = []
    if data_id in other_list or rng.rand() < solo_prob:
        # If the object is not in the on_list or under_list, it is a standalone object
        goal_bbox_list.append(np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]))
        align_direction_list.append(np.array([0.0, 0.0, 1.0]))
        data_ids.append(data_id)
    elif data_id in on_list:
        on_bbox_list = [
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.6]],
            [[-0.5, 0.0, 0.0], [0.0, 0.5, 0.6]],
            [[0.0, -0.5, 0.0], [0.5, 0.0, 0.6]],
            [[-0.5, -0.5, 0.0], [0.0, 0.0, 0.6]],
        ]
        on_bbox_idxs = rng.choice(len(on_bbox_list), 2, replace=False)
        # First on
        data_ids.append(data_id)
        goal_bbox_list.append(np.array(on_bbox_list[on_bbox_idxs[0]]))
        align_direction_list.append(np.array([0.0, 0.0, -1.0]))
        under_data_id = rng.choice(under_list)
        obj_bbox = np.array(meta_info["meta"][under_data_id]["bbox"])
        obj_size = obj_bbox[1] - obj_bbox[0]
        scale_factor = np.max(1.2 / obj_size)  # Support expand a little more than 1.0
        data_ids.append(under_data_id)
        goal_bbox = scale_factor * obj_bbox
        goal_bbox[:, 2] -= np.max(goal_bbox[:, 2])
        goal_bbox_list.append(goal_bbox)
        align_direction_list.append(np.array([0.0, 0.0, 1.0]))
    elif data_id in under_list:
        data_ids.append(data_id)
        obj_bbox = np.array(meta_info["meta"][data_id]["bbox"])
        goal_bbox = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 0.0]])
        obj_size = obj_bbox[1] - obj_bbox[0]
        goal_size = goal_bbox[1] - goal_bbox[0]
        scale_factors = goal_size / obj_size
        scale_factor = np.min(scale_factors)
        scaled_obj_size = obj_size * scale_factor
        goal_bbox_list.append(goal_bbox)
        align_direction_list.append(np.array([0.0, 0.0, 1.0]))
        on_bbox_list = np.array(
            [
                [[0.0, 0.0, 0.0], [0.8, 0.8, 0.3]],
                [[-0.8, 0.0, 0.0], [0.0, 0.8, 0.3]],
                [[0.0, -0.8, 0.0], [0.8, 0.0, 0.3]],
                [[-0.8, -0.8, 0.0], [0.0, 0.0, 0.3]],
            ]
        )
        on_bbox_list[:, :, 0] = on_bbox_list[:, :, 0] * scaled_obj_size[0] / 2
        on_bbox_list[:, :, 1] = on_bbox_list[:, :, 1] * scaled_obj_size[1] / 2
        # First on
        on_bbox_idxs = rng.choice(len(on_bbox_list), 2, replace=False)
        on_data_id = rng.choice(on_list)
        data_ids.append(on_data_id)
        goal_bbox_list.append(np.array(on_bbox_list[on_bbox_idxs[0]]))
        align_direction_list.append(np.array([0.0, 0.0, -1.0]))
    else:
        raise ValueError(f"Unknown object category: {data_id}")
    return data_ids, goal_bbox_list, align_direction_list


def generate_robot_mesh(
    data_id, robot: URDF, joint_cfg: dict, goal_bbox, align_dir, is_bg_obj=False
):
    """Generate trimesh for different part."""
    # Save current robot mesh as obj
    robot_link_mesh_map = {}
    for link, link_pose in robot.link_fk(cfg=joint_cfg).items():
        link_mesh = link.collision_mesh
        link_name = link.name if not is_bg_obj else "bg"
        if link_mesh is not None:
            robot_link_mesh_map[link_mesh] = (link_pose, link_name)
            link_mesh.apply_transform(link_pose)
    robot_visual_map = {}
    for mesh, pose in robot.visual_trimesh_fk(cfg=joint_cfg).items():
        name = "visual" if not is_bg_obj else "bg"
        robot_visual_map[mesh] = (pose, name)
        mesh.apply_transform(pose)

    # Merged mesh
    robot_mesh_list = []
    for mesh, (pose, name) in robot_link_mesh_map.items():
        # mesh.apply_transform(pose)
        robot_mesh_list.append(mesh)
    robot_mesh = trimesh.util.concatenate(robot_mesh_list)
    robot_bbox = robot_mesh.bounds
    # # # # [DEBUG]: Trimesh Visualize
    # scene = trimesh.Scene()
    # scene.add_geometry(robot_mesh)
    # axis = trimesh.creation.axis(origin_size=0.1)  # You can adjust the size as needed
    # scene.add_geometry(axis)
    # scene.show()
    return robot_link_mesh_map, robot_visual_map, robot_bbox


def generate_robot_meshes(
    data_dir,
    data_ids,
    goal_bbox_list,
    align_direction_list,
    meta_info,
    rng=np.random.RandomState(0),
    keep_ratio=True,
    **kwargs,
):
    robot_link_mesh_map = {}
    robot_visual_map = {}
    obj_transforms = []
    for idx, (data_id, goal_bbox, align_direction) in enumerate(
        zip(data_ids, goal_bbox_list, align_direction_list)
    ):
        is_bg_obj = (
            True if idx > 0 else False
        )  # Only the first object is the target object
        data_file = f"{data_dir}/{data_id}/mobility.urdf"
        try:
            robot = URDF.load(data_file)
        except Exception as e:
            print(f"Error in loading {data_file}: {e}")
            if is_bg_obj:
                continue
            else:
                return None, None, None
        joint_cfg, link_cfg = generate_robot_cfg(robot, rng, is_bg_obj, **kwargs)
        _robot_link_mesh_map, _robot_visual_map, robot_bbox_real = generate_robot_mesh(
            data_id, robot, joint_cfg, goal_bbox, align_direction, is_bg_obj
        )

        # Compute bbox & transform
        robot_bbox_static = np.array(meta_info["meta"][data_id]["bbox"])
        transform = compute_bbox_transform(
            robot_bbox_static, goal_bbox, align_direction, keep_ratio, is_bg_obj
        )
        for mesh, (pose, name) in _robot_link_mesh_map.items():
            mesh.apply_transform(transform)
        for mesh, (pose, name) in _robot_visual_map.items():
            mesh.apply_transform(transform)

        robot_link_mesh_map.update(_robot_link_mesh_map)
        robot_visual_map.update(_robot_visual_map)
        obj_transforms.append(transform)
    # # # [DEBUG]
    # articulation_info = read_articulation(data_dir, data_ids[0])
    # scene = trimesh.Scene()
    # for mesh, (pose, name) in robot_link_mesh_map.items():
    #     scene.add_geometry(mesh)
    # for id, _info in articulation_info.items():
    #     axis = trimesh.creation.axis(origin_size=0.1)
    #     axis_origin = _info["axis_origin"]
    #     axis_origin = np.array([-axis_origin[2], -axis_origin[0], axis_origin[1]])
    #     transform = obj_transforms[0]
    #     axis_origin = transform[:3, :3] @ axis_origin + transform[:3, 3]
    #     print(f"V0: {id}: {axis_origin}")
    #     axis.apply_translation(axis_origin)
    #     scene.add_geometry(axis)
    # # Add a sphere at [-0.5, 0.0, 0.0]
    # sphere = trimesh.creation.uv_sphere(radius=0.05)
    # sphere.apply_translation([-0.5, 0.0, 0.0])
    # scene.add_geometry(sphere)
    # scene.show()

    return robot_link_mesh_map, robot_visual_map, obj_transforms


def compute_bbox_transform(
    init_bbox, goal_bbox, align_direction, keep_ratio=True, is_bg_obj=False
):
    """Bg object and foreground object has different emphasis."""
    # Compute dimensions
    obj_min = np.min(init_bbox, axis=0)
    obj_max = np.max(init_bbox, axis=0)
    obj_size = obj_max - obj_min
    robot_center = (obj_min + obj_max) / 2
    goal_min = np.min(goal_bbox, axis=0)
    goal_max = np.max(goal_bbox, axis=0)
    goal_size = goal_max - goal_min
    goal_center = (goal_min + goal_max) / 2

    # Compute scale factors
    scale_factors = goal_size / obj_size
    scaling_matrix = np.eye(4)
    if not keep_ratio:
        scaling_matrix[0, 0] = scale_factors[0]
        scaling_matrix[1, 1] = scale_factors[1]
        scaling_matrix[2, 2] = scale_factors[2]
    else:
        # if is_bg_obj and align_direction[2] > 0:
        #     # Meaning the object is a supportting background
        #     scale_factors = np.max(scale_factors[:2])  # Only care about x, y
        #     scaling_matrix[0, 0] = scale_factors
        #     scaling_matrix[1, 1] = scale_factors
        #     scaling_matrix[2, 2] = scale_factors
        # else:
        #     scale_factors = np.min(scale_factors)
        #     scaling_matrix[0, 0] = scale_factors
        #     scaling_matrix[1, 1] = scale_factors
        #     scaling_matrix[2, 2] = scale_factors
        scale_factors = np.min(scale_factors)
        scaling_matrix[0, 0] = scale_factors
        scaling_matrix[1, 1] = scale_factors
        scaling_matrix[2, 2] = scale_factors
    # Compute translation
    translation = goal_center - robot_center * scale_factors
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation

    # Combine scaling and translation
    transform_matrix = np.dot(translation_matrix, scaling_matrix)

    cur_min = goal_center - scaling_matrix[:3, :3].diagonal() * obj_size / 2
    cur_max = goal_center + scaling_matrix[:3, :3].diagonal() * obj_size / 2
    # Perform alignment
    align_transform = np.eye(4)
    if align_direction[0] < 0:
        align_transform[0, 3] = goal_min[0] - cur_min[0]
    elif align_direction[0] > 0:
        align_transform[0, 3] = goal_max[0] - cur_max[0]
    if align_direction[1] < 0:
        align_transform[1, 3] = goal_min[1] - cur_min[1]
    elif align_direction[1] > 0:
        align_transform[1, 3] = goal_max[1] - cur_max[1]
    if align_direction[2] < 0:
        align_transform[2, 3] = goal_min[2] - cur_min[2]
    elif align_direction[2] > 0:
        align_transform[2, 3] = goal_max[2] - cur_max[2]

    transform_matrix = np.dot(align_transform, transform_matrix)
    return transform_matrix


def read_articulation(data_dir, data_name):
    # Load joint info
    joint_info_file = os.path.join(data_dir, data_name, "mobility_v2.json")
    joint_info = json.load(open(joint_info_file))
    # Filterout junk data
    joint_info = [joint for joint in joint_info if joint["joint"] != "junk"]
    # Load semantic info
    semantic_data = []
    semantic_file = os.path.join(data_dir, data_name, "semantics.txt")
    with open(semantic_file, "r") as file:
        for line in file:
            parts = line.strip().split(" ")
            if len(parts) == 3:
                semantic_data.append(
                    {
                        "link_name": parts[0],
                        "joint_type": parts[1],
                        "semantic": parts[2],
                    }
                )
    articulation_info = {
        "id_map": {},
    }
    for link_idx, link_data in enumerate(joint_info):
        if "jointData" in link_data and link_data["jointData"]:
            joint_type = semantic_data[link_idx]["joint_type"]
            if joint_type in ["fixed", "free", "heavy"]:
                continue
            axis_origin = link_data["jointData"]["axis"]["origin"]
            axis_direction = link_data["jointData"]["axis"]["direction"]
            articulation_info[link_data["id"]] = {
                "axis_origin": np.array(axis_origin),
                "axis_direction": np.array(axis_direction),
                "name": link_data["name"],
            }
            articulation_info["id_map"][link_idx] = link_data["id"]
    return articulation_info


def generater_label_3d(
    color_imgs,
    depth_imgs,
    mask_imgs,
    camera_poses,
    obj_transforms,
    camera_info,
    articulation_info,
    debug=False,
):
    sample_size = -1
    fx = camera_info["fx"]
    fy = camera_info["fy"]
    cx = camera_info["cx"]
    cy = camera_info["cy"]
    intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    fix_func = lambda v: np.array([x if x is not None else 0 for x in v])

    label_3ds = []
    # Generate pcd
    for image_idx in range(len(color_imgs)):
        label_3d = {}
        # Align mask to articulation info
        mask = mask_imgs[image_idx]
        new_mask = np.ones_like(mask) * 255
        for mask_id in np.unique(mask):
            if mask_id == 0:
                continue
            mask_i = mask == mask_id
            if np.sum(mask_i) > 0 and (mask_id - 1) in articulation_info["id_map"]:
                new_id = int(articulation_info["id_map"][mask_id - 1])
                new_mask[mask_i] = new_id
        color = color_imgs[image_idx]
        depth = (depth_imgs[image_idx] / 1000.0).astype(np.float32)
        points, colors, masks = get_pointcloud(
            color,
            -depth,
            new_mask,
            intrinsic,
            sample_size,
            flip_x=True,
        )
        camera_pose = camera_poses[image_idx]
        for id, _info in articulation_info.items():
            if not isinstance(id, int):
                continue
            axis_origin = fix_func(_info["axis_origin"])
            axis_origin = np.array([-axis_origin[2], -axis_origin[0], axis_origin[1]])
            axis_direction = fix_func(_info["axis_direction"])
            axis_direction = np.array(
                [-axis_direction[2], -axis_direction[0], axis_direction[1]]
            )
            obj_transform = np.array(obj_transforms[image_idx])
            axis_origin = obj_transform[:3, :3] @ axis_origin + obj_transform[:3, 3]
            axis_direction = obj_transform[:3, :3] @ axis_direction
            axis_direction = axis_direction / np.linalg.norm(axis_direction + 1e-6)
            # Assemble joint_T
            j_axis_z = axis_direction
            j_axis_x = (
                np.array([1.0, 0.0, 0.0])
                if np.abs(j_axis_z[0]) < 0.9
                else np.array([0.0, 1.0, 0.0])
            )
            j_axis_y = np.cross(j_axis_z, j_axis_x)
            j_axis_y = j_axis_y / (np.linalg.norm(j_axis_y) + 1e-6)
            j_axis_x = np.cross(j_axis_y, j_axis_z)
            j_axis_x = j_axis_x / (np.linalg.norm(j_axis_x) + 1e-6)
            joint_R = np.array([j_axis_x, j_axis_y, j_axis_z]).T
            j_T = np.eye(4)
            j_T[:3, :3] = joint_R
            j_T[:3, 3] = axis_origin
            j_T = np.linalg.inv(camera_pose) @ j_T
            j_T_inv = np.linalg.inv(j_T)

            points_j = points[np.where(masks == id)[0]]
            points_j = points_j @ j_T_inv[:3, :3].T + j_T_inv[:3, 3]
            # Generate 3D bbox
            bbox = AxisBBox3D()
            if points_j.shape[0] >= 8:
                bbox.create_minimum_projected_bbox(points_j)
            else:
                continue
            bbox.rotate(j_T[:3, :3], (0, 0, 0))
            bbox.translate(j_T[:3, 3])
            bbox_array = bbox.get_bbox_array()
            axis_array = bbox.get_axis_array()
            label_3d[id] = {
                "bbox": bbox_array.tolist(),
                "axis": axis_array.tolist(),
                "name": _info["name"],
                "image_idx": image_idx,
                "camera_pose": camera_pose,
            }
            # [DEBUG]
            if debug:
                # 3D visualization
                # masks[masks == 255] = 20
                # masks = masks.astype(np.uint8)
                # check_annotations_3d(points, bbox_array, axis_array, masks)
                # 2D visualization
                vis_image = check_annotations_2d(
                    color, bbox_array, axis_array, intrinsic
                )
                cv2.imshow("vis_image", vis_image)
                cv2.waitKey(0)

        label_3ds.append(label_3d)
    return label_3ds


################################ Render functions ################################
def render_parts_into_block(
    mesh_map,
    camera_info,
    predefined_camera_poses,
    predefined_light_poses,
    idx_func,
    num_poses,
    keep_ratio=True,
    is_link_map=False,
    is_bg_image=False,
):
    """Render parts into block."""
    mesh_map = copy.deepcopy(mesh_map)  # Avoid modifying the original mesh_map
    scene = pyrender.Scene()
    # Add lights
    for light_pose in predefined_light_poses:
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=50.0)
        scene.add(light, pose=light_pose)
    # Add mesh
    mesh_name_dict = {}
    # Debug
    for mesh, (pose, name) in mesh_map.items():
        if is_bg_image:
            # Only render the background
            if name != "bg":
                continue
        # mesh.apply_transform(transform)
        pymesh = pyrender.Mesh.from_trimesh(mesh)
        pymesh.name = name
        if is_link_map:
            # Use a random color for the link mesh
            color = np.random.rand(3)
            pymesh.primitives[0].material.baseColorFactor = color
        mesh_node = scene.add(pymesh, pose=np.eye(4))

        # Compute center
        mesh_pose = mesh_node.matrix
        center_3d = np.mean(mesh.vertices, axis=0)
        center_3d = np.dot(mesh_pose, np.append(center_3d, 1))[:3]
        mesh_name_dict[name] = center_3d

    # Add camera
    fx = camera_info["fx"]
    fy = camera_info["fy"]
    cx = camera_info["cx"]
    cy = camera_info["cy"]
    width = camera_info["width"]
    height = camera_info["height"]
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
    r = pyrender.OffscreenRenderer(width, height)

    # Prepare data
    camera_node = None
    color_imgs = []
    depth_imgs = []
    mask_imgs = []
    annotations = []
    # Do rendering
    for pose_idx in range(num_poses):
        image_idx = idx_func(pose_idx)
        camera_pose = predefined_camera_poses[image_idx]
        if camera_node is not None:
            scene.remove_node(camera_node)
        camera_node = scene.add(camera, pose=np.array(camera_pose))

        # Render color image
        flags = pyrender.RenderFlags.RGBA
        try:
            color, _ = r.render(scene, flags=flags)
        except Exception as e:
            print(f"Error in rendering: {e}")
            return None, None, None, None
        color_imgs.append(color[:, :, :3])

        if not is_bg_image and not is_link_map:
            continue
        # Render the depth of the whole scene
        flags = pyrender.RenderFlags.DEPTH_ONLY
        full_depth = r.render(scene, flags=flags)
        depth_imgs.append(full_depth)

        if is_bg_image or not is_link_map:
            continue
        mask = np.zeros((height, width), dtype=np.uint8)
        # Hide all mesh nodes
        for mn in scene.mesh_nodes:
            mn.mesh.is_visible = False

        # Iterate through each mesh node
        for node in scene.mesh_nodes:
            if node.mesh.name == "bg":
                continue
            node.mesh.is_visible = True
            # Parse link_idx
            link_idx = int(node.mesh.name.split("_")[-1])
            # Part center: Average of all vertices
            center_3d = mesh_name_dict[node.mesh.name]
            # Compute mask & bbox
            depth = r.render(scene, flags=flags)
            mask_vis = np.logical_and(depth <= full_depth, np.abs(depth) > 0)
            mask_all = np.abs(depth) > 0
            vis_ratio = mask_vis.sum() / (mask_all.sum() + 1e-6)
            if vis_ratio > 0.1:
                # Only consider visible parts
                mask[mask_vis] = link_idx + 1
            node.mesh.is_visible = False
        # Show all meshes again
        for mn in scene.mesh_nodes:
            mn.mesh.is_visible = True
        # # [DEBUG]
        # plt.imshow(mask_img)
        # plt.show()
        mask_imgs.append(mask)
    r.delete()

    return color_imgs, depth_imgs, mask_imgs, annotations


def render_object_into_block(
    data_id,
    data_dir,
    output_dir,
    camera_info,
    on_list,
    under_list,
    other_list,
    meta_info,
    num_poses,
    num_joint_values,
    video_mode=False,
    joint_select_idx=0,
    solo_prob=0.1,
    debug=False,
    skip_exist=False,
):
    """Render an object into a bbox. This bbox is axis-aligned."""
    if data_id not in meta_info["meta"]:
        # Skip if the object is not in the meta info
        return False
    rng = np.random.RandomState(int(data_id) + int(joint_select_idx))
    export_dir = os.path.join(output_dir, data_id)
    os.makedirs(export_dir, exist_ok=True)
    # Check existed files
    if skip_exist:
        if not video_mode:
            exist_info_file = os.path.join(export_dir, "info.json")
        else:
            exist_info_file = os.path.join(export_dir, f"info_{joint_select_idx}.json")
        if os.path.exists(exist_info_file):
            with open(exist_info_file, "r") as f:
                exist_info = json.load(f)
            if len(exist_info["camera_poses"]) == num_joint_values * num_poses:
                return True
    sample_type = "xy"
    only_front = True
    keep_ratio = False
    cam_radius_min = 2.5
    cam_radius_max = 3.0
    light_radius_min = 3.0
    light_radius_max = 3.5
    num_samples = num_joint_values * num_poses
    width = camera_info["width"]
    height = camera_info["height"]
    render_result = {
        "color": np.zeros((num_samples, width, height, 3), dtype=np.uint8),
        "depth": np.zeros((num_samples, width, height), dtype=np.uint16),
        "mask": np.zeros((num_samples, width, height), dtype=np.uint8),
        "bg_color": np.zeros((num_poses, width, height, 3), dtype=np.uint8),
        "bg_depth": np.zeros((num_poses, width, height), dtype=np.uint16),
    }
    info = {}
    info["camera_info"] = camera_info
    meta_file = f"{data_dir}/{data_id}/meta.json"
    with open(meta_file, "r") as f:
        meta = json.load(f)
    info["model_cat"] = meta["model_cat"]
    info["camera_poses"] = [None] * num_samples
    info["obj_transforms"] = [None] * num_samples
    info["data_ids"] = [None] * num_samples
    # Generate camera pose
    radius = 1.0
    cam_radius_min = radius * cam_radius_min  # Minimum distance from the look-at point
    cam_radius_max = radius * cam_radius_max  # Maximum distance from the look-at point
    look_at = np.array([0.0, 0.0, 0.0])
    look_at += np.random.normal(scale=0.1 * radius, size=3)
    up = np.array([0.0, 0.0, 1.0])
    # Apply a small disturbance to the up vector
    up += np.random.normal(scale=0.07 * np.pi, size=3)
    up /= np.linalg.norm(up)

    predefined_camera_poses = []
    for i in range(num_poses):
        if sample_type == "uniform":
            # Compute the camera pose
            camera_pose = sample_camera_pose(
                cam_radius_min, cam_radius_max, look_at, up, only_front, rng
            )
        elif sample_type == "xy":
            camera_pose = sample_camera_pose_xy(
                cam_radius_min, cam_radius_max, look_at, up, only_front, rng
            )
        else:
            raise NotImplementedError
        predefined_camera_poses += [camera_pose] * num_joint_values

    # Generate light poses
    predefined_light_poses = []
    for i in range(3):
        light_radius = np.random.uniform(low=light_radius_min, high=light_radius_max)
        light_pose = np.eye(4)
        if i == 0:
            light_pose[:3, 3] = np.array([-light_radius, 0.0, 0.0])
        elif i == 1:
            light_pose[:3, 3] = np.array([0.0, light_radius, 0.0])
        elif i == 2:
            light_pose[:3, 3] = np.array([0.0, 0.0, light_radius])
        predefined_light_poses.append(light_pose)

    # Read articulation informtation
    articulation_info = read_articulation(data_dir, data_id)
    if video_mode:
        if rng.rand() > 0.5:
            joint_speed = 1.0 / num_joint_values
        else:
            joint_speed = -1.0 / num_joint_values
    else:
        joint_speed = 0.0
    for joint_value_idx in range(num_joint_values):
        if not video_mode or joint_value_idx == 0:
            # Generate random set-up
            data_ids, goal_bbox_list, align_direction_list = (
                generate_block_updown_setup(
                    data_id, on_list, under_list, other_list, meta_info, solo_prob, rng
                )
            )
        robot_link_mesh_map, robot_visual_map, obj_transforms = generate_robot_meshes(
            data_dir,
            data_ids,
            goal_bbox_list,
            align_direction_list,
            meta_info,
            rng,
            video_mode=video_mode,
            joint_value_idx=joint_value_idx,
            joint_speed=joint_speed,
            joint_select_idx=joint_select_idx,
            model_cat=meta["model_cat"],
        )
        if robot_link_mesh_map is None:
            return False
        # Render objects
        idx_func = lambda x: x * num_joint_values + joint_value_idx
        # 1. Render on link level
        no_texture_color_imgs, depth_imgs, mask_imgs, annotations = (
            render_parts_into_block(
                robot_link_mesh_map,
                camera_info,
                predefined_camera_poses,
                predefined_light_poses,
                idx_func,
                num_poses,
                keep_ratio=keep_ratio,
                is_link_map=True,
            )
        )
        # 2. Render on visual level
        texture_color_imgs, _, _, _ = render_parts_into_block(
            robot_visual_map,
            camera_info,
            predefined_camera_poses,
            predefined_light_poses,
            idx_func,
            num_poses,
            keep_ratio=keep_ratio,
            is_link_map=False,
        )
        if texture_color_imgs is not None:
            color_imgs = texture_color_imgs
        elif no_texture_color_imgs is not None:
            color_imgs = no_texture_color_imgs
        else:
            return False
        if joint_value_idx == 0:
            # Generate background image
            bg_color_imgs, bg_depth_imgs, _, __ = render_parts_into_block(
                robot_visual_map,
                camera_info,
                predefined_camera_poses,
                predefined_light_poses,
                idx_func,
                num_poses,
                keep_ratio=keep_ratio,
                is_link_map=False,
                is_bg_image=True,
            )
            render_result["bg_depth"] = bg_depth_imgs
            render_result["bg_color"] = bg_color_imgs
        # Save images
        for pose_idx in range(num_poses):
            image_idx = idx_func(pose_idx)
            color_img = cv2.cvtColor(color_imgs[pose_idx], cv2.COLOR_RGBA2BGRA)
            depth_img = (depth_imgs[pose_idx] * 1000).astype(np.uint16)
            mask_img = mask_imgs[pose_idx]
            render_result["color"][image_idx] = color_img[:, :, :3].astype(np.uint8)
            render_result["depth"][image_idx] = depth_img.astype(np.uint16)
            render_result["mask"][image_idx] = mask_img.astype(np.uint8)
            # Append info
            info["camera_poses"][image_idx] = predefined_camera_poses[
                image_idx
            ].tolist()
            info["obj_transforms"][image_idx] = obj_transforms[
                0
            ].tolist()  # Only first object
        # Update info
        info["data_ids"][joint_value_idx] = data_ids
    # Generate labels for the scene
    label_3ds = generater_label_3d(
        render_result["color"],
        render_result["depth"],
        render_result["mask"],
        info["camera_poses"],
        info["obj_transforms"],
        camera_info,
        articulation_info,
        debug,
    )
    if video_mode:
        video_dir = f"{export_dir}/video"
        os.makedirs(video_dir, exist_ok=True)
        # Write video
        for pose_idx in range(num_poses):
            color_images = render_result["color"][
                pose_idx * num_joint_values : (pose_idx + 1) * num_joint_values
            ]
            color_images = [
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in color_images
            ]  # For imageio
            imageio.mimsave(
                f"{video_dir}/color_{joint_select_idx}_{pose_idx}.mp4",
                color_images,
                fps=30,
            )
            post_fix = f"_{joint_select_idx}"
        # Save background image
        for pose_idx in range(num_poses):
            bg_color_img = render_result["bg_color"][pose_idx]
            bg_color_img = cv2.cvtColor(bg_color_img, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(
                f"{video_dir}/bg_color_{joint_select_idx}_{pose_idx}.png", bg_color_img
            )
            bg_depth_img = render_result["bg_depth"][pose_idx]
            # Save normalized depth image
            depth_max = np.max(bg_depth_img)
            if bg_depth_img[bg_depth_img > 0].shape[0] == 0:
                depth_min = np.min(bg_depth_img)
            else:
                depth_min = np.min(bg_depth_img[bg_depth_img > 0])
            bg_depth_img[bg_depth_img == 0] = depth_max
            bg_depth_img = (
                (bg_depth_img - depth_min) / (depth_max - depth_min + 1e-6) * 255
            )
            bg_depth_img = 255 - bg_depth_img
            cv2.imwrite(
                f"{video_dir}/bg_depth_{joint_select_idx}_{pose_idx}.png", bg_depth_img
            )
            # # Save canny edge image
            # bg_color_img = cv2.cvtColor(bg_color_img, cv2.COLOR_BGR2GRAY)
            # edges = cv2.Canny(bg_color_img, 50, 100)
            # cv2.imwrite(f"{video_dir}/bg_edge_{pose_idx}.png", edges)
        info["joint_select_idx"] = joint_select_idx
        info["joint_speed"] = joint_speed
    else:
        post_fix = ""
    # Save results
    np.savez_compressed(
        f"{export_dir}/color_imgs{post_fix}.npz", images=render_result["color"]
    )
    np.savez_compressed(
        f"{export_dir}/depth_imgs{post_fix}.npz", images=render_result["depth"]
    )
    np.savez_compressed(
        f"{export_dir}/mask_imgs{post_fix}.npz", images=render_result["mask"]
    )
    np.savez_compressed(
        f"{export_dir}/bg_color{post_fix}.npz", images=render_result["bg_color"]
    )
    np.savez_compressed(
        f"{export_dir}/bg_depth{post_fix}.npz", images=render_result["bg_depth"]
    )
    with open(f"{export_dir}/annotations_3d{post_fix}.json", "w") as f:
        json.dump(label_3ds, f)
    with open(f"{export_dir}/info{post_fix}.json", "w") as f:
        json.dump(info, f)
    # Copy meta info
    shutil.copy(meta_file, f"{export_dir}/meta.json")
    return True


###################################### Multi-processing ######################################


def launch_multi_process(
    data_ids,
    data_dir,
    output_dir,
    camera_info,
    on_list,
    under_list,
    other_list,
    meta_info,
    num_poses,
    num_joint_values,
    num_joint_select_idx,
    max_process=64,
    video_mode=False,
    solo_prob=0.1,
    debug=False,
    wait_time=1200,
    skip_exist=False,
):
    # Assemble tuple
    data_tuples = []
    for data_id in data_ids:
        for joint_select_idx in range(num_joint_select_idx):
            data_tuples.append((data_id, joint_select_idx))
    num_processes = min(min(multiprocessing.cpu_count(), len(data_tuples)), max_process)
    print(f"Process {len(data_tuples)} data with {num_processes} processes")
    pool = multiprocessing.Pool(processes=num_processes)
    with tqdm(total=len(data_tuples)) as pbar:
        result_objects = []
        for data_id, joint_select_idx in data_tuples:
            result = pool.apply_async(
                render_object_into_block,
                args=(
                    data_id,
                    data_dir,
                    output_dir,
                    camera_info,
                    on_list,
                    under_list,
                    other_list,
                    meta_info,
                    num_poses,
                    num_joint_values,
                    video_mode,
                    joint_select_idx,
                    solo_prob,
                    debug,
                    skip_exist,
                ),
                error_callback=lambda e: logger.error(
                    f"Error processing ({data_id}, {joint_select_idx}): {e}"
                ),
            )
            result_objects.append((data_id, joint_select_idx, result))
        # Collect results with a timeout
        for data_id, joint_select_idx, result in result_objects:
            try:
                status = result.get(timeout=wait_time)  # Timeout set to 20 mins
                if not status:
                    logger.error(
                        f"Error: {data_id}-{joint_select_idx} processing failed."
                    )
            except multiprocessing.TimeoutError:
                logger.error(
                    f"Timeout: {data_id}-{joint_select_idx} processing exceeded time limit."
                )
            except Exception as e:
                logger.error(
                    f"Error: {data_id}-{joint_select_idx} processing failed with exception: {e}."
                )
            finally:
                pbar.update(1)
    pool.close()
    pool.join()


if __name__ == "__main__":
    ## Configurations
    import argparse

    debug = False
    video_mode = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="9748", help="Data name")
    parser.add_argument(
        "--data_dir", type=str, default="/home/harvey/Data/partnet-mobility-v0/dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/harvey/Data/partnet-mobility-v0/output_v2",
    )
    parser.add_argument("--num_poses", type=int, default=2)
    parser.add_argument("--num_joint_select_idx", type=int, default=1)
    parser.add_argument("--num_joint_values", type=int, default=40)
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    data_name = args.data_name
    num_poses = args.num_poses
    num_joint_select_idx = args.num_joint_select_idx
    num_joint_values = args.num_joint_values

    meta_info = json.load(open(f"{data_dir}/meta.json"))
    on_list = meta_info["on"]
    under_list = meta_info["under"]
    other_list = meta_info["other"]
    camera_info = {
        "fx": 1000,
        "fy": 1000,
        "cx": 480,
        "cy": 480,
        "width": 960,
        "height": 960,
    }
    data_ids = os.listdir(data_dir)
    data_ids = meta_info["all"] if data_name == "all" else [data_name]
    # Sort
    data_ids = sorted(data_ids)
    data_ids = data_ids[:2]
    solo_prob = 0.25
    skip_exist = True

    # # Launch [DEBUG]
    # launch_multi_process(
    #     data_ids,
    #     data_dir,
    #     output_dir,
    #     camera_info,
    #     on_list,
    #     under_list,
    #     other_list,
    #     meta_info,
    #     num_poses,
    #     num_joint_values,
    #     num_joint_select_idx,
    #     video_mode=video_mode,
    #     solo_prob=solo_prob,
    #     debug=debug,
    #     skip_exist=skip_exist,
    # )

    if data_name == "all":
        # Render object
        launch_multi_process(
            data_ids,
            data_dir,
            output_dir,
            camera_info,
            on_list,
            under_list,
            other_list,
            meta_info,
            num_poses,
            num_joint_values,
            num_joint_select_idx,
            video_mode=video_mode,
            solo_prob=solo_prob,
            debug=debug,
            skip_exist=skip_exist,
        )
    else:
        # Render object
        render_object_into_block(
            data_name,
            data_dir,
            output_dir,
            camera_info,
            on_list,
            under_list,
            other_list,
            meta_info,
            num_poses,
            num_joint_values,
            video_mode=video_mode,
            debug=debug,
            skip_exist=skip_exist,
        )
