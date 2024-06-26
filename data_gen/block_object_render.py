"""Do object render as building blocks."""

import os
import cv2
import copy
import pyrender
import trimesh
import numpy as np
from urchin import URDF


################################ Utility functions ################################
def sample_camera_pose(
    cam_radius_min: float,
    cam_radius_max: float,
    look_at: np.ndarray,
    up: np.ndarray,
    only_front: bool = False,
):
    # Sample a radius within the given range
    radius = np.random.uniform(cam_radius_min, cam_radius_max)

    # Sample spherical coordinates
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, np.pi)

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
):
    """Sample a camera pose given the look-at point and up vector at x-y plane."""
    # Sample a radius within the given range
    radius = np.random.uniform(cam_radius_min, cam_radius_max)

    # Convert spherical coordinates to Cartesian coordinates for the camera position
    if only_front:
        theta = np.random.uniform(np.pi * 0.6, np.pi * 1.4)
        phi = np.random.uniform(0.23 * np.pi, 0.26 * np.pi)
        x = radius * np.cos(theta) * np.cos(phi) + look_at[0]
        y = radius * np.sin(theta) * np.cos(phi) + look_at[1]
        z = radius * np.sin(phi) + look_at[2]
    else:
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(-0.25 * np.pi, 0.25 * np.pi)
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


def generate_robot_cfg(robot: URDF, rng: np.random.RandomState):
    # Compute joint kinematic level
    kinematic_level = compute_kinematic_level(robot)

    actuaion_joints = robot.actuated_joints
    joint_cfg = {}  # Save in joint's name, raw value
    link_cfg = {}  # Save in link's name, normalized
    for joint in actuaion_joints:
        joint_value_sample = rng.rand()
        if joint.limit is None:
            joint_limit = [-np.pi, np.pi]
        else:
            joint_limit = [joint.limit.lower, joint.limit.upper]
        limit_low, limit_high = joint_limit[0], joint_limit[1]
        if joint_value_sample < 0.1:
            joint_value = limit_low
        elif joint_value_sample > 0.9:
            joint_value = limit_high
        else:
            joint_value = joint_value_sample * (limit_high - limit_low) + limit_low
        if kinematic_level > 1:
            # Existing hierarchical joint, disable the joint
            joint_value = 0.0
        joint_cfg[joint.name] = joint_value
        link_cfg[joint.child] = (joint_value - limit_low) / (
            limit_high - limit_low + 1e-6
        )  # normalize
    return joint_cfg, link_cfg


def generate_block_setup(data_id, on_list, under_list, rng=np.random.RandomState(0)):
    """
    On list is the list of objects that are placed upon other objects.
    Under list is the list of objects that are placed under other objects.
    Return [goal_bbox_list, align_direction_list]
    """
    on_bbox_list = [
        [[0.0, 0.0, 0.0], [0.5, 0.5, 1.0]],
        [[-0.25, -0.25, 0.0], [0.25, 0.25, 1.0]],
        [[0.5, 0.5, 0.0], [1.0, 1.0, 1.0]],
    ]
    goal_bbox_list = []
    align_direction_list = []
    data_ids = []
    if data_id in on_list:
        # Object is placed on other objects
        data_ids.append(data_id)
        goal_bbox_list.append(np.array(on_bbox_list[rng.choice(len(on_bbox_list))]))
        align_direction_list.append(np.array([0.0, 0.0, -1.0]))
        # Select under object
        under_data_id = rng.choice(under_list)
        data_ids.append(under_data_id)
        goal_bbox_list.append(np.array([[-0.5, -0.5, 0.0], [0.5, 0.5, -1.0]]))
        align_direction_list.append(np.array([0.0, 0.0, 1.0]))
    elif data_id in under_list:
        data_ids.append(data_id)
        goal_bbox_list.append(np.array([[-0.5, -0.5, 0.0], [0.5, 0.5, -1.0]]))
        align_direction_list.append(np.array([0.0, 0.0, 1.0]))
        on_data_id = rng.choice(on_list)
        data_ids.append(on_data_id)
        goal_bbox_list.append(np.array(on_bbox_list[rng.choice(len(on_bbox_list))]))
        align_direction_list.append(np.array([0.0, 0.0, -1.0]))
    return data_ids, goal_bbox_list, align_direction_list


def generate_robot_mesh(
    robot: URDF, joint_cfg: dict, goal_bbox, align_dir, is_bg_obj=False
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
        robot_visual_map[mesh] = (pose, "visual")
        mesh.apply_transform(pose)

    # Merged mesh
    robot_mesh_list = []
    for mesh, (pose, name) in robot_link_mesh_map.items():
        # mesh.apply_transform(pose)
        robot_mesh_list.append(mesh)
    robot_mesh = trimesh.util.concatenate(robot_mesh_list)
    robot_bbox = robot_mesh.bounds
    # # # [DEBUG]: Trimesh Visualize
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
    rng=np.random.RandomState(0),
    keep_ratio=True,
):
    robot_link_mesh_map = {}
    robot_visual_map = {}
    for idx, (data_id, goal_bbox, align_direction) in enumerate(
        zip(data_ids, goal_bbox_list, align_direction_list)
    ):
        is_bg_obj = (
            True if idx > 0 else False
        )  # Only the first object is the target object
        data_file = f"{data_dir}/{data_id}/mobility.urdf"
        robot = URDF.load(data_file)
        joint_cfg, link_cfg = generate_robot_cfg(robot, rng)
        _robot_link_mesh_map, _robot_visual_map, robot_bbox = generate_robot_mesh(
            robot, joint_cfg, goal_bbox, align_direction, is_bg_obj
        )

        # Compute bbox & transform
        transform = compute_bbox_transform(
            robot_bbox, goal_bbox, align_direction, keep_ratio
        )
        for mesh, (pose, name) in _robot_link_mesh_map.items():
            mesh.apply_transform(transform)
        for mesh, (pose, name) in _robot_visual_map.items():
            mesh.apply_transform(transform)

        robot_link_mesh_map.update(_robot_link_mesh_map)
        robot_visual_map.update(_robot_visual_map)
    # # # [DEBUG]
    scene = trimesh.Scene()
    for mesh, (pose, name) in robot_visual_map.items():
        scene.add_geometry(mesh)
    axis = trimesh.creation.axis(
        origin_size=0.1
    )  # You can adjust the size as needed
    scene.add_geometry(axis)
    scene.show()
    return robot_link_mesh_map, robot_visual_map


def compute_bbox_transform(init_bbox, goal_bbox, align_direction, keep_ratio=True):
    # Compute dimensions
    robot_min = np.min(init_bbox, axis=0)
    robot_max = np.max(init_bbox, axis=0)
    robot_size = robot_max - robot_min
    robot_center = (robot_min + robot_max) / 2

    goal_min = np.min(goal_bbox, axis=0)
    goal_max = np.max(goal_bbox, axis=0)
    goal_size = goal_max - goal_min
    goal_center = (goal_min + goal_max) / 2

    # Compute scale factors
    scale_factors = goal_size / robot_size
    scaling_matrix = np.eye(4)
    if not keep_ratio:
        scaling_matrix[0, 0] = scale_factors[0]
        scaling_matrix[1, 1] = scale_factors[1]
        scaling_matrix[2, 2] = scale_factors[2]
    else:
        scaling_matrix[0, 0] = np.min(scale_factors)
        scaling_matrix[1, 1] = np.min(scale_factors)
        scaling_matrix[2, 2] = np.min(scale_factors)
        scale_factors = np.min(scale_factors)

    # Compute translation
    translation = goal_center - robot_center * scale_factors
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = translation

    # Combine scaling and translation
    transform_matrix = np.dot(translation_matrix, scaling_matrix)

    cur_min = goal_center - scaling_matrix[:3, :3].diagonal() * robot_size / 2
    cur_max = goal_center + scaling_matrix[:3, :3].diagonal() * robot_size / 2
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
        # mesh.apply_transform(transform)
        pymesh = pyrender.Mesh.from_trimesh(mesh)
        pymesh.name = name
        if is_link_map:
            # Use a random color for the link mesh
            color = np.random.rand(3)
            pymesh.primitives[0].material.baseColorFactor = color
        # pose[:3, 3] = transform[:3, :3] @ pose[:3, 3] + transform[:3, 3]
        pose = np.eye(4)
        mesh_node = scene.add(pymesh, pose=pose)

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

    # Do rendering
    camera_node = None
    color_imgs = []
    depth_imgs = []
    mask_imgs = []
    annotations = []
    for pose_idx in range(num_poses):
        image_idx = idx_func(pose_idx)
        camera_pose = predefined_camera_poses[image_idx]
        if camera_node is not None:
            scene.remove_node(camera_node)
        camera_node = scene.add(camera, pose=np.array(camera_pose))

        # Render color image
        if not is_link_map:
            flags = pyrender.RenderFlags.RGBA
            color, depth = r.render(scene, flags=flags)
            color_imgs.append(color)
        else:
            # Render the depth of the whole scene
            flags = pyrender.RenderFlags.DEPTH_ONLY
            full_depth = r.render(scene, flags=flags)
            depth_imgs.append(full_depth)

            mask_img = np.zeros((height, width), dtype=np.uint8)
            # Hide all mesh nodes
            for mn in scene.mesh_nodes:
                mn.mesh.is_visible = False

            # Iterate through each mesh node
            for node in scene.mesh_nodes:
                if node.mesh.name == "bg":
                    continue
                segimg = np.zeros((height, width), dtype=np.uint8)
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

                if np.any(mask_vis):
                    # Extract a rotating bbox from mask
                    contours, _ = cv2.findContours(
                        mask_vis.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    # Find the largest contour based on area
                    largest_contour = max(contours, key=cv2.contourArea)

                    # Compute the minimal area bounding box for the largest contour
                    rect = cv2.minAreaRect(largest_contour)

                    y, x = np.where(mask_vis)
                    x_min, x_max = x.min(), x.max()
                    y_min, y_max = y.min(), y.max()

                    area = int((x_max - x_min) * (y_max - y_min))
                    node.mesh.is_visible = False

                    # Save annotation
                    mask_img[mask_vis] = link_idx + 1
                    annotation = {
                        "bbox": [
                            int(x_min),
                            int(y_min),
                            int(x_max - x_min),
                            int(y_max - y_min),
                        ],
                        "rot_bbox": [
                            rect[0][0],
                            rect[0][1],
                            rect[1][0],
                            rect[1][1],
                            rect[2],
                        ],
                        "area": area,
                        "vis_ratio": vis_ratio,
                        "center_3d": center_3d.tolist(),
                        "image_id": image_idx,
                        "id": link_idx,
                        "name": node.mesh.name,
                        "camera_pose": camera_pose,
                    }
                    annotations.append(annotation)
            # Show all meshes again
            for mn in scene.mesh_nodes:
                mn.mesh.is_visible = True
            mask_imgs.append(mask_img)
    r.delete()

    return color_imgs, depth_imgs, mask_imgs, annotations


def render_object_into_block(data_id, data_dir, camera_info, on_list, under_list):
    """Render an object into a bbox. This bbox is axis-aligned."""
    sample_type = "uniform"
    only_front = True
    keep_ratio = False
    num_joint_values = 4
    cam_radius_min = 2.5
    cam_radius_max = 3.0
    light_radius_min = 3.0
    light_radius_max = 3.5
    num_poses = 2
    num_samples = num_joint_values * num_poses
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
                cam_radius_min, cam_radius_max, look_at, up, only_front=only_front
            )
        elif sample_type == "xy":
            camera_pose = sample_camera_pose_xy(
                cam_radius_min, cam_radius_max, look_at, up, only_front=only_front
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

    rng = np.random.RandomState(0)
    for joint_idx in range(num_joint_values):
        # Generate random set-up
        data_ids, goal_bbox_list, align_direction_list = generate_block_setup(
            data_id, on_list, under_list, rng
        )
        robot_link_mesh_map, robot_visual_map = generate_robot_meshes(
            data_dir, data_ids, goal_bbox_list, align_direction_list, rng
        )
        goal_bbox = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
        # Render objects
        idx_func = lambda x: x * num_joint_values + joint_idx
        # 1. Render on link level
        _, depth_imgs, mask_imgs, annotations = render_parts_into_block(
            robot_link_mesh_map,
            camera_info,
            predefined_camera_poses,
            predefined_light_poses,
            idx_func,
            num_poses,
            keep_ratio=keep_ratio,
            is_link_map=True,
        )
        # 2. Render on visual level
        color_imgs, _, _, _ = render_parts_into_block(
            robot_visual_map,
            camera_info,
            predefined_camera_poses,
            predefined_light_poses,
            idx_func,
            num_poses,
            keep_ratio=keep_ratio,
            is_link_map=False,
        )
        # Save images
        for pose_idx, img in enumerate(color_imgs):
            cv2.imwrite(f"{output_dir}/color_test/color_{idx_func(pose_idx)}.png", img)
        pass


if __name__ == "__main__":
    under_list = ["103351", "40417"]
    on_list = ["920", "101564", "152", "991", "103007", "103037"]
    # on_list = ["152"]

    data_dir = "/home/harvey/Data/partnet-mobility-v0/dataset"
    output_dir = "/home/harvey/Data/partnet-mobility-v0/output"
    data_name = "103351"  #
    data_file = f"{data_dir}/{data_name}/mobility.urdf"
    output_dir = f"{output_dir}/{data_name}"
    camera_info = {
        "fx": 1000,
        "fy": 1000,
        "cx": 480,
        "cy": 480,
        "width": 1000,
        "height": 1000,
    }
    os.makedirs(os.path.join(output_dir, "color_test"), exist_ok=True)

    render_object_into_block(data_name, data_dir, camera_info, on_list, under_list)
