"""Pyrender tools for rendering Parts."""

import numpy as np
from packaging import version

# Check NumPy version
if version.parse(np.__version__) >= version.parse("1.20.0"):
    # Create an alias for np.int
    np.int = int

import matplotlib.pyplot as plt
import pyrender
import cv2


def sample_camera_pose(cam_radius_min: float, cam_radius_max: float, look_at: np.ndarray, up: np.ndarray, only_front: bool = False):
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


def sample_camera_pose_xy(cam_radius_min: float, cam_radius_max: float, look_at: np.ndarray, up: np.ndarray, only_front: bool = False):
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


def spherical_to_cartesian(r, theta, phi):
    """Convert spherical coordinates to cartesian coordinates.
    Args:
        r (float): Radius.
        theta (float): Azimuthal angle in radians.
        phi (float): Polar angle in radians.
    Returns:
        np.array: Cartesian coordinates.
    """
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z])


def visualize_bbox(img, bbox, color=(0, 255, 0), thickness=2):
    """Visualize bounding box on image.
    Args:
        img (np.array): The image.
        bbox (List[int]): The bounding box.
        color (Tuple[int], optional): Color of the bounding box. Defaults to (0, 255, 0).
        thickness (int, optional): Thickness of the bounding box. Defaults to 2.
    Returns:
        np.array: The image with bounding box.
    """
    x, y, w, h = bbox
    x_min, x_max = x, x + w
    y_min, y_max = y, y + h
    vis_img = np.copy(img)
    cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    return vis_img


def render_parts(
    mesh_map,
    num_cam_poses,
    camera_info,
    cam_radius_max,
    cam_radius_min,
    image_idx_offset=0,
    only_front: bool = False,
    camera_sample_method="uniform",
    predefine_camera_poses=None,
    is_link_map: bool = True,
):
    """Render image from different camera poses & provide bbox for parts.
    Args:
        mesh_pose_dict (dict): Dictionary from mesh -> pose.
        num_cam_poses (int): Number of camera poses.
    Returns:
        List[Dict]: List of bounding boxes in COCO format.
    """
    scene = pyrender.Scene()

    # Add multiple lights
    for i in range(3):
        light_radius_min = 2.0
        light_radius_max = 2.5
        light_radius = np.random.uniform(low=light_radius_min, high=light_radius_max)
        light_pose = np.eye(4)
        if i == 0:
            light_pose[:3, 3] = np.array([-light_radius, 0.0, 0.0])
        elif i == 1:
            light_pose[:3, 3] = np.array([0.0, light_radius, 0.0])
        elif i == 2:
            light_pose[:3, 3] = np.array([0.0, 0.0, light_radius])
        light = pyrender.PointLight(color=np.ones(3), intensity=10.0)
        scene.add(light, pose=light_pose)

    mesh_name_dict = {}
    for mesh, (pose, name) in mesh_map.items():
        pymesh = pyrender.Mesh.from_trimesh(mesh)
        pymesh.name = name
        if is_link_map:
            # Use a random color for the link mesh
            color = np.random.rand(3)
            pymesh.primitives[0].material.baseColorFactor = color
        mesh_node = scene.add(pymesh, pose=pose)

        # Compute center
        mesh_pose = mesh_node.matrix
        center_3d = np.mean(mesh.vertices, axis=0)
        center_3d = np.dot(mesh_pose, np.append(center_3d, 1))[:3]
        mesh_name_dict[name] = center_3d

    fx = camera_info["fx"]
    fy = camera_info["fy"]
    cx = camera_info["cx"]
    cy = camera_info["cy"]
    width = camera_info["width"]
    height = camera_info["height"]
    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy)
    r = pyrender.OffscreenRenderer(width, height)

    # Compute the bounding sphere of the scene
    center = np.mean([mesh.centroid for mesh in mesh_map.keys()], axis=0)
    radius = np.max([np.max(np.linalg.norm(mesh.vertices - center, axis=1)) for mesh in mesh_map.keys()])
    cam_radius_min = radius * cam_radius_min  # Minimum distance from the look-at point
    cam_radius_max = radius * cam_radius_max  # Maximum distance from the look-at point

    # Radius of the sphere
    annotations = []
    camera_poses = []
    camera_node = None
    color_imgs = []
    depth_imgs = []
    mask_imgs = []
    for img_idx in range(num_cam_poses):
        # Randomly select an object from mesh_dict
        mesh_node = np.random.choice(list(scene.mesh_nodes))
        # Look at the center of the object
        look_at = np.array([0.0, 0.0, 0.0])
        look_at += np.random.normal(scale=0.1 * radius, size=3)
        up = np.array([0.0, 0.0, 1.0])
        # Apply a small disturbance to the up vector
        up += np.random.normal(scale=0.07 * np.pi, size=3)
        up /= np.linalg.norm(up)
        if predefine_camera_poses is not None:
            camera_pose = predefine_camera_poses[image_idx_offset + img_idx]
        else:
            if camera_sample_method == "uniform":
                # Compute the camera pose
                camera_pose = sample_camera_pose(cam_radius_min, cam_radius_max, look_at, up, only_front=only_front)
            elif camera_sample_method == "xy":
                camera_pose = sample_camera_pose_xy(cam_radius_min, cam_radius_max, look_at, up, only_front=only_front)
            else:
                raise NotImplementedError
        if type(camera_pose) == np.ndarray:
            camera_pose = camera_pose.tolist()
        camera_poses.append(camera_pose)
        if camera_node is not None:
            scene.remove_node(camera_node)
        camera_node = scene.add(camera, pose=np.array(camera_pose))

        # Render color image
        flags = pyrender.RenderFlags.RGBA
        color, depth = r.render(scene, flags=flags)
        color_imgs.append(color)

        if not is_link_map:
            continue

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
                contours, _ = cv2.findContours(mask_vis.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
                    "bbox": [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
                    "rot_bbox": [rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]],
                    "area": area,
                    "vis_ratio": vis_ratio,
                    "center_3d": center_3d.tolist(),
                    "image_id": img_idx + image_idx_offset,
                    "id": link_idx,
                    "name": node.mesh.name,
                    "camera_pose": camera_pose,
                }
                annotations.append(annotation)

        # Show all meshes again
        for mn in scene.mesh_nodes:
            mn.mesh.is_visible = True

        # DEBUG: check img
        # cv2.imshow("seg_img", mask_img * 30)
        # cv2.waitKey(0)
        mask_imgs.append(mask_img)
    r.delete()

    return annotations, camera_poses, color_imgs, depth_imgs, mask_imgs
