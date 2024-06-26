"""Generate point cloud data v2. reading from npz data."""

import argparse
import numpy as np

EPS = 1e-6
import os
import time
import shutil
import json
import cv2
from scipy.spatial import KDTree
from copy import deepcopy
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from utils import AxisBBox3D


############################### Point cloud related ########################################
def check_annotations_o3d(points, bbox_3d):
    import open3d as o3d

    bbox_3d = np.array(bbox_3d)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.center = bbox_3d[0:3]
    bbox.R = R.from_rotvec(bbox_3d[6:9]).as_matrix()
    bbox.extent = bbox_3d[3:6]
    bbox.color = [1, 0, 0]

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=[0, 0, 0]
    )
    o3d.visualization.draw_geometries([pcd, bbox, origin])


def farthest_point_sample(point, npoint):
    """
    Farthest point sampling algorithm to sample points from a 3D point cloud.

    Args:
        point (numpy.ndarray): The point cloud data, shape [N, D] where D >= 3.
        npoint (int): The number of samples to be selected.

    Returns:
        numpy.ndarray: The sampled point cloud subset, shape [npoint, D].
    """
    assert (
        npoint <= point.shape[0]
    ), "npoint should be less than or equal to the number of points in the cloud"

    N, D = point.shape
    centroids = np.zeros(npoint, dtype=int)
    distance = np.full(N, np.inf)  # Initialize distances with infinity
    farthest = random.randint(0, N - 1)  # Start with a random point

    for i in range(npoint):
        centroids[i] = farthest
        centroid_point = point[
            farthest, :3
        ]  # Assuming points are 3D in the first three dimensions
        dist = np.sum((point[:, :3] - centroid_point) ** 2, axis=1)
        # Update the distance array to maintain the nearest distance to any selected centroid
        distance = np.minimum(distance, dist)
        # Select the farthest point based on updated distances
        farthest = np.argmax(distance)

    # Gather sampled points using the indices in centroids
    sampled_points = point[centroids.astype(int)]

    return sampled_points


def estimate_normal(point, neighbors):
    # Calculate the centroid of the neighborhood
    centroid = np.mean(neighbors, axis=0)
    # Center the neighborhood points around the centroid
    centered_points = neighbors - centroid
    # Use SVD to find the normals
    u, s, vh = np.linalg.svd(centered_points, full_matrices=True)
    # The normal vector is the last column of vh
    normal = vh[-1, :]
    return normal


def estimate_normals_for_cloud(points, k=10, camera_location=np.array([0.0, 0.0, 0.0])):
    # Build a KD-Tree
    tree = KDTree(points)
    normals = []
    for point in points:
        # Find k nearest neighbors (including the point itself)
        _, idx = tree.query(point, k=k + 1)
        # Extract the neighbor points
        neighbors = points[idx]
        # Estimate the normal for the current point
        normal = estimate_normal(point, neighbors)
        # Orient the normal towards the camera
        if np.dot(normal, point - camera_location) > 0:
            normal = -normal
        normals.append(normal)
    return np.array(normals)


def random_point_sampling(points, k):
    num_points = points.shape[0]
    selected_indices = np.random.choice(num_points, k, replace=False)
    return selected_indices


def get_pointcloud(
    color,
    depth,
    mask,
    intrinsic,
    sample_size,
    flip_x: bool = False,
    flip_y: bool = False,
    enable_normal=False,
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
    if points.shape[0] > sample_size:
        indices = random_point_sampling(points, sample_size)
        points = points[indices]
        colors = colors[indices]
        masks = masks[indices]

    # Compute normals
    if enable_normal:
        normals = estimate_normals_for_cloud(points, k=10)
    else:
        normals = np.zeros_like(points)
    # return pcd_with_color
    return points, colors, normals, masks


############################### Label Utils ###############################
def vector_fix(vector_raw):
    """Fix the vector to have a unit length."""
    vector = []
    for v in vector_raw:
        if v is None:
            vector.append(0)
        else:
            vector.append(v)
    return np.array(vector)


def generate_label_3d(
    points,
    masks,
    joint_info,
    semantic_data,
    camera_pose_inv,
    obj_transform,
    data_name,
):
    bbox_list = []
    label_3d_dict = {}
    for link_idx, link_data in enumerate(joint_info):
        if "jointData" in link_data and link_data["jointData"]:
            joint_type = semantic_data[link_idx]["joint_type"]
            if joint_type in ["fixed", "free", "heavy"]:
                continue
            axis_origin_raw = link_data["jointData"]["axis"]["origin"]
            axis_direction_raw = link_data["jointData"]["axis"]["direction"]
            axis_origin = vector_fix(axis_origin_raw)
            axis_direction = vector_fix(axis_direction_raw)
            # Apply object transformation
            axis_origin = obj_transform[:3, :3] @ axis_origin + obj_transform[:3, 3]
            axis_direction = obj_transform[:3, :3] @ axis_direction
            # Normalize
            axis_direction = axis_direction / (np.linalg.norm(axis_direction) + EPS)
            # Convert y-up to z-up
            axis_origin = np.array([-axis_origin[2], -axis_origin[0], axis_origin[1]])
            axis_direction = np.array(
                [-axis_direction[2], -axis_direction[0], axis_direction[1]]
            )
            # Show axis (aligned)
            joint_coord_z = axis_direction
            joint_coord_x = (
                np.array([1.0, 0.0, 0.0])
                if np.abs(joint_coord_z[0]) < 0.9
                else np.array([0.0, 1.0, 0.0])
            )
            joint_coord_y = np.cross(joint_coord_z, joint_coord_x)
            try:
                joint_coord_y = joint_coord_y / (np.linalg.norm(joint_coord_y) + EPS)
                joint_coord_x = np.cross(joint_coord_y, joint_coord_z)
                joint_coord_x = joint_coord_x / (np.linalg.norm(joint_coord_x) + EPS)
                joint_R = np.array([joint_coord_x, joint_coord_y, joint_coord_z]).T
                joint_T = np.eye(4)
                joint_T[:3, :3] = joint_R
                joint_T[:3, 3] = axis_origin
                joint_T = camera_pose_inv @ joint_T
                joint_T_inv = np.linalg.inv(joint_T)
            except Exception as e:
                print(f"Error in {link_data['id']} of {data_name}")
                print(e)
                continue
            # Bbox
            joint_id = link_data["id"]
            pcd_id = np.where(masks == joint_id)[0]
            if len(pcd_id) > 0:
                mask_pcd = points[pcd_id]
                # Transform to joint coordinate
                mask_pcd = mask_pcd @ joint_T_inv[:3, :3].T + joint_T_inv[:3, 3]
                ##################### axis aligned bbox #####################
                bbox = AxisBBox3D()
                if mask_pcd.shape[0] >= 8:
                    # Have enough points to compute the minimum bbox
                    bbox.create_minimum_projected_bbox(mask_pcd)
                else:
                    bbox.create_axis_aligned_from_points(mask_pcd)
                # Compute interaction points
                min_z = np.min(mask_pcd[:, 2])
                max_z = np.max(mask_pcd[:, 2])
                if joint_type == "slider":
                    bbox_center = np.array(bbox.center)
                    inter_points = np.array(
                        [
                            [bbox_center[0], bbox_center[1], min_z],
                            [bbox_center[0], bbox_center[1], max_z],
                        ]
                    )
                else:
                    inter_points = np.array([[0, 0, min_z], [0, 0, max_z]])
                inter_points = inter_points @ joint_T[:3, :3].T + joint_T[:3, 3]
                bbox.rotate(joint_T[:3, :3], (0, 0, 0))
                bbox.translate(joint_T[:3, 3])
                bbox.color = (1, 0, 0)
                bbox_list.append(deepcopy(bbox))
                # Represent bbox_3d as: center, size, rotation
                bbox_center = np.array(bbox.center)
                bbox_size = np.array(bbox.extent)
                bbox_rotation = R.from_matrix(np.array(bbox.R)).as_rotvec()
                bbox_rep = np.concatenate([bbox_center, bbox_size, bbox_rotation])
                # DEBUG
                check_annotations_o3d(points, bbox_rep)
                label_3d_dict[joint_id] = {
                    "joint_T": joint_T.tolist(),
                    "bbox_3d": bbox_rep.tolist(),
                    "itp_points": inter_points.tolist(),
                    "name": link_data["name"],
                }
    return label_3d_dict


def process_one_data(data_name, output_dir, sample_size, save_label_3d):
    output_dir = os.path.join(output_dir, data_name)
    # Load data
    semantic_file = os.path.join(output_dir, "semantics.txt")
    joint_info_file = os.path.join(output_dir, "mobility_v2.json")
    if not os.path.exists(semantic_file) or not os.path.exists(joint_info_file):
        print(f"Skip {data_name} since not all files exist")
        return False
    color_images = np.load(os.path.join(output_dir, "color_imgs.npz"))["images"]
    depth_images = np.load(os.path.join(output_dir, "depth_imgs.npz"))["images"]
    mask_images = np.load(os.path.join(output_dir, "mask_imgs.npz"))["images"]
    info = json.load(open(os.path.join(output_dir, "info.json")))
    joint_info = json.load(open(joint_info_file))
    # Filterout junk data
    joint_info = [joint for joint in joint_info if joint["joint"] != "junk"]
    # Prepare directory
    try:
        semantic_data = []
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

        # Load camera poses
        cam_info = info["camera_info"]
        num_images = len(info["camera_poses"])
        cam_intrinsics = np.array(
            [
                [cam_info["fx"], 0, cam_info["cx"]],
                [0, cam_info["fy"], cam_info["cy"]],
                [0, 0, 1],
            ]
        )
        label_3d_dicts = []
        for image_idx in range(num_images):
            color = color_images[image_idx]
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            depth = depth_images[image_idx].astype(np.float32) / 1000.0
            mask = mask_images[image_idx]

            # Replace mask with joint id
            new_mask = np.zeros_like(mask)
            for mask_id in np.unique(mask):
                if mask_id == 0:
                    continue
                mask_i = mask == mask_id
                if np.sum(mask) > 0:
                    new_mask[mask_i] = int(joint_info[mask_id - 1]["id"])
                    # [DEBUG]
                    # mask_image = np.zeros_like(mask)
                    # mask_image[mask_i] = 255
                    # mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
                    # cv2.imshow("mask", mask_image)
                    # cv2.waitKey(0)
                    # cv2.imwrite(f"mask_{int(joint_info[mask_id - 1]['id'])}.png", mask_image)

            camera_pose = np.array(info["camera_poses"][image_idx]).reshape(4, 4)
            obj_transform = np.array(info["obj_transforms"][image_idx]).reshape(4, 4)
            points, colors, normals, masks = get_pointcloud(
                color,
                -depth,
                new_mask,
                cam_intrinsics,
                sample_size,
                flip_x=True,
                enable_normal=False,
            )
            # [DEBUG] Show point cloud
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            o3d.visualization.draw_geometries([pcd])

            # Generating 3D labels
            camera_pose_inv = np.linalg.inv(camera_pose)
            if save_label_3d:
                label_3d_dict = generate_label_3d(
                    points,
                    masks,
                    joint_info,
                    semantic_data,
                    camera_pose_inv,
                    obj_transform,
                    data_name,
                )
                label_3d_dict["meta"] = {}
                label_3d_dict["meta"]["disturbance"] = camera_pose_inv.tolist()

            label_3d_dict["meta"]["camera_pose"] = camera_pose.tolist()
            label_3d_dicts.append(label_3d_dict)

        # Export label 3d
        label_3d_dict_json = os.path.join(output_dir, f"annotations_3d.json")
        with open(label_3d_dict_json, "w") as f:
            json.dump(label_3d_dicts, f)
        return True
    except Exception as e:
        print(f"Error in {data_name}")
        print(e)
        return False


if __name__ == "__main__":
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=0, help="Random seed")
    argparser.add_argument("--data_dir", type=str, required=True)
    argparser.add_argument("--output_dir", type=str, required=True)
    argparser.add_argument("--data_name", type=str, default="920")
    argparser.add_argument(
        "--sample_size", type=int, default=32768, help="Numbe of points to sample"
    )
    argparser.add_argument("--save_label_3d", type=bool, default=True)
    args = argparser.parse_args()

    save_label_3d = args.save_label_3d
    sample_size = args.sample_size
    data_dir = args.data_dir
    output_dir = args.output_dir

    # Load data
    data_name = args.data_name
    if data_name != "all":
        process_one_data(data_name, output_dir, sample_size, save_label_3d)
    else:
        from functools import partial
        from multiprocessing import Pool, cpu_count
        from tqdm import tqdm

        data_names = sorted(os.listdir(data_dir))
        data_names = [x for x in data_names if os.path.isdir(os.path.join(data_dir, x))]

        print(f"Generating point clouds for {len(data_names)} datasets")
        # the data indexing specified
        render_function = partial(
            process_one_data,
            output_dir=output_dir,
            sample_size=sample_size,
            save_label_3d=save_label_3d,
        )

        workers = min(cpu_count(), len(data_names))
        with Pool(workers) as p:
            status = list(
                tqdm(p.imap(render_function, data_names), total=len(data_names))
            )
