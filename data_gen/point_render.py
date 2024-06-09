"""Generate point cloud data"""

import argparse
import numpy as np

EPS = 1e-6
import os
from shapely.geometry import MultiPoint
import time
import shutil
import json
import cv2
from scipy.spatial import KDTree
from copy import deepcopy
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
from matplotlib.patches import Polygon as MplPolygon


############################### DEBUG ########################################
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

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, bbox, origin])


############################### BBOX related ########################################
class BBox3D:
    """3D bounding box tool."""

    def __init__(self, center=None, extent=None, rot_vec=None) -> None:
        self.extent = np.ones(3) if extent is None else np.array(extent)
        self.center = np.zeros(3) if center is None else np.array(center)
        self.R = np.eye(3) if rot_vec is None else R.from_rotvec(rot_vec).as_matrix()

    def create_axis_aligned_from_points(self, points):
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        self.center = (min_bound + max_bound) / 2
        self.extent = max_bound - min_bound
        self.R = np.eye(3)

    def create_minimum_axis_aligned_bbox(self, points):
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        obb = pcd.get_minimal_oriented_bounding_box()
        self.center = np.asarray(obb.center)
        self.extent = np.asarray(obb.extent)
        self.R = np.asarray(obb.R)

    def create_minium_projected_bbox(self, points):
        points_xy = points[:, :2]
        # Minimum bounding box in 2D
        multipoint = MultiPoint(points_xy)
        min_rect = multipoint.minimum_rotated_rectangle
        rect_coords = list(min_rect.exterior.coords)
        rect_coords = np.array(rect_coords)[:, :2]
        edges = [rect_coords[i + 1] - rect_coords[i] for i in range(len(rect_coords) - 1)]
        longest_edge = max(edges, key=lambda x: np.linalg.norm(x))  # Use this as x-axis
        shortest_edge = min(edges, key=lambda x: np.linalg.norm(x))
        longest_edge_len = np.linalg.norm(longest_edge)
        shortest_edge_len = np.linalg.norm(shortest_edge)
        center_xy = np.mean(rect_coords[:4, :], axis=0)
        min_z = np.min(points[:, 2])
        max_z = np.max(points[:, 2])
        center = np.array([center_xy[0], center_xy[1], (min_z + max_z) / 2])
        x_axis = np.array([longest_edge[0], longest_edge[1], 0])
        z_axis = np.array([0, 0, max_z - min_z])
        x_axis = x_axis / (np.linalg.norm(x_axis) + EPS)
        z_axis = z_axis / (np.linalg.norm(z_axis) + EPS)
        y_axis = np.cross(z_axis, x_axis)

        if (longest_edge_len - shortest_edge_len) / (shortest_edge_len + EPS) < 0.1:
            # Could be a circle
            min_bound = np.min(points, axis=0)
            max_bound = np.max(points, axis=0)
            axis_aligned_extent = max_bound - min_bound
            longest_edge_len_aa = np.max(axis_aligned_extent[:2])
            shortest_edge_len_aa = np.min(axis_aligned_extent[:2])
            if (np.abs(longest_edge_len_aa - longest_edge_len) / (longest_edge_len + EPS) < 0.1) and (np.abs(shortest_edge_len_aa - shortest_edge_len) / (shortest_edge_len + EPS) < 0.1):
                # aa box is similar to box
                return self.create_axis_aligned_from_points(points)

        self.center = np.array(center)
        self.extent = np.array([longest_edge_len, shortest_edge_len, max_z - min_z])
        self.R = np.array([x_axis, y_axis, z_axis]).T
        # # [Debug] Plot 2D
        # # Creating the plot
        # fig, ax = plt.subplots()
        # # Plot points
        # x, y = points[:, 0], points[:, 1]
        # ax.scatter(x, y, label="Points", s=1)
        # # Plot the minimum bounding box
        # rect_patch = MplPolygon(rect_coords, closed=True, edgecolor="red", fill=False)
        # ax.add_patch(rect_patch)
        # # Set equal scaling and labels
        # ax.set_aspect("equal")
        # ax.legend()
        # plt.title("Minimum Bounding Box and Points")
        # plt.xlabel("X coordinate")
        # plt.ylabel("Y coordinate")
        # plt.savefig("min_bbox.png")
        # plt.close()

    def get_min_bound(self):
        return self.center - self.extent / 2

    def get_max_bound(self):
        return self.center + self.extent / 2

    def rotate(self, R, center=np.array([0, 0, 0])):
        self.center = R @ (self.center - center) + center
        self.R = R @ self.R

    def translate(self, T):
        self.center += T

    def transform(self, T):
        self.center = T[:3, :3] @ self.center + T[:3, 3]
        self.R = T[:3, :3] @ self.R

    def get_points(self):
        bbox_R = self.R
        x_axis = np.dot(bbox_R, np.array([self.extent[0] / 2, 0, 0]))
        y_axis = np.dot(bbox_R, np.array([0, self.extent[1] / 2, 0]))
        z_axis = np.dot(bbox_R, np.array([0, 0, self.extent[2] / 2]))

        points = np.zeros((8, 3))
        points[0] = self.center - x_axis - y_axis - z_axis
        points[1] = self.center + x_axis - y_axis - z_axis
        points[2] = self.center - x_axis + y_axis - z_axis
        points[3] = self.center - x_axis - y_axis + z_axis
        points[4] = self.center + x_axis + y_axis + z_axis
        points[5] = self.center - x_axis + y_axis + z_axis
        points[6] = self.center + x_axis - y_axis + z_axis
        points[7] = self.center + x_axis + y_axis - z_axis
        return points

    def get_array(self):
        return np.concatenate([self.center, self.extent, R.from_matrix(self.R).as_rotvec()])

    def get_pose(self):
        pose = np.eye(4)
        pose[:3, :3] = self.R
        pose[:3, 3] = self.center
        return pose

    ########## Annotation tools ##########
    def get_bbox_3d_proj(self, intrinsics, camera_pose, depth_min, depth_max, img_width, img_height):
        """BBox 3d projected to pixel space."""
        points = self.get_points()
        points_cam = points @ camera_pose[:3, :3].T + camera_pose[:3, 3]
        points_pixel = []
        for point_3d in points_cam:
            point_2d = [-point_3d[0] / point_3d[2], point_3d[1] / point_3d[2]]
            pixel_x = (point_2d[0] * intrinsics[0, 0] + intrinsics[0, 2]) / img_width
            pixel_y = (point_2d[1] * intrinsics[1, 1] + intrinsics[1, 2]) / img_height
            pixel_z = (np.abs(point_3d[2]) - depth_min) / (depth_max - depth_min + 1e-6)
            points_pixel.append([pixel_x, pixel_y, pixel_z])
        # Clip to [0, 1]
        points_pixel = np.clip(points_pixel, 0, 1)
        return np.array(points_pixel)

    @staticmethod
    def project_points(points, intrinsics, camera_pose, depth_min, depth_max, img_width, img_height):
        proj_points = []
        for point in points:
            point_cam = point @ camera_pose[:3, :3].T + camera_pose[:3, 3]
            point_2d = [-point_cam[0] / point_cam[2], point_cam[1] / point_cam[2]]
            pixel_x = (point_2d[0] * intrinsics[0, 0] + intrinsics[0, 2]) / img_width
            pixel_y = (point_2d[1] * intrinsics[1, 1] + intrinsics[1, 2]) / img_height
            pixel_z = (np.abs(point_cam[2]) - depth_min) / (depth_max - depth_min + 1e-6)
            pixel = np.array([pixel_x, pixel_y, pixel_z])
            proj_points.append(pixel)
        proj_points = np.clip(proj_points, 0, 1)
        return np.array(proj_points)


############################### Point cloud related ########################################
def farthest_point_sample(point, npoint):
    """
    Farthest point sampling algorithm to sample points from a 3D point cloud.

    Args:
        point (numpy.ndarray): The point cloud data, shape [N, D] where D >= 3.
        npoint (int): The number of samples to be selected.

    Returns:
        numpy.ndarray: The sampled point cloud subset, shape [npoint, D].
    """
    assert npoint <= point.shape[0], "npoint should be less than or equal to the number of points in the cloud"

    N, D = point.shape
    centroids = np.zeros(npoint, dtype=int)
    distance = np.full(N, np.inf)  # Initialize distances with infinity
    farthest = random.randint(0, N - 1)  # Start with a random point

    for i in range(npoint):
        centroids[i] = farthest
        centroid_point = point[farthest, :3]  # Assuming points are 3D in the first three dimensions
        dist = np.sum((point[:, :3] - centroid_point) ** 2, axis=1)
        # Update the distance array to maintain the nearest distance to any selected centroid
        distance = np.minimum(distance, dist)
        # Select the farthest point based on updated distances
        farthest = np.argmax(distance)

    # Gather sampled points using the indices in centroids
    sampled_points = point[centroids.astype(int)]

    return sampled_points


def save_point_cloud_with_normals_to_npy(filename, points, normals, colors=None, sample=False):
    assert len(points) == len(normals), "Points and normals arrays must have the same length"
    if colors is not None:
        assert len(points) == len(colors), "Points and colors arrays must have the same length"
    if colors is not None:
        points = np.hstack((points, normals, colors))
    else:
        points = np.hstack((points, normals))

    if sample and points.shape[0] > 8192:
        points = farthest_point_sample(points, 8192)

    np.save(filename, points)


def save_point_cloud_with_normals_to_ply(filename, points, normals, colors=None):
    assert len(points) == len(normals), "Points and normals arrays must have the same length"
    if colors is not None:
        assert len(points) == len(colors), "Points and colors arrays must have the same length"

    with open(filename, "w") as file:
        # Write the PLY header
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property float nx\n")
        file.write("property float ny\n")
        file.write("property float nz\n")
        if colors is not None:
            file.write("property uchar red\n")
            file.write("property uchar green\n")
            file.write("property uchar blue\n")
        file.write("end_header\n")

        # Write point data with normals (and colors if provided)
        for i in range(len(points)):
            point_str = " ".join(f"{coord:.6f}" for coord in points[i])
            normal_str = " ".join(f"{norm:.6f}" for norm in normals[i])
            line = f"{point_str} {normal_str}"
            if colors is not None:
                color_str = " ".join(str(int(color)) for color in colors[i])
                line += f" {color_str}"
            file.write(line + "\n")

        file.flush()  # Explicitly flush to ensure all data is written before file is closed


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


def get_pointcloud(color, depth, mask, intrinsic, sample_size, flip_x: bool = False, flip_y: bool = False, enable_normal=False):
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


############################### Data augmentation ###############################
def jitter_brightness(colors, brightness_range=(0.8, 1.2)):
    """
    Apply brightness jittering to a list of colors.

    Args:
        colors: NumPy array of shape (n, 3), where n is the number of points and
                each row contains the RGB color values [0, 255] of a point.
        brightness_range: Tuple specifying the range (min, max) to randomly select
                          a brightness factor.

    Returns:
        jittered_colors: NumPy array of jittered colors with the same shape as input.
    """
    # Randomly choose a brightness factor
    brightness_factor = np.random.uniform(brightness_range[0], brightness_range[1])
    # Apply brightness jittering
    jittered_colors = colors * brightness_factor
    # Clip to valid range and convert back to uint8
    jittered_colors = np.clip(jittered_colors, 0, 255)

    return jittered_colors


############################### Visualization in Cluster ###############################
def read_ply(filename):
    """Read a PLY file and return points, normals, and colors."""
    with open(filename, "r") as f:
        # Skip the header by finding the 'end_header' line
        while True:
            line = f.readline().strip()
            if line == "end_header":
                break

        # Read the rest of the file into arrays
        points = []
        normals = []
        colors = []
        for line in f:
            split_line = line.split()
            # Convert each line to floats or ints as appropriate
            point = list(map(float, split_line[0:3]))
            normal = list(map(float, split_line[3:6]))
            color = list(map(float, split_line[6:9]))
            color = [int(c) for c in color]
            points.append(point)
            normals.append(normal)
            colors.append(color)

    return np.array(points), np.array(normals), np.array(colors)


def visualize_point_cloud(file_name, output_image):
    """Visualize the point cloud."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if ".ply" in file_name:
        points, normals, colors = read_ply(file_name)
    elif ".npy" in file_name:
        points_normals_coldrs = np.load(file_name)
        points = points_normals_coldrs[:, :3]
        colors = points_normals_coldrs[:, 6:9]
    else:
        raise ValueError("Unsupported file format")
    # Scatter plot
    if colors is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors / 255, s=1, edgecolor="none")
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, edgecolor="none")

    # Equal aspect ratio for all axes
    ax.set_box_aspect([np.ptp(d) for d in points.T])  # PTP is the "peak to peak" (max - min) value

    # Labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect("equal")
    # Save the figure to an image file
    # ax.view_init(elev=90, azim=-90)
    # Turn off the axis
    # ax.axis('off')
    plt.savefig(output_image, dpi=300)
    plt.close(fig)


def axis_bbox_intersect(origin, direction, outer_bbox):
    """Compute the intersections between a ray and a bounding box, considering both directions.

    Args:
        origin (tuple): The origin of the ray, as (x, y, z).
        direction (tuple): The direction of the ray, as (x, y, z).
        outer_bbox (tuple): The bounding box represented by its minimum and
                            maximum corners, ((min_x, min_y, min_z), (max_x, max_y, max_z)).

    Returns:
        A tuple containing a boolean indicating whether there is an intersection,
        and a list of intersection points (can be zero, one, or two points).
    """
    min_point, max_point = outer_bbox
    t_min = float("-inf")
    t_max = float("inf")

    for i in range(3):  # Iterate over each axis (x, y, z)
        if direction[i] != 0:
            t1 = (min_point[i] - origin[i]) / direction[i]
            t2 = (max_point[i] - origin[i]) / direction[i]

            t_min_temp = min(t1, t2)
            t_max_temp = max(t1, t2)

            t_min = max(t_min, t_min_temp)
            t_max = min(t_max, t_max_temp)
        elif origin[i] < min_point[i] or origin[i] > max_point[i]:
            return (False, [])  # The ray is parallel and outside the bbox

    if t_min > t_max or t_max < 0:
        return (False, [])  # There is no intersection or it's behind the origin

    intersections = []
    # If t_min is positive, it means the intersection is in front of the ray's origin.
    if t_min >= 0:
        intersection_entry = (origin[0] + t_min * direction[0], origin[1] + t_min * direction[1], origin[2] + t_min * direction[2])
        intersections.append(intersection_entry)

    # If t_max is different from t_min, it means there's another intersection point.
    if t_max != t_min and t_max >= 0:
        intersection_exit = (origin[0] + t_max * direction[0], origin[1] + t_max * direction[1], origin[2] + t_max * direction[2])
        intersections.append(intersection_exit)

    return (True, intersections) if intersections else (False, [])


def vector_fix(vector_raw):
    """Fix the vector to have a unit length."""
    vector = []
    for v in vector_raw:
        if v is None:
            vector.append(0)
        else:
            vector.append(v)
    return np.array(vector)


def generate_label_3d(points, colors, normals, masks, joint_info, semantic_data, camera_pose_inv, data_name):
    bbox_list = []
    label_3d_dict = {}
    for link_idx, link_data in enumerate(joint_info):
        if "jointData" in link_data and link_data["jointData"]:
            joint_type = semantic_data[link_idx]["joint_type"]
            if joint_type in ["fixed", "free", "heavy"]:
                continue
            axis_origin_raw = link_data["jointData"]["axis"]["origin"]
            axis_origin = vector_fix(axis_origin_raw)
            axis_direction_raw = link_data["jointData"]["axis"]["direction"]
            axis_direction = vector_fix(axis_direction_raw)
            # Normalize
            axis_direction = axis_direction / (np.linalg.norm(axis_direction) + EPS)
            # Convert y-up to z-up
            axis_origin = np.array([-axis_origin[2], -axis_origin[0], axis_origin[1]])
            axis_direction = np.array([-axis_direction[2], -axis_direction[0], axis_direction[1]])
            # Show axis (aligned)
            joint_coord_z = axis_direction
            joint_coord_x = np.array([1.0, 0.0, 0.0]) if np.abs(joint_coord_z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
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
                bbox = BBox3D()
                if mask_pcd.shape[0] >= 8:
                    # bbox.create_axis_aligned_from_points(mask_pcd)
                    # Have enough points to compute the minimum bbox
                    # bbox.create_minimum_axis_aligned_bbox(mask_pcd)
                    bbox.create_minium_projected_bbox(mask_pcd)
                else:
                    bbox.create_axis_aligned_from_points(mask_pcd)
                # Compute interaction points
                min_z = np.min(mask_pcd[:, 2])
                max_z = np.max(mask_pcd[:, 2])
                if joint_type == "slider":
                    bbox_center = np.array(bbox.center)
                    inter_points = np.array([[bbox_center[0], bbox_center[1], min_z], [bbox_center[0], bbox_center[1], max_z]])
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
                # check_annotations_o3d(points, bbox_rep)
                label_3d_dict[joint_id] = {"joint_T": joint_T.tolist(), "bbox_3d": bbox_rep.tolist(), "itp_points": inter_points.tolist(), "name": link_data["name"]}
    return label_3d_dict


def process_one_data(
    data_name,
    output_dir,
    use_world_coordinate,
    sample_size,
    gaussian_noise,
    random_drop_rate,
    save_label_3d,
    save_visualization,
    enable_normal=False,
    use_sd_image=False,
    enable_aug=False,
    export_ply=False,
):
    output_dir = os.path.join(output_dir, data_name)
    label_3d_dict_json = os.path.join(output_dir, f"annotations_3d.json")

    # if os.path.exists(pointcloud_dir) and len(os.listdir(pointcloud_dir)) > 10 and os.path.exists(label_3d_dict_json):
    #     print(f"Skip {data_name} since already processed")
    #     return True
    if not use_sd_image:
        color_dir = os.path.join(output_dir, "raw_images")
        pointcloud_dir = os.path.join(output_dir, "pointclouds")
    else:
        color_dir = os.path.join(output_dir, "controlnet_images_seg")
        if not os.path.exists(color_dir):
            color_dir = os.path.join(output_dir, "controlnet_images")
        pointcloud_dir = os.path.join(output_dir, "pointclouds_sd")

    depth_dir = os.path.join(output_dir, "real_depth_images")
    mask_dir = os.path.join(output_dir, "mask")
    semantic_file = os.path.join(output_dir, "semantics.txt")

    if not os.path.exists(depth_dir) or not os.path.exists(color_dir) or not os.path.exists(mask_dir) or not os.path.exists(semantic_file):
        print(f"Skip {data_name} since not all files exist")
        return False

    if os.path.exists(pointcloud_dir):
        shutil.rmtree(pointcloud_dir)
        os.makedirs(pointcloud_dir)
    else:
        os.makedirs(pointcloud_dir)

    # remove the old npy folder to save space
    if not use_sd_image:
        npy_folder = os.path.join(output_dir, "npy_8192")
        visual_path = os.path.join(output_dir, "pc_visualization")
    else:
        npy_folder = os.path.join(output_dir, "npy_8192_sd")
        visual_path = os.path.join(output_dir, "pc_visualization_sd")

    if os.path.exists(npy_folder):
        shutil.rmtree(npy_folder)
        os.makedirs(npy_folder)
    else:
        os.makedirs(npy_folder)

    # Prepare directory
    try:
        semantic_data = []
        with open(semantic_file, "r") as file:
            for line in file:
                parts = line.strip().split(" ")
                if len(parts) == 3:
                    semantic_data.append({"link_name": parts[0], "joint_type": parts[1], "semantic": parts[2]})
        joint_info_file = os.path.join(output_dir, "mobility_v2.json")
        info = json.load(open(os.path.join(output_dir, "info.json")))
        if not os.path.exists(depth_dir) or not os.path.exists(color_dir) or not os.path.exists(joint_info_file):
            return False
        joint_info = json.load(open(joint_info_file))
        # Filterout junk data
        joint_info = [joint for joint in joint_info if joint["joint"] != "junk"]

        # Load camera poses
        cam_info = info["camera_info"]
        num_images = len(info["camera_poses"])
        cam_intrinsics = np.array([[cam_info["fx"], 0, cam_info["cx"]], [0, cam_info["fy"], cam_info["cy"]], [0, 0, 1]])
        label_3d_dicts = []
        for image_idx in range(num_images):
            if not use_sd_image:
                color = cv2.imread(os.path.join(color_dir, f"{image_idx:06}.png"))
            else:
                color_file_id = random.randint(0, 3)
                color_file_name = f"{image_idx}_{color_file_id}.png"
                color = cv2.imread(os.path.join(color_dir, color_file_name))
            color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            depth = cv2.imread(os.path.join(depth_dir, f"{image_idx:06}.png"), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
            mask = cv2.imread(os.path.join(mask_dir, f"{image_idx:06}.png"), cv2.IMREAD_UNCHANGED).astype(np.uint8)
            #
            # [DEBUG]: Show different masks
            new_mask = np.zeros_like(mask)  # mask is filled with real part id, replace it with joint id
            for mask_id in np.unique(mask):
                if mask_id == 0:
                    continue
                mask_i = mask == mask_id
                if np.sum(mask) > 0:
                    new_mask[mask_i] = int(joint_info[mask_id - 1]["id"])
                    # # # Save to images
                    # mask_image = np.zeros_like(mask)
                    # mask_image[mask_i] = 255
                    # mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
                    # cv2.imwrite(f"mask_{int(joint_info[mask_id - 1]['id'])}.png", mask_image)

            camera_pose = np.array(info["camera_poses"][image_idx]).reshape(4, 4)
            points, colors, normals, masks = get_pointcloud(color, -depth, new_mask, cam_intrinsics, sample_size, flip_x=True, enable_normal=enable_normal)
            # Apply noise
            points += np.random.normal(0, gaussian_noise, points.shape)
            normals += np.random.normal(0, gaussian_noise, normals.shape)
            colors = jitter_brightness(colors)
            if use_world_coordinate:
                # Assume floor is known, augmentaiton is appened to rot along z-axis
                disturbance = np.eye(4)
                if enable_aug:
                    random_rot_angle = np.random.uniform(-20, 20)
                    # Add more disturbance
                    disturbance[:3, 3] = np.random.uniform(-0.2, 0.2, 3)
                else:
                    random_rot_angle = 0
                disturbance[:3, :3] = R.from_euler("z", 90 + random_rot_angle, degrees=True).as_matrix()
                dist_camera_pose = disturbance @ camera_pose  # Sync with CAD3D coordinate
                points = points @ dist_camera_pose[:3, :3].T + dist_camera_pose[:3, 3]
                normals = normals @ dist_camera_pose[:3, :3].T
            else:
                disturbance = np.linalg.inv(camera_pose)

            # Generating 3D labels
            if save_label_3d:
                label_3d_dict = generate_label_3d(points, colors, normals, masks, joint_info, semantic_data, disturbance, data_name)
                label_3d_dict["meta"] = {}
                label_3d_dict["meta"]["disturbance"] = disturbance.tolist()

            prob = 0.1
            save_visualization = random.random() < prob  # Save 5% of the visualizations

            label_3d_dict["meta"]["camera_pose"] = camera_pose.tolist()
            label_3d_dicts.append(label_3d_dict)

            if export_ply:
                # Export point cloud
                save_point_cloud_with_normals_to_ply(os.path.join(pointcloud_dir, f"{image_idx:06}.ply"), points, normals, colors)
                # save to npy
                save_point_cloud_with_normals_to_npy(os.path.join(npy_folder, f"{image_idx:06}_8192.npy"), points, normals, colors, sample=True)

            if save_visualization and export_ply:
                # Save visualization
                if not use_sd_image:
                    visual_path = os.path.join(output_dir, "pc_visualization")
                else:
                    visual_path = os.path.join(output_dir, "pc_visualization_sd")
                if not os.path.exists(visual_path):
                    os.makedirs(visual_path)
                output_image = os.path.join(visual_path, f"{image_idx:06}.png")
                visualize_point_cloud(os.path.join(npy_folder, f"{image_idx:06}_8192.npy"), output_image)

        # Export label 3d
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
    argparser.add_argument("--version", type=int, default=0, help="Version of the dataset")
    argparser.add_argument("--data_dir", type=str, required=True)
    argparser.add_argument("--output_dir", type=str, required=True)
    argparser.add_argument("--data_name", type=str, default="all")
    argparser.add_argument("--gaussian_noise", type=float, default=0.00, help="Gaussian noise")
    argparser.add_argument("--random_drop_rate", type=float, default=0.3, help="Random drop rate")
    argparser.add_argument("--sample_size", type=int, default=32768, help="Numbe of points to sample")
    argparser.add_argument("--use_world_coordinate", type=bool, default=True, help="Use world coordinate")
    argparser.add_argument("--enable_normal", action="store_true", help="Enable normal")
    argparser.add_argument("--save_label_3d", action="store_true", help="Save label 3d")
    argparser.add_argument("--save_visualization", action="store_true", help="Save visualization")
    argparser.add_argument("--use_sd_image", action="store_true", help="Use sd image")
    argparser.add_argument("--enable_aug", type=bool, default=False, help="Enable augmentation")
    argparser.add_argument("--export_ply", type=bool, default=False, help="Export ply")
    args = argparser.parse_args()

    save_visualization = args.save_visualization
    save_label_3d = args.save_label_3d
    save_label_3d = True
    enable_normal = args.enable_normal
    sample_size = args.sample_size
    use_world_coordinate = args.use_world_coordinate
    data_dir = args.data_dir
    output_dir = args.output_dir
    gaussian_noise = args.gaussian_noise
    random_drop_rate = args.random_drop_rate
    use_sd_image = args.use_sd_image
    enable_aug = args.enable_aug
    export_ply = args.export_ply

    # Load data
    data_name = args.data_name
    if data_name != "all":
        process_one_data(
            data_name,
            output_dir,
            use_world_coordinate,
            sample_size,
            gaussian_noise,
            random_drop_rate,
            save_label_3d,
            save_visualization,
            enable_normal=enable_normal,
            use_sd_image=use_sd_image,
            enable_aug=enable_aug,
            export_ply=export_ply,
        )
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
            use_world_coordinate=use_world_coordinate,
            sample_size=sample_size,
            gaussian_noise=gaussian_noise,
            random_drop_rate=random_drop_rate,
            save_label_3d=save_label_3d,
            save_visualization=save_visualization,
            enable_normal=enable_normal,
            use_sd_image=use_sd_image,
            enable_aug=enable_aug,
            export_ply=export_ply,
        )

        workers = min(cpu_count(), len(data_names))
        with Pool(workers) as p:
            status = list(tqdm(p.imap(render_function, data_names), total=len(data_names)))
