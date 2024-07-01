"""Joint labler Version: Provide labeling for 3d & 2d tasks"""

import json
import math
import os
import cv2
import numpy as np
import argparse
from skimage.draw import line
from utils import calculate_iou, get_rotated_box, colors, draw_rotating_bbox
from vqa_config import open_close_status, joint_types_mapping, HOLDOUT_CLASSES
from vqa_task_construction import (
    create_single_link_rec_rotated_task,
    create_det_all_rotated_task,
    create_rec_joint_rotated_task,
    create_reg_status_qa_task,
    create_grounding_task,
    create_3d_grounding_task,
    create_single_link_3d_rec_task,
    create_3d_rec_joint_task,
    create_det_all_bbox_3d_task,
    create_3d_reg_status_qa_task,
    create_rec_joint_rotated_ext_task,
    create_rec_joint_rotated_ext_task_with_sep_depth,
)
import xml.etree.ElementTree as ET
from utils import read_ply_ascii, convert_depth_to_color
from scipy.spatial.transform import Rotation as R
from point_render import BBox3D, farthest_point_sample

# import urdfpy
from tqdm import tqdm
import logging
import random
import copy

import multiprocessing

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


################################# Utils #################################
def save_annotations(
    annotations,
    task_folder,
    cato=None,
):
    print(f"Saving annotations for split {cato} to {task_folder}")
    ###################################### Save 2D tasks ######################################
    single_link_rec_tasks = annotations["single_link_rec_tasks"]
    all_parts_det_tasks = annotations["all_parts_det_tasks"]
    joint_rec_tasks = annotations["joint_rec_tasks"]
    joint_rec_ext_tasks = annotations["joint_rec_ext_tasks"]
    joint_rec_sep_depth_tasks = annotations["joint_rec_sep_depth_tasks"]
    status_joint_reg_tasks = annotations["status_joint_reg_tasks"]
    grounding_tasks = annotations["grounding_tasks"]

    # Save single_link_rec_tasks
    single_link_rec_tasks_filename = os.path.join(task_folder, f"single_link_rec_tasks_{cato}_{len(single_link_rec_tasks)}.json")
    if len(single_link_rec_tasks) > 0:
        with open(single_link_rec_tasks_filename, "w") as f:
            json.dump(single_link_rec_tasks, f)

    # Save all_parts_det_tasks
    all_parts_det_tasks_filename = os.path.join(task_folder, f"all_parts_det_tasks_{cato}_{len(all_parts_det_tasks)}.json")
    if len(all_parts_det_tasks) > 0:
        with open(all_parts_det_tasks_filename, "w") as f:
            json.dump(all_parts_det_tasks, f)

    # Save joint_rec_tasks
    joint_rec_tasks_filename = os.path.join(task_folder, f"joint_rec_tasks_{cato}_{len(joint_rec_tasks)}.json")
    if len(joint_rec_tasks) > 0:
        with open(joint_rec_tasks_filename, "w") as f:
            json.dump(joint_rec_tasks, f)

    joint_rec_ext_tasks_filename = os.path.join(task_folder, f"joint_rec_ext_tasks_{cato}_{len(joint_rec_ext_tasks)}.json")
    if len(joint_rec_ext_tasks) > 0:
        with open(joint_rec_ext_tasks_filename, "w") as f:
            json.dump(joint_rec_ext_tasks, f)

    joint_rec_sep_depth_tasks_filename = os.path.join(task_folder, f"joint_rec_sep_depth_tasks_{cato}_{len(joint_rec_sep_depth_tasks)}.json")
    if len(joint_rec_sep_depth_tasks) > 0:
        with open(joint_rec_sep_depth_tasks_filename, "w") as f:
            json.dump(joint_rec_sep_depth_tasks, f)

    # Save status_joint_reg_tasks
    status_joint_reg_tasks_filename = os.path.join(task_folder, f"status_joint_reg_tasks_{cato}_{len(status_joint_reg_tasks)}.json")
    if len(status_joint_reg_tasks) > 0:
        with open(status_joint_reg_tasks_filename, "w") as f:
            json.dump(status_joint_reg_tasks, f)

    # Save grounding_tasks
    grounding_tasks_filename = os.path.join(task_folder, f"grounding_tasks_{cato}_{len(grounding_tasks)}.json")
    if len(grounding_tasks) > 0:
        with open(grounding_tasks_filename, "w") as f:
            json.dump(grounding_tasks, f)

    ###################################### Save 3D tasks ######################################
    single_link_3d_rec_tasks = annotations["single_link_3d_rec_tasks"]
    all_parts_3d_det_tasks = annotations["all_parts_3d_det_tasks"]
    joint_3d_rec_tasks = annotations["joint_3d_rec_tasks"]
    status_joint_3d_reg_tasks = annotations["status_joint_3d_reg_tasks"]
    grounding_3d_tasks = annotations["grounding_3d_tasks"]

    # Save single_link_3d_rec_tasks
    single_link_3d_rec_tasks_filename = os.path.join(task_folder, f"single_link_3d_rec_tasks_{cato}_{len(single_link_3d_rec_tasks)}.json")
    if len(single_link_3d_rec_tasks) > 0:
        with open(single_link_3d_rec_tasks_filename, "w") as f:
            json.dump(single_link_3d_rec_tasks, f)

    # Save all_parts_3d_det_tasks
    all_parts_3d_det_tasks_filename = os.path.join(task_folder, f"all_parts_3d_det_tasks_{cato}_{len(all_parts_3d_det_tasks)}.json")
    if len(all_parts_3d_det_tasks) > 0:
        with open(all_parts_3d_det_tasks_filename, "w") as f:
            json.dump(all_parts_3d_det_tasks, f)

    # Save joint_3d_rec_tasks
    joint_3d_rec_tasks_filename = os.path.join(task_folder, f"joint_3d_rec_tasks_{cato}_{len(joint_3d_rec_tasks)}.json")
    if len(joint_3d_rec_tasks) > 0:
        with open(joint_3d_rec_tasks_filename, "w") as f:
            json.dump(joint_3d_rec_tasks, f)

    # Save status_joint_3d_reg_tasks
    status_joint_3d_reg_tasks_filename = os.path.join(task_folder, f"status_joint_3d_reg_tasks_{cato}_{len(status_joint_3d_reg_tasks)}.json")
    if len(status_joint_3d_reg_tasks) > 0:
        with open(status_joint_3d_reg_tasks_filename, "w") as f:
            json.dump(status_joint_3d_reg_tasks, f)

    # Save grounding_3d_tasks
    grounding_3d_tasks_filename = os.path.join(task_folder, f"grounding_3d_tasks_{cato}_{len(grounding_3d_tasks)}.json")
    if len(grounding_3d_tasks) > 0:
        with open(grounding_3d_tasks_filename, "w") as f:
            json.dump(grounding_3d_tasks, f)


def normalize_and_round_angle(theta, granularity=5, range_start=0, range_end=360):
    # Normalize theta to be within [range_start, range_end)
    theta_normalized = (theta - range_start) % (range_end - range_start) + range_start
    # Round theta to the nearest granularity
    rounded_angle = round(theta_normalized / granularity) * granularity
    # Make sure the rounded angle is still within the range
    if rounded_angle == range_end:
        rounded_angle = range_start
    return rounded_angle / 180 * np.pi


def annotation_visualization(vis_image, bbox_3d_points_cam, axis_points_3d_cam, intrinsics, parent_rot_bbox, child_rot_bbox, axis_rot_bbox, closed, vis_image_file, only_save_image=True):
    vis_image_3d = vis_image.copy()
    vis_image_2d = vis_image.copy()
    ####################################### 3D Visualization #######################################
    # Project 3D bounding box to 2D
    points_2d = []
    for point in bbox_3d_points_cam:
        point = [-point[0] / point[2], point[1] / point[2]]
        pixel_x = point[0] * intrinsics[0, 0] + intrinsics[0, 2]
        pixel_y = point[1] * intrinsics[1, 1] + intrinsics[1, 2]
        points_2d.append([pixel_x, pixel_y])
    points_2d = np.array(points_2d, dtype=np.int32)
    # Draw 3D bounding box: 8 points
    vis_image_3d = cv2.line(vis_image_3d, tuple(points_2d[0]), tuple(points_2d[1]), (0, 0, 255), 2)
    vis_image_3d = cv2.line(vis_image_3d, tuple(points_2d[0]), tuple(points_2d[2]), (0, 255, 0), 2)
    vis_image_3d = cv2.line(vis_image_3d, tuple(points_2d[0]), tuple(points_2d[3]), (255, 0, 0), 2)
    vis_image_3d = cv2.line(vis_image_3d, tuple(points_2d[1]), tuple(points_2d[6]), (200, 200, 200), 2)
    vis_image_3d = cv2.line(vis_image_3d, tuple(points_2d[1]), tuple(points_2d[7]), (200, 200, 200), 2)
    vis_image_3d = cv2.line(vis_image_3d, tuple(points_2d[2]), tuple(points_2d[7]), (200, 200, 200), 2)
    vis_image_3d = cv2.line(vis_image_3d, tuple(points_2d[2]), tuple(points_2d[5]), (200, 200, 200), 2)
    vis_image_3d = cv2.line(vis_image_3d, tuple(points_2d[6]), tuple(points_2d[4]), (200, 200, 200), 2)
    vis_image_3d = cv2.line(vis_image_3d, tuple(points_2d[5]), tuple(points_2d[4]), (200, 200, 200), 2)
    vis_image_3d = cv2.line(vis_image_3d, tuple(points_2d[4]), tuple(points_2d[7]), (200, 200, 200), 2)
    vis_image_3d = cv2.line(vis_image_3d, tuple(points_2d[3]), tuple(points_2d[6]), (200, 200, 200), 2)
    vis_image_3d = cv2.line(vis_image_3d, tuple(points_2d[3]), tuple(points_2d[5]), (200, 200, 200), 2)
    # Draw axis; Project 3D axis to 2D
    axis_points_2d = []
    for point in axis_points_3d_cam:
        point = [-point[0] / point[2], point[1] / point[2]]
        pixel_x = int(point[0] * intrinsics[0, 0] + intrinsics[0, 2])
        pixel_y = int(point[1] * intrinsics[1, 1] + intrinsics[1, 2])
        axis_points_2d.append([pixel_x, pixel_y])
    vis_image_3d = cv2.arrowedLine(vis_image_3d, tuple(axis_points_2d[0]), tuple(axis_points_2d[1]), (0, 200, 200), 5)
    ####################################### 2D Visualization #######################################
    # Draw rotating bbox
    draw_rotating_bbox(vis_image_2d, (parent_rot_bbox[0], parent_rot_bbox[1], parent_rot_bbox[2], parent_rot_bbox[3]), parent_rot_bbox[4], (0, 0, 255))
    draw_rotating_bbox(vis_image_2d, (child_rot_bbox[0], child_rot_bbox[1], child_rot_bbox[2], child_rot_bbox[3]), child_rot_bbox[4], (0, 255, 0))
    draw_rotating_bbox(vis_image_2d, (axis_rot_bbox[0], axis_rot_bbox[1], axis_rot_bbox[2], axis_rot_bbox[3]), axis_rot_bbox[4], (255, 0, 0))

    vis_image = vis_image_3d
    # Add a vertical line to separate 2D and 3D images
    # vis_image = np.concatenate((vis_image_2d, vis_image_3d), axis=1)
    # vis_image = cv2.line(vis_image, (vis_image.shape[1] // 2, 0), (vis_image.shape[1] // 2, vis_image.shape[0]), (0, 0, 0), 2)
    # if closed:
    #     vis_image = cv2.putText(vis_image, "Closed", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, colors["yellow"], 2, cv2.LINE_AA)

    # Show image
    window_name = vis_image_file.split("/")[-1].split(".")[0]
    if not only_save_image:
        cv2.imshow(window_name, vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # save image to file instead
    cv2.imwrite(vis_image_file, vis_image)
    return vis_image


def check_annotations_o3d(points, bbox_3d, axis_points_3d):
    import open3d as o3d

    bbox_3d = np.array(bbox_3d)
    axis_points_3d = np.array(axis_points_3d)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.center = bbox_3d[0:3]
    bbox.R = R.from_rotvec(bbox_3d[6:9]).as_matrix()
    bbox.extent = bbox_3d[3:6]
    bbox.color = [1, 0, 0]
    axis_points = []
    for point in axis_points_3d:
        axis_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
        axis_point.translate(point)
        axis_points.append(axis_point)

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, bbox, origin] + axis_points)


################################# Class #################################
class PartNetLabeler:
    """Labelling joint on image/point cloud given joint information"""

    def __init__(self, grounding_dataset_folder):
        self.joint_info = None
        self.cam_info = None
        self.info = None
        self.link_cfg = None
        self.pcd = None
        self.link_dict = {}
        self.annotations = None
        self.annotations_3d = None
        self.num_links = 0
        self.num_images = 0
        self.semantic_data = None
        self.img_link_anno_dict = {}
        self.joint_type_semantics = []
        self.object_cato = None
        self.opened_closed_status_parts = None
        self.grounding_tasks = None
        self.grounding_dataset_folder = grounding_dataset_folder
        self.vqa_tasks = {
            # 2D tasks
            "single_link_rec_tasks": [],
            "all_parts_det_tasks": [],
            "joint_rec_tasks": [],
            "status_joint_reg_tasks": [],
            "grounding_tasks": [],
            "joint_rec_ext_tasks": [],
            "joint_rec_sep_depth_tasks": [],
            # 3D tasks
            "single_link_3d_rec_tasks": [],
            "all_parts_3d_det_tasks": [],
            "joint_3d_rec_tasks": [],
            "status_joint_3d_reg_tasks": [],
            "grounding_3d_tasks": [],
        }
        # place-holder
        self.invalid_vqa_tasks = copy.deepcopy(self.vqa_tasks)

    def read_info(self, joint_info_file, info_file, coco_annotation_file, annotation_3d_file, semantic_file):
        """Read joint information from file"""
        with open(joint_info_file, "r") as f:
            self.joint_info = json.load(f)
        # Filter out junk joints
        self.joint_info = [joint for joint in self.joint_info if joint["joint"] != "junk"]
        self.semantic_data = self.parse_semantic_file(semantic_file)
        # Compute some parameters
        self.num_links = len(self.semantic_data)

        self.parse_joint_info()
        with open(info_file, "r") as f:
            self.info = json.load(f)
        self.cam_info = self.info["camera_info"]
        with open(coco_annotation_file, "r") as f:
            self.annotations = json.load(f)
        with open(annotation_3d_file, "r") as f:
            self.annotations_3d = json.load(f)
        self.build_coco_annotation_dict()
        self.object_cato = self.info["model_cat"]
        self.opened_closed_status_parts = open_close_status.get(self.object_cato, None)
        idx_str = self.get_idx_str()
        self.idx_str = self.object_cato + "_" + idx_str
        self.grounding_tasks = self.load_grounding_tasks()

    def clean_info(self):
        self.joint_info = None
        self.cam_info = None
        self.info = None
        self.link_cfg = None
        self.pcd = None
        self.link_dict = {}
        self.annotations = None
        self.annotations_3d = None
        self.num_links = 0
        self.semantic_data = None
        self.img_link_anno_dict = {}

    def build_coco_annotation_dict(self):
        img_id_set = set()
        for annotation in self.annotations:
            img_id = annotation["image_id"]
            img_id_set.add(img_id)
            link_id = annotation["id"]
            img_link_id = img_id * self.num_links + link_id
            self.img_link_anno_dict[img_link_id] = annotation
        self.num_images = len(img_id_set)

    def parse_semantic_file(self, file_path):
        parsed_data = []
        with open(file_path, "r") as file:
            for line in file:
                parts = line.strip().split(" ")
                if len(parts) == 3:
                    parsed_data.append({"link_name": parts[0], "joint_type": parts[1], "semantic": parts[2]})
                else:
                    logger.warning(f"Error: {line} has wrong format")
        return parsed_data

    def get_idx_str(self):
        idx_str = ""
        idx_str_list = []
        for line_idx, link in enumerate(self.semantic_data):
            joint_type_from_urdf = joint_types_mapping[link["joint_type"]]
            senmantic_name = link["semantic"]
            cur_idx_str = f"{joint_type_from_urdf}_{senmantic_name}"
            if cur_idx_str in idx_str_list:
                continue
            idx_str_list.append(cur_idx_str)
        idx_str_list = list(set(sorted(idx_str_list)))
        for idx_str_ele in idx_str_list:
            idx_str += idx_str_ele + "_"
        return idx_str

    def load_grounding_tasks(self):
        task_json = os.path.join(self.grounding_dataset_folder, f"{self.idx_str}.json")
        if os.path.exists(task_json):
            logger.debug(f"Loading grounding tasks from {task_json}")
            with open(task_json, "r") as f:
                task_data = json.load(f)
            tasks = task_data[self.object_cato]
            return tasks
        else:
            return None

    def parse_joint_info(self):
        """Parse joint information"""
        self.link_dict = {}
        if len(self.joint_info) != len(self.semantic_data):
            return

        for link_idx, link_data in enumerate(self.joint_info):
            id = link_data["id"]
            parent_id = link_data["parent"]
            parent = -1
            for _i, _link in enumerate(self.joint_info):
                if _link["id"] == parent_id:
                    parent = _i
                    break
            parsed_link_data = {}
            # Parse joint information
            if link_data["joint"] == "hinge":
                axis_origin = np.array(link_data["jointData"]["axis"]["origin"])
                axis_direction = np.array(link_data["jointData"]["axis"]["direction"])
                # Convert y-up to z-up
                axis_origin = np.array([-axis_origin[2], -axis_origin[0], axis_origin[1]])
                axis_direction = np.array([-axis_direction[2], -axis_direction[0], axis_direction[1]])
                parsed_link_data = {
                    "id": id,
                    "parent": parent,
                    "type": "hinge",
                    "axis_origin": axis_origin,
                    "axis_direction": axis_direction,
                }
            elif link_data["joint"] == "slider":
                axis_origin = np.array(link_data["jointData"]["axis"]["origin"])
                axis_direction = np.array(link_data["jointData"]["axis"]["direction"])
                # Convert y-up to z-up
                axis_origin = np.array([-axis_origin[2], -axis_origin[0], axis_origin[1]])
                axis_direction = np.array([-axis_direction[2], -axis_direction[0], axis_direction[1]])
                parsed_link_data = {
                    "id": id,
                    "parent": parent,
                    "type": "slider",
                    "axis_origin": axis_origin,
                    "axis_direction": axis_direction,
                }
            else:
                parsed_link_data = {
                    "id": id,
                    "parent": parent,
                    "type": link_data["joint"],
                }
            # Parse semantic information
            parsed_link_data["link_name"] = self.semantic_data[link_idx]["link_name"]
            parsed_link_data["joint_type"] = self.semantic_data[link_idx]["joint_type"]
            parsed_link_data["semantic"] = self.semantic_data[link_idx]["semantic"]
            self.link_dict[link_idx] = parsed_link_data

    def get_annoation(self, img_idx, link_idx, key):
        """Get the bbox of link in the image"""
        img_link_idx = img_idx * self.num_links + link_idx
        if img_link_idx not in self.img_link_anno_dict:
            return None
        else:
            return self.img_link_anno_dict[img_link_idx][key]

    def is_visible(self, img_idx, link_idx, threshold: int = 1000):
        """Check if the link is visible in the image"""
        area = self.get_annoation(img_idx, link_idx, "area")
        vis_ratio = self.get_annoation(img_idx, link_idx, "vis_ratio")
        if area is not None and area > threshold:
            if vis_ratio is not None and vis_ratio > 0.2:
                return True
        else:
            return False

    def label_instances(
        self,
        image_folder,
        pcd_folder,
        num_bins: int,
        vis_thresh: int,
        vis: bool = False,
        only_save_image: bool = True,
        project_3d: bool = False,
        tolerance_gap: float = 0.3,
        z_angle_threshold: float = 0.5,
        SD_image: bool = False,
        normalize_output: bool = True,
        use_eight_points: bool = False,
    ):
        if len(self.link_dict) == 0:
            return None

        joint_annotations = []
        for image_idx in range(self.num_images):
            if SD_image:
                color_file_id = random.randint(0, 3)
                color_file_name = f"{image_idx}_{color_file_id}.png"
                image_file = os.path.join(image_folder, color_file_name)
            else:
                image_file = os.path.join(image_folder, f"{image_idx:06d}.png")
            image = cv2.imread(image_file)
            mask_file = os.path.join(os.path.dirname(image_folder), "mask", f"{image_idx:06d}.png")
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            # Read poses
            camera_pose = np.array(self.info["camera_poses"][image_idx]).reshape(4, 4)
            robot_pose = np.eye(4, dtype=np.float32)
            # Read camera intrinsics
            cam_intrinsics = np.array(
                [
                    [self.cam_info["fx"], 0, self.cam_info["cx"]],
                    [0, self.cam_info["fy"], self.cam_info["cy"]],
                    [0, 0, 1],
                ]
            )
            # Read pcd & Preprocess
            pcd_file = os.path.join(pcd_folder, f"{image_idx:06d}.ply")
            npy_folder = pcd_folder.replace("pointclouds", "npy_8192")
            if not os.path.exists(npy_folder):
                os.makedirs(npy_folder, exist_ok=True)
            npy_file = pcd_file.replace(".ply", "_8192.npy").replace("pointclouds", "npy_8192")
            if os.path.exists(npy_file):
                pcd = np.load(npy_file)
                pcd_file = npy_file
            else:
                if not os.path.exists(pcd_file):
                    pcd = np.zeros((8192, 3), dtype=np.float32)
                else:
                    pcd = read_ply_ascii(pcd_file)
                    print(f"Processing {npy_file} with fps")
                    sampled_pts = farthest_point_sample(pcd, 8192)
                    if not os.path.exists(os.path.dirname(npy_file)):
                        os.makedirs(os.path.dirname(npy_file))
                    np.save(npy_file, sampled_pts)
                    pcd = sampled_pts
                    pcd_file = npy_file

            # Process depth image
            if "sd" in pcd_folder:
                depth_folder = pcd_folder.replace("pointclouds_sd", "real_depth_images")
            else:
                depth_folder = pcd_folder.replace("pointclouds", "real_depth_images")
            depth = cv2.imread(os.path.join(depth_folder, f"{image_idx:06d}.png"), cv2.IMREAD_UNCHANGED)
            depth_color = convert_depth_to_color(depth)
            depth_color_folder = pcd_folder.replace("pointclouds", "depth_color_images")
            if not os.path.exists(depth_color_folder):
                os.makedirs(depth_color_folder, exist_ok=True)
            depth_color_file = os.path.join(depth_color_folder, f"{image_idx:06d}.png")
            if not os.path.exists(depth_color_file):
                cv2.imwrite(depth_color_file, depth_color)
            # Label image
            visual_image_save_folder = os.path.join(image_folder, "visual_images")
            if not os.path.exists(visual_image_save_folder):
                os.makedirs(visual_image_save_folder)

            joint_annotation = self.label_one_instance(
                image,
                depth,
                pcd,  # seems pcd is not used in labeling?
                mask,
                image_idx,
                camera_pose,
                robot_pose,
                cam_intrinsics,
                vis_thresh,
                vis,
                visual_image_save_folder,
                image_file,
                pcd_file,
                only_save_image,
                tolerance_gap,
                z_angle_threshold,
                SD_image,
                normalize_output,
                use_eight_points,
                depth_color_file,
            )
            if joint_annotation == "None box":
                print(f"Warning: {image_file} has None box")
            joint_annotations += joint_annotation
        return joint_annotations

    @staticmethod
    def find_minimum_rotated_bounding_box(mask, corner_points_representation=False):
        # Connecting left and right points on mask
        ys, xs = np.where(mask > 0)  # Foreground points
        leftmost_point = (min(xs), ys[np.argmin(xs)])
        rightmost_point = (max(xs), ys[np.argmax(xs)])
        topmost_point = (xs[np.argmin(ys)], min(ys))
        bottommost_point = (xs[np.argmax(ys)], max(ys))
        cv2.line(mask, leftmost_point, rightmost_point, 255, thickness=1)
        cv2.line(mask, topmost_point, bottommost_point, 255, thickness=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            rotated_rect = cv2.minAreaRect(largest_contour)
            center, size, angle = rotated_rect
            if corner_points_representation:
                box_points = cv2.boxPoints(rotated_rect).astype(int)
                box_points = [(int(point[0]), int(point[1])) for point in box_points]
                return box_points
            else:
                return center, size, angle
        else:
            return None, None, None

    def load_rotated_bbox_from_sem_masks(self, mask, link_idx, corner_points_representation=False):
        """
        If semantic masks are availible when rendering, use the masks to obtain the rotated bounding boxes
        The representation of the rotated bounding box is (cx, cy, w, h, angle)
        """
        index = np.where(mask == (link_idx + 1))
        if index[0].size == 0:
            return None, None, None
        mask_link = np.zeros_like(mask)
        mask_link[index] = 255
        center, size, angle = self.find_minimum_rotated_bounding_box(mask_link, corner_points_representation)
        if center is None:
            return None, None, None
        else:
            return center, size, angle

    def label_one_instance(
        self,
        image,
        depth,
        pcd,
        mask,
        image_idx,
        camera_pose,
        robot_pose,
        cam_intrinsics,
        vis_thresh,
        vis,
        visual_image_save_folder,
        image_full_path,
        pcd_full_path,
        only_save_image=True,
        tolerance_gap=0.3,
        z_angle_threshold=0.5,
        SD_image=False,
        normalize_output=True,
        use_eight_points=False,
        depth_color_file=None,
    ) -> np.ndarray:
        """Label one image/pcd instance"""
        joint_annotations = []
        link_info_annos = []
        vis_image = image.copy()  # Use a copy of the image for all joints visualization
        for link_idx, link_data in self.link_dict.items():
            if self.is_visible(image_idx, link_idx, vis_thresh):
                if link_data["type"] == "hinge" or link_data["type"] == "slider":
                    ####################################### Load 3D data [Joint] #######################################
                    joint_id = str(link_data["id"])
                    if joint_id not in self.annotations_3d[image_idx]:
                        continue
                    camera_pose = self.annotations_3d[image_idx]["meta"]["camera_pose"]  # Camera pose for point render
                    disturbance = self.annotations_3d[image_idx]["meta"]["disturbance"]  # Disturbance for point render
                    joint_T_3d = self.annotations_3d[image_idx][joint_id]["joint_T"]
                    disturbance_inv = np.linalg.inv(disturbance)
                    camera_pose_inv = np.linalg.inv(camera_pose)

                    # get axis_points
                    axis_points_3d = self.annotations_3d[image_idx][joint_id]["itp_points"]
                    axis_points_3d = np.array(axis_points_3d)
                    axis_points_3d_cam = axis_points_3d @ disturbance_inv[:3, :3].T + disturbance_inv[:3, 3]
                    axis_points_3d_cam = axis_points_3d_cam @ camera_pose_inv[:3, :3].T + camera_pose_inv[:3, 3]

                    # get bbox
                    bbox_3d = self.annotations_3d[image_idx][joint_id]["bbox_3d"]
                    _bbox_3d = BBox3D(bbox_3d[0:3], bbox_3d[3:6], bbox_3d[6:9])
                    _bbox_3d_cam = copy.deepcopy(_bbox_3d)
                    _bbox_3d_cam.transform(disturbance_inv)
                    _bbox_3d_cam.transform(camera_pose_inv)
                    bbox_3d_points_cam = _bbox_3d_cam.get_points()
                    bbox_3d_cam = _bbox_3d_cam.get_array().tolist()
                    ## Compute axis_proj_pos: proj axis to bbox
                    bbox_pose = _bbox_3d.get_pose()
                    bbox_pose_inv = np.linalg.inv(bbox_pose)
                    axis_points_proj = (axis_points_3d @ bbox_pose_inv[:3, :3].T + bbox_pose_inv[:3, 3]) / _bbox_3d.extent
                    axis_points_proj = axis_points_proj[0, :2] + 0.5  # normalize to [0, 1]
                    axis_points_proj = np.clip(axis_points_proj, 0, 1)
                    # project 3d axis points to 2d
                    axis_points_2d = []
                    for point in axis_points_3d_cam:
                        point = [-point[0] / point[2], point[1] / point[2]]
                        pixel_x = int(point[0] * cam_intrinsics[0, 0] + cam_intrinsics[0, 2])
                        pixel_y = int(point[1] * cam_intrinsics[1, 1] + cam_intrinsics[1, 2])
                        axis_points_2d.append([pixel_x, pixel_y])
                    axis_points_2d = np.array(axis_points_2d, dtype=np.int32)

                    # Draw rotation bbox
                    bbox_center = (axis_points_2d[0] + axis_points_2d[1]) / 2
                    # bbox_w, bbox_h = int(child_bbox[2] * tolerance_gap), int(child_bbox[3] * tolerance_gap)
                    bbox_w = np.linalg.norm(axis_points_2d[0] - axis_points_2d[1])
                    bbox_h = min(bbox_w, 10)  # 10 pixels

                    # Compute joint angle in pixel
                    joint_pos_angle = np.arctan2(
                        axis_points_2d[1, 1] - axis_points_2d[0, 1],
                        axis_points_2d[1, 0] - axis_points_2d[0, 0],
                    )
                    angle = joint_pos_angle * 180 / np.pi
                    angle_flip = False
                    if angle < 0:
                        angle = 180 + angle
                        angle_flip = True
                    # FIXME: after normalization, visualization seems wrong.
                    joint_angle = normalize_and_round_angle(angle, range_end=180)
                    # joint_angle = angle
                    axis_rot_bbox = [bbox_center[0], bbox_center[1], bbox_w, bbox_h, joint_angle]

                    ####################################### Load 2D data [BBOX] #######################################
                    # get bbox of link
                    child_bbox = self.get_annoation(image_idx, link_idx, "bbox")
                    # the rot bbox from the json file
                    child_rot_bbox = self.get_annoation(image_idx, link_idx, "rot_bbox")
                    # the rot bbox from the mask image
                    center_sem, size_sem, angle_sem = self.load_rotated_bbox_from_sem_masks(mask, link_idx)
                    if child_bbox is None or child_rot_bbox is None or center_sem is None:
                        logger.debug(f"Warning: {image_idx} link {link_idx} has None child box")
                        continue
                    else:
                        child_rot_bbox[-1] = normalize_and_round_angle(child_rot_bbox[-1], range_end=180)

                    parent_bbox = self.get_annoation(image_idx, link_data["parent"], "bbox")
                    parent_rot_bbox = self.get_annoation(image_idx, link_data["parent"], "rot_bbox")
                    # the rot bbox from the mask image:
                    center_sem, size_sem, angle_sem = self.load_rotated_bbox_from_sem_masks(mask, link_data["parent"])
                    if parent_bbox is None or parent_rot_bbox is None or center_sem is None:
                        logger.debug(f"Warning: {image_idx} link {link_idx} has None parent box")
                        continue
                    else:
                        parent_rot_bbox[-1] = normalize_and_round_angle(parent_rot_bbox[-1], range_end=180)

                    ## Convert 2D & 3D bbox to 2.5D bbox for 2D image.
                    zero_mask = depth == 0
                    depth_m = depth / 1000.0  # Convert to meters
                    depth_min = np.min(depth_m[~zero_mask])
                    depth_max = np.max(depth_m[~zero_mask])
                    d0 = abs(axis_points_3d_cam[0, 2])
                    d1 = abs(axis_points_3d_cam[1, 2])
                    d0 = (d0 - depth_min) / (depth_max - depth_min)
                    d1 = (d1 - depth_min) / (depth_max - depth_min)
                    if angle_flip:
                        d1, d0 = d0, d1
                    axis_rot_bbox_ext = [bbox_center[0], bbox_center[1], bbox_w, bbox_h, joint_angle, d0, d1]

                    # If child or parent bbox is None, meaning it is not visible in the image, skip
                    if child_bbox is None or parent_bbox is None:
                        logger.debug(f"Warning: {image_idx} has None box for child or parent")
                        continue

                    joint_value = 0.0
                    for link_name in self.info.keys():
                        if link_name.startswith(link_data["link_name"]):
                            joint_value = self.info[link_name][image_idx]

                    closed = False
                    if joint_value < 0.2:
                        closed = True

                    ####################################### Visualization #######################################
                    if vis:
                        # vis_image = image.copy()
                        # use a function to draw the bbox
                        vis_image_file = os.path.join(visual_image_save_folder, f"{image_idx}_joint_{link_idx}_parent_{link_data['parent']}.png")
                        # Draw 3D annotation
                        vis_image = annotation_visualization(
                            vis_image, bbox_3d_points_cam, axis_points_3d_cam, cam_intrinsics, parent_rot_bbox, child_rot_bbox, axis_rot_bbox, closed, vis_image_file, only_save_image
                        )

                    ####################################### Output #######################################
                    # Write annotation
                    joint_annotation = {
                        "image_idx": image_idx,
                        "semantic": link_data["semantic"],
                        "camera_pose": camera_pose,
                        "camera_intrinsics": cam_intrinsics.tolist(),
                        "depth_min": depth_min,
                        "depth_max": depth_max,
                        ######### 2D annotation #########
                        "axis_rot_bbox": axis_rot_bbox,
                        "axis_rot_bbox_ext": axis_rot_bbox_ext,
                        "child_rot_bbox": child_rot_bbox,
                        "parent_rot_bbox": parent_rot_bbox,
                        "joint_type": link_data["joint_type"],  # should use the joint_type from the urdf file
                        "joint_value": joint_value,
                        "joint_rot_bbox": [bbox_center[0], bbox_center[1], bbox_w, bbox_h, angle],
                        ######## 3D annotation ########
                        "joint_T_3d": joint_T_3d,
                        "bbox_3d": bbox_3d,
                        "bbox_3d_cam": bbox_3d_cam,
                        "axis_3d": axis_points_3d.tolist(),
                        "axis_3d_cam": axis_points_3d_cam.tolist(),
                        "axis_3d_proj": axis_points_proj.tolist(),
                        "closed": closed,
                    }
                    joint_annotations.append(joint_annotation)

                    # [DEBUG]
                    ####################################### Construct VQA tasks #######################################
                    anno_meta = {"intrinsics": cam_intrinsics, "camera_pose": np.eye(4), "depth_min": depth_min, "depth_max": depth_max, "img_width": image.shape[1], "img_height": image.shape[0]}
                    # Special operations:
                    pcd_full_path = image_full_path  # use the image path as the pcd path

                    # construct the 2D VQA task
                    child_rot_bbox_w_normalized_angle = copy.deepcopy(child_rot_bbox)
                    child_rot_bbox_w_normalized_angle[4] = normalize_and_round_angle(child_rot_bbox_w_normalized_angle[4], range_end=180)
                    self.vqa_tasks["single_link_rec_tasks"].append(create_single_link_rec_rotated_task(link_data["semantic"], child_rot_bbox_w_normalized_angle, image_full_path))
                    link_info = random.choice([child_rot_bbox_w_normalized_angle, link_data["semantic"]])
                    joint_type_urdf = joint_types_mapping[link_data["joint_type"]]
                    self.vqa_tasks["joint_rec_tasks"].append(create_rec_joint_rotated_task(link_info, axis_rot_bbox, joint_type_urdf, image_full_path))
                    self.vqa_tasks["joint_rec_ext_tasks"].append(create_rec_joint_rotated_ext_task(link_info, axis_rot_bbox_ext, joint_type_urdf, image_full_path))
                    self.vqa_tasks["joint_rec_sep_depth_tasks"].append(create_rec_joint_rotated_ext_task_with_sep_depth(link_info, axis_rot_bbox_ext, joint_type_urdf, image_full_path))

                    # construct the 3D VQA task
                    self.vqa_tasks["single_link_3d_rec_tasks"].append(
                        create_single_link_3d_rec_task(link_data["semantic"], bbox_3d_cam, pcd_full_path, anno_meta=anno_meta, use_eight_points=use_eight_points)
                    )
                    # link_info_3d = random.choice([bbox_3d_cam, link_data["semantic"]])
                    link_info_3d = bbox_3d_cam
                    self.vqa_tasks["joint_3d_rec_tasks"].append(
                        create_3d_rec_joint_task(link_info_3d, axis_points_3d_cam, joint_type_urdf, pcd_full_path, anno_meta=anno_meta, use_eight_points=use_eight_points, axis_3d_proj=None)
                    )

                    # Update the link_info_annos
                    link_name = link_data["semantic"]
                    link_name_with_status = copy.deepcopy(link_name)
                    if self.opened_closed_status_parts:
                        if link_data["semantic"] in self.opened_closed_status_parts:
                            self.vqa_tasks["status_joint_reg_tasks"].append(create_reg_status_qa_task(link_info, closed, image_full_path))
                            self.vqa_tasks["status_joint_3d_reg_tasks"].append(
                                create_3d_reg_status_qa_task(link_info_3d, closed, pcd_full_path, anno_meta=anno_meta, use_eight_points=use_eight_points)
                            )
                            link_name_with_status = "closed_" + link_name if closed else "opened_" + link_name
                    link_info_annos.append(
                        {
                            "link_name": link_data["semantic"],
                            "bbox": child_rot_bbox_w_normalized_angle,
                            "joint_type": joint_type_urdf,
                            "link_name_status": link_name_with_status,
                            "bbox_3d": bbox_3d_cam,
                            "axis_3d": axis_points_3d_cam,
                        }
                    )

        if len(link_info_annos) > 0:
            self.vqa_tasks["all_parts_det_tasks"].append(create_det_all_rotated_task(link_info_annos, image_full_path))
            self.vqa_tasks["all_parts_3d_det_tasks"].append(create_det_all_bbox_3d_task(link_info_annos, pcd_full_path, anno_meta=anno_meta, use_eight_points=use_eight_points))
            all_possible_tasks = {}
            if self.grounding_tasks:
                # use the link_info_annos to get the tasks
                for link_info_ in link_info_annos:
                    link_name_with_status_ = link_info_["link_name_status"]
                    if link_name_with_status_ in self.grounding_tasks:
                        all_possible_tasks["link_name_with_status_"] = self.grounding_tasks[link_name_with_status_]
                grounding_task = create_grounding_task(all_possible_tasks, link_info_annos, image_full_path, indexing="bbox")
                if grounding_task:
                    self.vqa_tasks["grounding_tasks"].append(grounding_task)

                grounding_3d_task = create_3d_grounding_task(all_possible_tasks, link_info_annos, pcd_full_path, indexing="bbox_3d", anno_meta=anno_meta, use_eight_points=use_eight_points)
                if grounding_3d_task:
                    self.vqa_tasks["grounding_3d_tasks"].append(grounding_3d_task)

        return joint_annotations


def label_one_data(
    data_name,
    grounding_dataset_folder,
    data_dir,
    output_dir,
    num_bins,
    vis_thresh,
    vis,
    use_texture,
    only_save_vis,
    tolerance_gap,
    z_angle_threshold=np.pi / 3.0,
    normalize_output=True,
    use_eight_points=False,
):
    if type(data_name) == int:
        data_name = str(data_name)
    export_folder = os.path.join(output_dir, data_name)

    if not use_texture:
        image_folder = os.path.join(export_folder, "raw_images")
        pcd_folder = os.path.join(export_folder, "pointclouds")

    else:
        image_folder = os.path.join(export_folder, "controlnet_images_seg")
        if not os.path.exists(image_folder):
            image_folder = os.path.join(export_folder, "controlnet_images")
        pcd_folder = os.path.join(export_folder, "pointclouds_sd")

    joint_annotations_file = os.path.join(export_folder, "joint_annotations.json")
    if not os.path.exists(image_folder):
        print(f"Skip {data_name} since there is no image folder...")
        return {}
    if len(os.listdir(image_folder)) == 0:
        print(f"Skip {data_name} since there is no image generated...")
        return {}

    data_folder = os.path.join(data_dir, data_name)
    data_file = os.path.join(data_folder, "mobility.urdf")
    coco_annotation_file = os.path.join(export_folder, "annotations.json")
    annotation_3d_file = os.path.join(export_folder, "annotations_3d.json")
    joint_info_file = os.path.join(export_folder, "mobility_v2.json")
    info_file = os.path.join(export_folder, "info.json")
    cam_info_file = os.path.join(export_folder, "camera.json")
    semantic_file = os.path.join(export_folder, "semantics.txt")

    file_incomplete = False
    for file in [data_file, coco_annotation_file, joint_info_file, info_file, semantic_file]:
        if not os.path.exists(file):
            file_incomplete = True
            break
    if file_incomplete:
        return "FileNotComplete"

    try:
        # Init PartNet labeler
        partnet_labeler = PartNetLabeler(grounding_dataset_folder)
        partnet_labeler.read_info(joint_info_file, info_file, coco_annotation_file, annotation_3d_file, semantic_file)
        # Start labeling
        joint_annotations = partnet_labeler.label_instances(
            image_folder=image_folder,
            pcd_folder=pcd_folder,
            num_bins=num_bins,
            vis_thresh=vis_thresh,
            vis=vis,
            only_save_image=only_save_vis,
            tolerance_gap=tolerance_gap,
            z_angle_threshold=z_angle_threshold,
            SD_image=use_texture,
            normalize_output=normalize_output,
            use_eight_points=use_eight_points,
        )
        if joint_annotations is not None:
            with open(joint_annotations_file, "w") as f:
                json.dump(joint_annotations, f)
            return partnet_labeler.vqa_tasks
    except Exception as e:
        logger.error(f"Error: {data_name} failed to label with error {e}")
        return str(e)


if __name__ == "__main__":
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_name", type=str, default="154")
    argparser.add_argument("--data_dir", type=str, required=True, help="Path to the original dataset")
    argparser.add_argument("--output_dir", type=str, required=True, help="Path to the rendering output folder")
    argparser.add_argument("--vqa_tasks_folder", type=str, default=True, help="Path to the VQA tasks folder")
    argparser.add_argument("--grounding_dataset_folder", type=str, default="openai_grounding_tasks", help="Path to the grounding dataset folder")
    argparser.add_argument("--classname_file", type=str, default="partnet_pyrender_dataset_v3_classname.json", help="Path to the class name file")
    argparser.add_argument("--normalize_output", type=bool, default=True, help="Normalize the output to 0-100")
    argparser.add_argument("--use_eight_points", type=bool, default=True, help="Use eight points to represent the bbox")
    argparser.add_argument("--num_bins", type=int, default=60)
    argparser.add_argument("--use_texture", action="store_true")
    argparser.add_argument("--vis_thresh", type=int, default=196)
    argparser.add_argument("--vis", action="store_true")
    argparser.add_argument("--only_save_vis", action="store_true")
    args = argparser.parse_args()

    # Rewrite path for debug
    data_dir = args.data_dir
    output_dir = args.output_dir
    grounding_dataset_folder = args.grounding_dataset_folder
    vqa_tasks_folder = args.vqa_tasks_folder
    classname_file = args.classname_file
    use_texture = args.use_texture
    normalize_output = args.normalize_output
    use_eight_points = args.use_eight_points

    # Override folder for testing
    # vqa_tasks_folder = "./vqa_tasks_v3_test"
    # grounding_dataset_folder = "./openai_grounding_tasks"
    # output_dir = "./output"
    # data_dir = "./test_data"

    if use_texture and "sd" not in vqa_tasks_folder:
        if vqa_tasks_folder[-1] == "/":
            vqa_tasks_folder = vqa_tasks_folder[:-1]
        vqa_tasks_folder += "_sd"

    os.makedirs(vqa_tasks_folder, exist_ok=True)

    num_bins = args.num_bins
    vis_thresh = args.vis_thresh
    vis = args.vis
    vis = True
    only_save_vis = True
    project_3d = True
    tolerance_gap = 0.3
    z_angle_threshold = np.pi / 3.0

    data_name = args.data_name
    if data_name != "all":
        annotations_result = label_one_data(
            data_name, grounding_dataset_folder, data_dir, output_dir, num_bins, vis_thresh, vis, use_texture, only_save_vis, tolerance_gap, z_angle_threshold, normalize_output, use_eight_points
        )
        if type(annotations_result) is not dict:
            print(f"Error: {data_name} failed to label")
        task_annotations = {
            # 2D tasks
            "single_link_rec_tasks": [],
            "all_parts_det_tasks": [],
            "joint_rec_tasks": [],
            "status_joint_reg_tasks": [],
            "grounding_tasks": [],
            "joint_rec_ext_tasks": [],
            "joint_rec_sep_depth_tasks": [],
            # 3D tasks
            "single_link_3d_rec_tasks": [],
            "all_parts_3d_det_tasks": [],
            "joint_3d_rec_tasks": [],
            "status_joint_3d_reg_tasks": [],
            "grounding_3d_tasks": [],
        }
        for task in task_annotations:
            task_annotations[task].extend(annotations_result.get(task, []))
        save_annotations(task_annotations, vqa_tasks_folder, data_name)
    else:
        all_sub_folders = os.listdir(output_dir)
        json_path = classname_file
        with open(json_path, "r") as f:
            data = json.load(f)

        # construct task lists according to the train-val splits
        val_ids, train_ids = [], []
        for class_name in data:
            if class_name in HOLDOUT_CLASSES:
                val_ids.extend(data[class_name])
            else:
                train_ids.extend(data[class_name])

        splits = {"train": train_ids, "val": val_ids}

        for split in splits:
            split_ids = splits[split]
            num_processes = min(min(multiprocessing.cpu_count(), len(split_ids)), 64)
            print(f"Process {len(split_ids)} data with {num_processes} processes")
            pool = multiprocessing.Pool(processes=num_processes)
            task_annotations = {
                # 2D tasks
                "single_link_rec_tasks": [],
                "all_parts_det_tasks": [],
                "joint_rec_tasks": [],
                "status_joint_reg_tasks": [],
                "grounding_tasks": [],
                "joint_rec_ext_tasks": [],
                "joint_rec_sep_depth_tasks": [],
                # 3D tasks
                "single_link_3d_rec_tasks": [],
                "all_parts_3d_det_tasks": [],
                "joint_3d_rec_tasks": [],
                "status_joint_3d_reg_tasks": [],
                "grounding_3d_tasks": [],
            }
            # with tqdm(total=len(split_ids)) as pbar:
            #     for i, dataset_item in enumerate(split_ids):
            #         dataset_item = str(dataset_item)
            #         annotations_result = pool.apply(
            #             label_one_data,
            #             args=(
            #                 dataset_item,
            #                 grounding_dataset_folder,
            #                 data_dir,
            #                 output_dir,
            #                 num_bins,
            #                 vis_thresh,
            #                 vis,
            #                 use_texture,
            #                 only_save_vis,
            #                 tolerance_gap,
            #                 z_angle_threshold,
            #                 normalize_output,
            #                 use_eight_points,
            #             ),
            #         )
            #         if type(annotations_result) is not dict:
            #             logger.error(f"Error: {dataset_item} failed to label and return not dict result with type {type(annotations_result)}: {annotations_result}")
            #             continue
            #         if annotations_result:
            #             for task in task_annotations:
            #                 task_annotations[task].extend(annotations_result.get(task, []))
            #         pbar.update(1)
            # save_annotations(task_annotations, vqa_tasks_folder, split)
            # pool.close()
            with tqdm(total=len(split_ids)) as pbar:
                result_objects = []
                for dataset_item in split_ids:
                    result = pool.apply_async(
                        label_one_data,
                        args=(
                            str(dataset_item),
                            grounding_dataset_folder,
                            data_dir,
                            output_dir,
                            num_bins,
                            vis_thresh,
                            vis,
                            use_texture,
                            only_save_vis,
                            tolerance_gap,
                            z_angle_threshold,
                            normalize_output,
                            use_eight_points,
                        ),
                        error_callback=lambda e: logger.error(f"Error processing {dataset_item}: {e}"),
                    )
                    result_objects.append((dataset_item, result))

                # Collect results with a timeout
                for dataset_item, result in result_objects:
                    try:
                        annotations_result = result.get(timeout=60)  # Timeout set to 60 seconds
                        if annotations_result:
                            for task in task_annotations:
                                task_annotations[task].extend(annotations_result.get(task, []))
                    except multiprocessing.TimeoutError:
                        logger.error(f"Timeout: {dataset_item} processing exceeded time limit.")
                    except Exception as e:
                        logger.error(f"Error: {dataset_item} processing failed with exception: {e}")
                    finally:
                        pbar.update(1)

            save_annotations(task_annotations, vqa_tasks_folder, split)
            pool.close()
            pool.join()
