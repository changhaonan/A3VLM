"""Viewer of multiscan data."""

import h5py
import os
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from utils import BBox3D

######################## General 3D tools ########################


def calculate_pts_in_sight(pts, world2cam, camera_intrinsics):
    """Calculate the points in sight of the camera."""
    # Transform the points to the camera coordinate
    pts_cam = np.dot(pts, world2cam[:3, :3].T) + world2cam[:3, 3]
    # Project the points to the image plane
    pts_img = np.dot(pts_cam, camera_intrinsics.T)
    pts_img = pts_img[:, :2] / pts_img[:, 2:]
    # Filter the points in the image plane
    pts_in_sight = pts_img[(pts_img[:, 0] >= 0) & (pts_img[:, 0] < camera_intrinsics[0, 2] * 2) & (pts_img[:, 1] >= 0) & (pts_img[:, 1] < camera_intrinsics[1, 2] * 2)]
    return pts_in_sight


def calculate_zy_rotation_for_arrow(vec):
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])

    vec = Rz.T @ vec

    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    return Rz, Ry


def get_arrow(end, origin=np.array([0, 0, 0]), scale=1.0, color=np.array([1, 0, 0])):
    assert not np.all(end == origin)
    vec = end - origin
    size = np.sqrt(np.sum(vec**2))

    Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = o3d.geometry.TriangleMesh.create_arrow(cone_radius=size / 17.5 * scale, cone_height=size * 0.2 * scale, cylinder_radius=size / 30 * scale, cylinder_height=size * (1 - 0.2 * scale))
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    mesh.paint_uniform_color(color)
    return mesh


def save_obj_parts_func(obj_part_dict):
    keys = ["pts", "motion_axes", "motion_origins", "part_instance_masks", "part_semantic_masks", "transformation_back"]

    def save_obj_parts(name, obj):
        if isinstance(obj, h5py.Group):
            obj_part_dict[name] = {}
            obj_part_dict[name]["objectId"] = obj.attrs.get("objectId")
            for key in obj.keys():
                if key in keys:
                    obj_part_dict[name][key] = np.array(obj[key])

    return save_obj_parts


if __name__ == "__main__":
    multi_scan_dir = "/home/harvey/Data/multi_scan"
    multi_scan_art_file = "/home/harvey/Data/multi_scan_art/articulated_dataset/articulated_objects.train.h5"
    data_id = "scene_00000_01"

    mesh_file = f"{data_id}.ply"
    mesh_file = os.path.join(multi_scan_dir, "output", data_id, mesh_file)

    # Load the mesh & alignment structure
    mesh = o3d.io.read_triangle_mesh(mesh_file)

    alignment_file = f"{data_id}.align.json"
    alignment_file = os.path.join(multi_scan_dir, "output", data_id, alignment_file)
    with open(alignment_file) as f:
        alignment = json.load(f)
    coordinate_transform = alignment.get("coordinate_transform")
    # coordinate_transform is the 16x1 vector
    coordinate_mat = np.reshape(coordinate_transform, (4, 4), order="F")
    # apply inplace transformation, the mesh is transformed back to the pose without alignment
    # mesh.transform(coordinate_mat)

    # Load transforms
    frame_file = f"{data_id}.jsonl"
    with open(os.path.join(multi_scan_dir, "output", data_id, frame_file)) as f:
        frames = [json.loads(line) for line in f]

    # Read the art file
    obj_part_dict = {}
    with h5py.File(multi_scan_art_file, "r") as h5_file:
        h5_file.visititems(save_obj_parts_func(obj_part_dict))

    # Annotations for parts
    obj_pts_all = []
    obj_seg_all = []
    exist_num_parts = 0
    parts_vis = []
    for obj_name, obj_parts in obj_part_dict.items():
        if not obj_name.startswith(data_id):
            continue
        transformation_back = np.asarray(obj_parts.get("transformation_back"))
        transformation_back = np.reshape(transformation_back, (4, 4), order="F")

        obj_pts = obj_parts.get("pts")
        # Transform back to world coordinate
        obj_pts_pos = obj_pts[:, :3]
        obj_pts_pos = np.dot(obj_pts_pos, transformation_back[:3, :3].T) + transformation_back[:3, 3]
        # obj_pts_color = obj_pts[:, 3:6]
        # Apply colormap
        obj_ins_seg = obj_parts.get("part_instance_masks")
        obj_sem_seg = obj_parts.get("part_semantic_masks")
        obj_pts_color = plt.cm.get_cmap("tab20")(obj_ins_seg)[:, :3]
        obj_pts = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pts_pos))
        obj_pts.colors = o3d.utility.Vector3dVector(obj_pts_color)
        obj_pts_all.append(obj_pts_pos)
        obj_seg_all.append(obj_ins_seg + exist_num_parts)
        exist_num_parts += np.max(obj_ins_seg) + 1

        # Motion axises
        arrows = []
        part_bboxes = []
        motion_axes = np.asarray(obj_parts.get("motion_axes"))
        motion_origins = np.asarray(obj_parts.get("motion_origins"))

        motion_axes = np.dot(motion_axes, transformation_back[:3, :3].T)
        motion_origins = np.dot(motion_origins, transformation_back[:3, :3].T) + transformation_back[:3, 3]
        for _i, (motion_axis, motion_origin) in enumerate(zip(motion_axes, motion_origins)):
            part_idx = _i + 1
            # Part pcd
            part_pts = np.copy(obj_pts_pos[obj_ins_seg == part_idx])
            part_bbox = BBox3D()
            part_bbox.create_joint_aligned_bbox(part_pts, motion_origin, motion_axis)
            part_bboxes.append(part_bbox)

            arrow = get_arrow(motion_origin + motion_axis, motion_origin, color=plt.cm.get_cmap("tab20")(part_idx)[:3])
            arrows.append(arrow)

        # Generate A3 annotations
        part_bboxes_o3d = [bbox.get_bbox_o3d() for bbox in part_bboxes]
        # parts_vis.extend(arrows + part_bboxes_o3d)
        o3d.visualization.draw_geometries([obj_pts] + arrows + part_bboxes_o3d, window_name=obj_name)
        # transform before visualize
        # o3d.visualization.draw_geometries([obj_pts, mesh] + arrows + part_bboxes_o3d)

    # Traverse frames
    obj_pts_all = np.vstack(obj_pts_all)
    obj_sem_seg = np.hstack(obj_seg_all)
    obj_pts_all_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pts_all))
    obj_pts_all_o3d.colors = o3d.utility.Vector3dVector(plt.cm.get_cmap("tab20")(obj_sem_seg % 20)[:, :3])
    o3d.visualization.draw_geometries([obj_pts_all_o3d])
    transform_origins = []
    for _i, frame in enumerate(frames):
        size = 1.0 if _i == 0 else 0.1
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        transform = np.asarray(frame.get("transform"))
        transform = np.reshape(transform, (4, 4), order="F")
        transform = np.linalg.inv(coordinate_mat) @ transform
        if _i % 10 == 0:
            transform_origins.append(origin.transform(transform))
            # Compute points in sight

    # Visualize the mesh
    # global_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    o3d.visualization.draw_geometries([mesh] + transform_origins + parts_vis)
