"""Generate lables for multiscan data."""

import cv2
import h5py
import os
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from data_utils import AxisBBox3D, get_arrow
from queue import PriorityQueue

######################## General 3D tools ########################


def calculate_pts_in_sight(pts, world2cam, camera_intrinsics, **kwargs):
    """Calculate the points in sight of the camera."""
    # Transform the points to the camera coordinate
    pts_cam = np.dot(pts, world2cam[:3, :3].T) + world2cam[:3, 3]
    pts_seg = kwargs.get("pts_seg", None)
    pts_color = plt.cm.get_cmap("tab20")(pts_seg % 20)[:, :3]
    # Project the points to the image plane
    pts_cam_pixel = pts_cam[:, :2] / (pts_cam[:, 2:] + 1e-6)
    pts_cam_pixel[:, 0] = (
        camera_intrinsics[0, 0] * pts_cam_pixel[:, 0] + camera_intrinsics[0, 2]
    )
    pts_cam_pixel[:, 1] = (
        camera_intrinsics[1, 1] * pts_cam_pixel[:, 1] + camera_intrinsics[1, 2]
    )
    pts_img = np.round(pts_cam_pixel).astype(int)
    # Filter the points in the image plane
    pts_in_sight_idx = (
        (pts_img[:, 0] >= 0)
        & (pts_img[:, 0] < camera_intrinsics[0, 2] * 2)
        & (pts_img[:, 1] >= 0)
        & (pts_img[:, 1] < camera_intrinsics[1, 2] * 2)
        & (pts_cam[:, 2] < 0)
    )
    pts_in_sight = pts_cam[pts_in_sight_idx]
    # Compute the visible seg
    visible_segs = []
    visible_ratios = []
    if pts_seg is not None:
        seg_counts = kwargs.get("seg_counts", None)
        visible_seg = pts_seg[pts_in_sight_idx]
        for seg in np.unique(visible_seg):
            visible_seg_count = np.sum(visible_seg == seg)
            seg_count = (
                seg_counts[seg] if seg_counts is not None else np.sum(pts_seg == seg)
            )
            visible_ratio = visible_seg_count / seg_count
            if visible_ratio > 0.2:
                visible_segs.append(seg)
                visible_ratios.append(visible_ratio)
    vis = kwargs.get("vis", False)
    if vis:
        # [Debug]: Visualize the visible region
        pts_all = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_cam))
        pts_all.colors = o3d.utility.Vector3dVector(pts_color)
        pts_in_sight_o3d = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(pts_in_sight)
        )
        pts_in_sight_color = pts_color[pts_in_sight_idx] * 0.3
        pts_in_sight_o3d.colors = o3d.utility.Vector3dVector(pts_in_sight_color)
        visible_bbox = []
        for seg in visible_segs:
            bbox = AxisBBox3D()
            bbox.create_minimum_axis_aligned_bbox(pts_cam[pts_seg == seg])
            visible_bbox.append(bbox.get_bbox_o3d())
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        o3d.visualization.draw_geometries(
            [pts_all, pts_in_sight_o3d, origin] + visible_bbox
        )
    return visible_segs, visible_ratios


def save_obj_parts_func(obj_part_dict):
    keys = [
        "pts",
        "motion_axes",
        "motion_origins",
        "part_instance_masks",
        "part_semantic_masks",
        "transformation_back",
    ]

    def save_obj_parts(name, obj):
        if isinstance(obj, h5py.Group):
            obj_part_dict[name] = {}
            obj_part_dict[name]["objectId"] = obj.attrs.get("objectId")
            obj_part_dict[name]["partLabels"] = obj.attrs.get("partLabels")
            for key in obj.keys():
                if key in keys:
                    obj_part_dict[name][key] = np.array(obj[key])

    return save_obj_parts


if __name__ == "__main__":
    # Parse config
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multi_scan_dir", type=str, default="/home/harvey/Data/multi_scan"
    )
    parser.add_argument(
        "--multi_scan_art_file",
        type=str,
        default="/home/harvey/Data/multi_scan_art/articulated_dataset/articulated_objects.train.h5",
    )
    parser.add_argument("--data_id", type=str, default="scene_00000_01")
    parser.add_argument("--enable_filter", type=bool, default=True)
    args = parser.parse_args()

    multi_scan_dir = args.multi_scan_dir
    multi_scan_art_file = args.multi_scan_art_file
    data_id = args.data_id
    enable_filter = args.enable_filter
    vis = False

    export_dir = os.path.join(multi_scan_dir, "output", data_id, "anno")
    os.makedirs(export_dir, exist_ok=True)

    mesh_file = f"{data_id}.ply"
    mesh_file = os.path.join(multi_scan_dir, "output", data_id, mesh_file)

    # Alignment structure
    alignment_file = f"{data_id}.align.json"
    alignment_file = os.path.join(multi_scan_dir, "output", data_id, alignment_file)
    with open(alignment_file) as f:
        alignment = json.load(f)
    coordinate_transform = alignment.get("coordinate_transform")
    # coordinate_transform is the 16x1 vector
    coordinate_mat = np.reshape(coordinate_transform, (4, 4), order="F")

    # Load transforms
    frame_file = f"{data_id}.jsonl"
    with open(os.path.join(multi_scan_dir, "output", data_id, frame_file)) as f:
        frames = [json.loads(line) for line in f]

    # Read the art file
    obj_part_dict = {}
    with h5py.File(multi_scan_art_file, "r") as h5_file:
        h5_file.visititems(save_obj_parts_func(obj_part_dict))

    # [Debug]: analysis the trainfiles
    # tran_ids = []
    # for obj_name, obj_parts in obj_part_dict.items():
    #     tran_ids.append("_".join(obj_name.split("_")[:3]))
    # print(len(set(tran_ids)))
    # print(set(tran_ids))
    # Annotations for parts
    obj_pts_all = []
    obj_seg_all = []
    exist_num_parts = 0
    parts_vis = []
    anno_3d_dict = {}
    for obj_name, obj_parts in obj_part_dict.items():
        if not obj_name.startswith(data_id):
            continue
        transformation_back = np.asarray(obj_parts.get("transformation_back"))
        transformation_back = np.reshape(transformation_back, (4, 4), order="F")

        obj_pts = obj_parts.get("pts")
        # Transform back to world coordinate
        obj_pts_pos = obj_pts[:, :3]
        obj_pts_pos = (
            np.dot(obj_pts_pos, transformation_back[:3, :3].T)
            + transformation_back[:3, 3]
        )
        # obj_pts_color = obj_pts[:, 3:6]
        # Apply colormap
        obj_ins_seg = obj_parts.get("part_instance_masks")
        obj_sem_seg = obj_parts.get("part_semantic_masks")
        obj_pts_color = plt.cm.get_cmap("tab20")(obj_ins_seg)[:, :3]
        obj_pts = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj_pts_pos))
        obj_pts.colors = o3d.utility.Vector3dVector(obj_pts_color)
        obj_pts_all.append(obj_pts_pos)
        obj_seg_all.append(obj_ins_seg + exist_num_parts)

        # Motion axises
        arrows = []
        part_bboxes = []
        motion_axes = np.asarray(obj_parts.get("motion_axes"))
        motion_origins = np.asarray(obj_parts.get("motion_origins"))

        motion_axes = np.dot(motion_axes, transformation_back[:3, :3].T)
        motion_origins = (
            np.dot(motion_origins, transformation_back[:3, :3].T)
            + transformation_back[:3, 3]
        )

        part_labels = obj_parts.get("partLabels")[1:]  # First one is static
        # Sort by numerical order
        part_labels = sorted(part_labels, key=lambda x: int(x.split(".")[-1]))
        for frame_idx, (part_label, motion_axis, motion_origin) in enumerate(
            zip(part_labels, motion_axes, motion_origins)
        ):
            part_idx = int(part_label.split(".")[-1])
            # Part pcd
            part_pts = np.copy(obj_pts_pos[obj_ins_seg == part_idx])
            if enable_filter:
                # Filter out outliers using DBSCAN
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(part_pts))
                pcd.estimate_normals()
                cl, ind = pcd.remove_radius_outlier(nb_points=10, radius=0.05)
                part_pts = np.asarray(pcd.points)[ind]

            part_bbox = AxisBBox3D()
            part_bbox.create_joint_aligned_bbox(part_pts, motion_origin, motion_axis)
            part_bboxes.append(part_bbox)

            axis = part_bbox.get_axis_array().reshape(2, 3)
            arrow = get_arrow(
                axis[1], axis[0], color=plt.cm.get_cmap("tab20")(part_idx)[:3]
            )
            arrows.append(arrow)

            # Save the annotation in part level
            part_name = f"{data_id}_{part_idx + exist_num_parts}"
            anno_3d_dict[part_name] = {}
            anno_3d_dict[part_name]["objectId"] = obj_name
            anno_3d_dict[part_name]["bbox_3d"] = part_bbox.get_bbox_array()
            anno_3d_dict[part_name]["axis_3d"] = part_bbox.get_axis_array()

        # Update the existing num parts
        exist_num_parts += np.max(obj_ins_seg) + 1
        print(exist_num_parts)
        # [Debug]: Visualize each part
        if vis:
            part_bboxes_o3d = [bbox.get_bbox_o3d() for bbox in part_bboxes]
            parts_vis.extend(arrows + part_bboxes_o3d)
            o3d.visualization.draw_geometries(
                [obj_pts] + arrows + part_bboxes_o3d, window_name=obj_name
            )
    if vis:
        # Load the mesh & alignment structure
        mesh = o3d.io.read_triangle_mesh(mesh_file)
    else:
        mesh = None

    # Traverse frames
    obj_pts_all = np.vstack(obj_pts_all)
    obj_sem_seg = np.hstack(obj_seg_all)
    obj_seg_counts = {}
    for seg in np.unique(obj_sem_seg):
        obj_seg_counts[seg] = np.sum(obj_sem_seg == seg)
    transform_origins = []

    # Generate and export the annotation
    priority_queue_dict = {}
    max_queue_len = 100
    for frame_idx, frame in enumerate(frames):
        print(f"Processing frame {frame_idx}/{len(frames)}...")
        size = 1.0 if frame_idx == 0 else 0.1
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        transform = np.asarray(frame.get("transform"))
        transform = np.reshape(transform, (4, 4), order="F")
        transform = np.linalg.inv(coordinate_mat) @ transform
        intrinsics = frame.get("intrinsics")
        intrinsics = np.reshape(intrinsics, (3, 3), order="F")
        if frame_idx % 10 == 0:
            transform_origins.append(origin.transform(transform))
            # Compute points in sight
            visible_segs, visible_ratios = calculate_pts_in_sight(
                obj_pts_all,
                np.linalg.inv(transform),
                intrinsics,
                pts_seg=obj_sem_seg,
                seg_counts=obj_seg_counts,
            )
            if visible_segs:
                for visible_seg, visible_ratio in zip(visible_segs, visible_ratios):
                    part_name = f"{data_id}_{visible_seg}"
                    if part_name not in anno_3d_dict:
                        continue
                    bbox_3d = anno_3d_dict[f"{data_id}_{visible_seg}"]["bbox_3d"]
                    axis_3d = anno_3d_dict[f"{data_id}_{visible_seg}"]["axis_3d"]
                    bbox = AxisBBox3D(
                        bbox_3d[:3], bbox_3d[3:6], bbox_3d[6:9], axis_3d.reshape(2, 3)
                    )

                    # Wrap up the annotation
                    annotation = {}
                    annotation["transform"] = transform.tolist()
                    annotation["intrinsics"] = intrinsics.tolist()
                    annotation["frame_idx"] = frame_idx
                    annotation["data_id"] = data_id
                    annotation["part_name"] = part_name
                    annotation["bbox_3d"] = bbox.get_bbox_array().tolist()
                    annotation["axis_3d"] = bbox.get_axis_array().tolist()
                    annotation["visible_ratio"] = visible_ratio

                    # Add the annotation to priority queue
                    if visible_seg not in priority_queue_dict:
                        priority_queue_dict[visible_seg] = PriorityQueue()
                    priority_queue_dict[visible_seg].put(
                        (-visible_ratio, frame_idx, annotation)
                    )

    # Save max_queue_len annotations which has max visibility for each part
    annotations = []
    for part_id, queue in priority_queue_dict.items():
        for i in range(10):
            _, __, annotation = queue.get()
            annotations.append(annotation)
            if True:
                frame_idx = annotation["frame_idx"]
                bbox_3d = annotation["bbox_3d"]
                axis_3d = annotation["axis_3d"]
                data_id = annotation["data_id"]
                transform = np.asarray(annotation["transform"]).reshape(4, 4)
                intrinsics = np.asarray(annotation["intrinsics"]).reshape(3, 3)
                # Load the image
                image_file = os.path.join(
                    multi_scan_dir, "output", data_id, "video", f"{frame_idx+1:04d}.png"
                )
                image = cv2.imread(image_file)
                # Project the annotation
                bbox = AxisBBox3D(
                    bbox_3d[:3],
                    bbox_3d[3:6],
                    bbox_3d[6:9],
                    np.array(axis_3d).reshape(2, 3),
                )
                bbox_points = bbox.get_bbox_3d_proj(
                    intrinsics,
                    np.linalg.inv(transform),
                    0,
                    1,
                    image.shape[1],
                    image.shape[0],
                )
                axis_points = bbox.get_axis_3d_proj(
                    intrinsics,
                    np.linalg.inv(transform),
                    0,
                    1,
                    image.shape[1],
                    image.shape[0],
                )
                # Draw the bbox
                for i, j in zip(
                    [0, 0, 0, 1, 1, 2, 2, 6, 5, 4, 3, 3],
                    [1, 2, 3, 6, 7, 7, 5, 4, 4, 7, 6, 5],
                ):
                    cv2.line(
                        image,
                        (
                            int(bbox_points[i][0] * image.shape[1]),
                            int(bbox_points[i][1] * image.shape[0]),
                        ),
                        (
                            int(bbox_points[j][0] * image.shape[1]),
                            int(bbox_points[j][1] * image.shape[0]),
                        ),
                        (0, 255, 0),
                        4,
                        lineType=cv2.LINE_8,
                    )
                # Draw the lines
                for i, j in zip([0], [1]):
                    cv2.line(
                        image,
                        (
                            int(axis_points[i][0] * image.shape[1]),
                            int(axis_points[i][1] * image.shape[0]),
                        ),
                        (
                            int(axis_points[j][0] * image.shape[1]),
                            int(axis_points[j][1] * image.shape[0]),
                        ),
                        (0, 0, 255),
                        4,
                        lineType=cv2.LINE_8,
                    )
                # Draw the lines
                # reshape the height to be 480 for vis
                image = cv2.resize(
                    image, (int(480 * image.shape[1] / image.shape[0]), 480)
                )
                # cv2.imshow("image", image)
                # cv2.waitKey(0)
                cv2.imwrite(
                    os.path.join(export_dir, f"{part_id:04d}_{frame_idx+1:04d}.png"),
                    image,
                )

    # Export the annotation
    with open(os.path.join(export_dir, f"{data_id}.json"), "w") as f:
        json.dump(annotations, f)
