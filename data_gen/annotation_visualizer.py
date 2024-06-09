"""Visualizing annotations on points"""

from vqa_task_construction import unnormalize_val
from point_render import BBox3D
import numpy as np
import json
import os
import re
import cv2
from matplotlib import pyplot as plt
from textwrap import wrap
import argparse
from tqdm import tqdm
import random

import logging

# Set the logging level for PIL to WARNING
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)


###################################### 3D Array Representation ######################################
def visualize_joint_3d(points, annotation, save_path=None, meta_info={}):
    question_str = "\n".join(wrap(meta_info["image"] + ": " + annotation["conversations"][0]["value"], 65))
    axis_str = annotation["conversations"][-1]["value"]
    pattern = r"\[(\d+),(\d+),(\d+),(\d+),(\d+),(\d+)\]"
    matches = re.search(pattern, axis_str)
    if matches:
        axis_3d_arr = np.array(list(map(int, matches.groups())))
    else:
        axis_3d_arr = []
        print(f"Error: Cannot find numbers in joint 3d in {meta_info['image']}")
        return False
    axis_3d_arr = unnormalize_val(axis_3d_arr, min_val=-1, max_val=1, scale=100)
    fig = plt.figure()
    fig.suptitle(question_str)
    ax = fig.add_subplot(111, projection="3d")
    points_color = points[:, -3:] / 255
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points_color, marker="o", s=1, alpha=0.1)
    # plot axis
    ax.quiver(axis_3d_arr[0], axis_3d_arr[1], axis_3d_arr[2], axis_3d_arr[3] - axis_3d_arr[0], axis_3d_arr[4] - axis_3d_arr[1], axis_3d_arr[5] - axis_3d_arr[2], color="b")
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    ax.set_aspect("equal")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
    return True


def visualize_grounding_3d(points, annotation, save_path=None, meta_info={}):
    question_str = "\n".join(wrap(meta_info["image"] + ": " + annotation["conversations"][0]["value"], 65))
    bbox_str = annotation["conversations"][-1]["value"]
    # pattern = r"\[(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+)\]"
    pattern = r"<box>([^<]+)</box>\[(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+)\]"
    matches = re.findall(pattern, bbox_str)
    if matches:
        bbox_datas = [(match[0], np.array(list(map(int, match[1:])))) for match in matches]
    else:
        bbox_datas = []
        print(f"Error: Cannot find numbers in grouding 3d in {meta_info['image']}")
        return False
    if len(bbox_datas) > 1:
        print(f"Warning: More than one bbox in grounding 3d... in {save_path}")
    for bbox_data in bbox_datas:
        action_str, bbox_arr = bbox_data
        center = unnormalize_val(bbox_arr[:3], min_val=-1, max_val=1, scale=100)
        size = unnormalize_val(bbox_arr[3:6], min_val=0, max_val=2, scale=100)
        orientation = unnormalize_val(bbox_arr[6:], min_val=-np.pi, max_val=np.pi, scale=100)
        bbox = BBox3D(center, size, orientation)
        bbox_points = bbox.get_points()
        # plot points & bbox
        fig = plt.figure()
        fig.suptitle(question_str + "; " + action_str)
        ax = fig.add_subplot(111, projection="3d")
        points_color = points[:, -3:] / 255
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points_color, marker="o", s=1, alpha=0.4)
        # Connecting lines
        for i, j in zip([0, 0, 0, 1, 1, 2, 2, 6, 5, 4, 3, 3], [1, 2, 3, 6, 7, 7, 5, 4, 4, 7, 6, 5]):
            ax.plot([bbox_points[i, 0], bbox_points[j, 0]], [bbox_points[i, 1], bbox_points[j, 1]], [bbox_points[i, 2], bbox_points[j, 2]], c="b")
        ax.set_xlabel("X Label")
        ax.set_ylabel("Y Label")
        ax.set_zlabel("Z Label")
        ax.set_aspect("equal")
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()
        return True


def visualize_det_all_3d(points, annotation, save_path=None, meta_info={}):
    question_str = "\n".join(wrap(meta_info["image"] + ": " + annotation["conversations"][0]["value"], 65))
    bbox_str = annotation["conversations"][-1]["value"]
    pattern = r"<box>[^<]*</box>\[(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+)\]"

    # Search for all matches using the pattern
    matches = re.findall(pattern, bbox_str)
    if matches:
        bbox_arrs = np.array([list(map(int, match)) for match in matches])
    else:
        bbox_arrs = []
        print(f"Error: Cannot find numbers in det_all 3d in {meta_info['image']}")
        return False
    # plot points & bbox
    fig = plt.figure()
    fig.suptitle(question_str)
    ax = fig.add_subplot(111, projection="3d")
    points_color = points[:, -3:] / 255
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points_color, marker="o", s=1, alpha=0.4)

    for bbox_arr in bbox_arrs:
        center = unnormalize_val(bbox_arr[:3], min_val=-1, max_val=1, scale=100)
        size = unnormalize_val(bbox_arr[3:6], min_val=0, max_val=2, scale=100)
        orientation = unnormalize_val(bbox_arr[6:], min_val=-np.pi, max_val=np.pi, scale=100)
        bbox = BBox3D(center, size, orientation)
        bbox_points = bbox.get_points()
        # Connecting lines
        for i, j in zip([0, 0, 0, 1, 1, 2, 2, 6, 5, 4, 3, 3], [1, 2, 3, 6, 7, 7, 5, 4, 4, 7, 6, 5]):
            ax.plot([bbox_points[i, 0], bbox_points[j, 0]], [bbox_points[i, 1], bbox_points[j, 1]], [bbox_points[i, 2], bbox_points[j, 2]], c="b")
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")
    ax.set_aspect("equal")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
    return True


###################################### 8 Points Representation ######################################
def visualize_link_3d_8points(image, annotation, save_path=None, meta_info={}):
    question_str = "\n".join(wrap(meta_info["image"] + ": " + annotation["conversations"][0]["value"], 65))
    bbox_str = annotation["conversations"][-1]["value"]
    # Use regular expression to extract the array of numbers
    matches = re.findall(r"\[([-.\d,]+)\]", bbox_str)
    if matches:
        array_str = ",".join(matches)
        array = np.fromstring(array_str, sep=",")
        bbox_points = array.reshape(-1, 3)
    else:
        bbox_points = []
        print(f"Error: Cannot find numbers in grouding 3d in {meta_info['image']}")
        return False
    # if len(bbox_points) > 1:
    #     print(f"Warning: More than one bbox in grounding 3d... in {save_path}")
    # Draw the 8 points
    for i, point in enumerate(bbox_points):
        point_pixel_x = int(point[0] * image.shape[1])
        point_pixel_y = int(point[1] * image.shape[0])
        cv2.circle(image, (point_pixel_x, point_pixel_y), 2, (0, 255, 0), -1)
        cv2.putText(image, str(i), (point_pixel_x, point_pixel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # Draw the lines
    for i, j in zip([0, 0, 0, 1, 1, 2, 2, 6, 5, 4, 3, 3], [1, 2, 3, 6, 7, 7, 5, 4, 4, 7, 6, 5]):
        cv2.line(
            image,
            (int(bbox_points[i][0] * image.shape[1]), int(bbox_points[i][1] * image.shape[0])),
            (int(bbox_points[j][0] * image.shape[1]), int(bbox_points[j][1] * image.shape[0])),
            (0, 255, 0),
            1,
        )
    # Add the question string on top
    question_lines = question_str.split("\n")
    for i, line in enumerate(question_lines):
        cv2.putText(image, line, (10, 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # Save the image
    if save_path:
        cv2.imwrite(save_path, image)


def visualize_joint_3d_proj(image, annotation, save_path=None, meta_info={}):
    question_str = "\n".join(wrap(meta_info["image"] + ": " + annotation["conversations"][0]["value"], 65))
    bbox_str = annotation["conversations"][-1]["value"]
    # Use regular expression to extract the array of numbers
    matches = re.findall(r"\[([-.\d,]+)\]", bbox_str)
    if matches:
        array_str = ",".join(matches)
        array = np.fromstring(array_str, sep=",")
        axis_points = array.reshape(-1, 3)
    else:
        axis_points = []
        print(f"Error: Cannot find numbers in grouding 3d in {meta_info['image']}")
        return False
    # Draw the axis points
    for i, point in enumerate(axis_points):
        point_pixel_x = int(point[0] * image.shape[1])
        point_pixel_y = int(point[1] * image.shape[0])
        cv2.circle(image, (point_pixel_x, point_pixel_y), 2, (0, 255, 0), -1)
        cv2.putText(image, str(i), (point_pixel_x, point_pixel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # Draw the lines
    for i, j in zip([0], [1]):
        cv2.line(
            image,
            (int(axis_points[i][0] * image.shape[1]), int(axis_points[i][1] * image.shape[0])),
            (int(axis_points[j][0] * image.shape[1]), int(axis_points[j][1] * image.shape[0])),
            (0, 255, 0),
            1,
        )
    # Add the question string on top
    question_lines = question_str.split("\n")
    for i, line in enumerate(question_lines):
        cv2.putText(image, line, (10, 20 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # Save the image
    if save_path:
        cv2.imwrite(save_path, image)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--vqa_task_folder", type=str, default="/mnt/petrelfs/XXXXX/data/ManipVQA2/vqa_tasks_v7_0428_sd")
    argparser.add_argument("--visual_save_path", type=str, default=None)
    argparser.add_argument("--num_samples", type=int, default=200)

    args = argparser.parse_args()
    vqa_task_folder = args.vqa_task_folder
    visual_save_path = args.visual_save_path
    if not visual_save_path:
        visual_save_path = os.path.join(vqa_task_folder, "visualization")
    if not os.path.exists(visual_save_path):
        os.makedirs(visual_save_path)
    else:
        import shutil

        shutil.rmtree(visual_save_path)
        os.makedirs(visual_save_path)

    for task in os.listdir(vqa_task_folder):
        if ".json" not in task or "3d" not in task:
            continue
        if "reg" in task:
            continue

        save_folder_task = os.path.join(visual_save_path, task.split(".json")[0])
        if not os.path.exists(save_folder_task):
            os.makedirs(save_folder_task)

        annotations = json.load(open(os.path.join(vqa_task_folder, task)))
        random.shuffle(annotations)
        annotations = annotations[: args.num_samples]
        for idx, annotation in tqdm(enumerate(annotations), total=len(annotations)):
            image_file = annotation["image"]
            if not os.path.exists(image_file):
                continue
            if image_file.endswith(".png"):
                points = cv2.imread(image_file)
            elif image_file.endswith(".npy"):
                points = np.load(image_file)
            else:
                raise ValueError(f"Error: Unknown image file format: {image_file}")
            save_path_item = os.path.join(save_folder_task, f"{idx}.png")
            # save_path_item = None
            success = False
            meta_info = {"image": image_file}
            try:
                if task.startswith("joint_3d_rec_tasks"):
                    # success = visualize_joint_3d(points, annotation, save_path_item, meta_info)
                    success = visualize_joint_3d_proj(points, annotation, save_path_item, meta_info)
                    continue
                elif task.startswith("grounding_3d_tasks"):
                    # success = visualize_grounding_3d(points, annotation, save_path_item, meta_info)
                    continue
                elif task.startswith("single_link_3d_rec_tasks"):
                    success = visualize_link_3d_8points(points, annotation, save_path_item, meta_info)
                elif task.startswith("all_parts_3d_det_tasks"):
                    # success = visualize_det_all_3d(points, annotation, save_path_item, meta_info)
                    continue
            except Exception as e:
                print(f"Error: {e}")
                success = False
                continue

            annotation["success"] = success
            annotation["visual_img"] = save_path_item

        parse_result_save = os.path.join(save_folder_task, "parse.json")
        with open(parse_result_save, "w") as f:
            json.dump(annotations, f)
