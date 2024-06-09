import os
import shutil
import json
import cv2
import random
from PIL import Image
from ast import literal_eval
import numpy as np
import re
from scipy.spatial.transform import Rotation as R

from vqa_config import NONE_PLACEHOLDER, DET_ALL_ROT_INSTRUCT, REC_JOINT_ROT_INSTRUCT, REG_STATUS_INSTRUCT, REC_SINGLE_LINK_INSTRUCT, GROUNDING_ACTIONS_INSTRUCT, REC_JOINT_ROT_EXT_INSTRUCT
from vqa_config import (
    DET_ALL_BBOX_3D_INSTRUCT,
    REC_JOINT_3D_INSTRUCT,
    REG_STATUS_3D_INSTRUCT,
    REC_SINGLE_LINK_3D_INSTRUCT,
    GROUNDING_ACTIONS_3D_INSTRUCT,
    DELIMIMTER_AXIS_3D_START,
    DELIMIMTER_AXIS_3D_END,
    DELIMIMTER_BOX_3D_START,
    DELIMIMTER_BOX_3D_END,
    DELIMIMTER_ROTATED_BOX_DEPTH_START,
    DELIMIMTER_ROTATED_BOX_DEPTH_END,
    DELIMIMTER_DEPTH_START,
    DELIMIMTER_DEPTH_END,
)
from vqa_config import DELIMIMTER_ROTATED_BOX_START, DELIMIMTER_ROTATED_BOX_END, DELIMIMTER_BOX_START, DELIMIMTER_BOX_END
from utils import draw_rotating_bbox, draw_rotating_bboxs_with_text, colors
from point_render import BBox3D

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

number_words_map_dict = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven"}


############################################# 2D VQA #############################################
def normalize_box_values(box):
    pass


def extract_info_from_string(input_string):
    if "None" in input_string:
        logger.debug(f"The input string is {input_string}")
    input_string = input_string.replace("None", str(NONE_PLACEHOLDER))
    # Pattern to match a tagged box with the format: "<tag>content</tag>[numbers]"
    tagged_box_pattern = r"<(\w+)>([^<]+)</\1>\[([\d.,-]+)\]"

    results = []
    # Find all tagged boxes
    none_flag = False
    tagged_boxes = re.findall(tagged_box_pattern, input_string)
    for tag, content, box_str in tagged_boxes:
        box_values = []
        for num_idx, num in enumerate(box_str.split(",")):
            if np.abs(NONE_PLACEHOLDER - float(num)) > 0.1:
                if float(num) > 1 and num_idx < 4:
                    num = int(num) / 100
                    box_values.append(num)
                else:
                    box_values.append(float(num))
            else:
                box_values.append(None)
                none_flag = True
        results.append((content, box_values))

    if none_flag:
        logger.debug(f"Input_string: {input_string}")
        logger.debug(f"parsed_result: {results}")

    return results


def get_actual_rotated_box(scaled_box, image_width=960, image_height=960, pad_x0=0, pad_y0=0):
    # Unpack the scaled bounding box
    scx, scy, sw, sh, sangle = scaled_box

    # Convert the scaled values back to pixel coordinates
    cx = scx * image_width
    cy = scy * image_height
    w = sw * image_width
    h = sh * image_height

    # Subtract the padding from the center coordinates
    cx_padded = cx - pad_x0
    cy_padded = cy - pad_y0

    # The angle remains the same since it's independent of the scale
    angle = sangle

    return cx_padded, cy_padded, w, h, angle


# NOTE, we are assuming the img with 960*960
def get_pad_value(height, width):
    if height > width:
        pad_x0 = int((height - width) / 2)
        pad_y0 = 0
        width = height
    else:
        pad_x0 = 0
        pad_y0 = int((width - height) / 2)
        height = width
    return pad_x0, pad_y0


def get_scaled_box(box, image_width=960, image_height=960, pad_x0=0, pad_y0=0):
    x0, y0, w, h = box
    x0 = x0 + pad_x0
    y0 = y0 + pad_y0
    sx0 = x0 / image_width
    sy0 = y0 / image_height
    sx1 = (x0 + w) / image_width
    sy1 = (y0 + h) / image_height
    return sx0, sy0, sx1, sy1


def get_scaled_rotated_box(box, image_width=960, image_height=960, pad_x0=0, pad_y0=0, str_rep=True, with_depth=False):
    # Unpack the original bounding box
    if not with_depth:
        cx, cy, w, h, angle = box[:5]
    else:
        cx, cy, w, h, angle, depth1, depth2 = box

    # Add the padding to the center coordinates
    cx_padded = cx + pad_x0
    cy_padded = cy + pad_y0

    # Scale the center coordinates
    scx = cx_padded / image_width
    scy = cy_padded / image_height

    # Scale the width and height
    sw = w / image_width
    sh = h / image_height

    # The angle remains the same since it's independent of the scale
    # sangle = angle
    # if str_rep and type(angle) is not str:
    #     sangle = "{:.2f}".format(angle)

    if str_rep:
        if not with_depth:
            return "[{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}]".format(scx, scy, sw, sh, angle)
        else:
            return "[{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}]".format(scx, scy, sw, sh, angle, depth1)

    return scx, scy, sw, sh, angle


# add the part detection ability
def create_single_link_rec_rotated_task(link_name, object_rot_box, img_full_path):
    question = REC_SINGLE_LINK_INSTRUCT + link_name
    scaled_rotated_box = get_scaled_rotated_box(object_rot_box, str_rep=True)
    vqa_task = {"image": img_full_path, "conversations": [{"from": "human", "value": question}, {"from": "gpt", "value": scaled_rotated_box}]}
    return vqa_task


# detect all
def create_det_all_rotated_task(list_object_rot_box_name, img_full_path, max_det=10):
    question = DET_ALL_ROT_INSTRUCT
    if len(list_object_rot_box_name) > max_det:
        # get the biggest 10
        list_object_rot_box_name = sorted(list_object_rot_box_name, key=lambda k: k["bbox"][2] * k["bbox"][3], reverse=True)[:max_det]
    object_num_in_words = number_words_map_dict[len(list_object_rot_box_name)]
    if object_num_in_words == "one":
        answer_str = f"There is one manipulable object part with its rotated bounding box: "
    else:
        answer_str = f"There are {object_num_in_words} manipulable object parts with their rotated bounding boxes: "
    for index, object_rot_box_name in enumerate(list_object_rot_box_name):
        object_rot_box = object_rot_box_name["bbox"]
        object_name = object_rot_box_name["link_name"]

        scaled_rotated_box = get_scaled_rotated_box(object_rot_box, str_rep=True)
        if index == len(list_object_rot_box_name) - 1:
            answer_str += DELIMIMTER_ROTATED_BOX_START + object_name + DELIMIMTER_ROTATED_BOX_END + scaled_rotated_box + "."
        else:
            answer_str += DELIMIMTER_ROTATED_BOX_START + object_name + DELIMIMTER_ROTATED_BOX_END + scaled_rotated_box + ","

    if answer_str[-1] == ",":
        answer_str[-1] = "."

    vqa_task = {"image": img_full_path, "conversations": [{"from": "human", "value": question}, {"from": "gpt", "value": answer_str}]}
    return vqa_task


# Refering the joint type and its rotated bounding box linked to the object part
def create_rec_joint_rotated_task(link_info, object_rot_box, joint_type, img_full_path):
    """
    The link info could either be a bounding box or the link name
    """
    if type(link_info) == str:
        question = REC_JOINT_ROT_INSTRUCT.format(REF=link_info)
    else:
        link_info = get_scaled_rotated_box(link_info, str_rep=True)
        question = REC_JOINT_ROT_INSTRUCT.format(REF=link_info)

    scaled_rotated_box = get_scaled_rotated_box(object_rot_box, str_rep=True)
    vqa_task = {
        "image": img_full_path,
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": DELIMIMTER_ROTATED_BOX_START + joint_type + DELIMIMTER_ROTATED_BOX_END + scaled_rotated_box},
        ],
    }
    return vqa_task


def create_rec_joint_rotated_ext_task(link_info, object_rot_box_ext, joint_type, img_full_path):
    """
    The link info could either be a bounding box or the link name
    """
    if type(link_info) == str:
        question = REC_JOINT_ROT_EXT_INSTRUCT.format(REF=link_info)
    else:
        link_info = get_scaled_rotated_box(link_info, str_rep=True)
        question = REC_JOINT_ROT_EXT_INSTRUCT.format(REF=link_info)

    scaled_rotated_box = get_scaled_rotated_box(object_rot_box_ext, str_rep=True, with_depth=True)
    vqa_task = {
        "image": img_full_path,
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": DELIMIMTER_ROTATED_BOX_START + joint_type + DELIMIMTER_ROTATED_BOX_END + scaled_rotated_box},
        ],
    }
    return vqa_task


def create_rec_joint_rotated_ext_task_with_sep_depth(link_info, object_rot_box_ext, joint_type, img_full_path):
    """
    The link info could either be a bounding box or the link name
    """
    if type(link_info) == str:
        question = REC_JOINT_ROT_EXT_INSTRUCT.format(REF=link_info)
    else:
        link_info = get_scaled_rotated_box(link_info, str_rep=True)
        question = REC_JOINT_ROT_EXT_INSTRUCT.format(REF=link_info)

    scaled_rotated_box = get_scaled_rotated_box(object_rot_box_ext, str_rep=True)
    depth_ans_string = "[{:.2f},{:.2f}]".format(object_rot_box_ext[-2], object_rot_box_ext[-1])
    vqa_task = {
        "image": img_full_path,
        "conversations": [
            {"from": "human", "value": question},
            {
                "from": "gpt",
                "value": DELIMIMTER_ROTATED_BOX_START
                + joint_type
                + DELIMIMTER_ROTATED_BOX_END
                + scaled_rotated_box
                + DELIMIMTER_ROTATED_BOX_DEPTH_START
                + depth_ans_string
                + DELIMIMTER_ROTATED_BOX_DEPTH_END,
            },
        ],
    }
    return vqa_task


# the openable link status
def create_reg_status_qa_task(link_info, status, img_full_path):
    """
    The link info could either be a bounding box or the link name.
    """
    if type(link_info) == str:
        question = REC_JOINT_ROT_INSTRUCT.format(REF=link_info)
    else:
        link_info = get_scaled_rotated_box(link_info, str_rep=True)
        question = REC_JOINT_ROT_INSTRUCT.format(REF=link_info)

    if status:
        answer = "Closed"
    else:
        answer = "Opened"

    vqa_task = {
        "image": img_full_path,
        "conversations": [
            {
                "from": "human",
                "value": question,
            },
            {"from": "gpt", "value": answer},
        ],
    }
    return vqa_task


# grounding_task
def replace_link_with_bbox(actions, link_info_list, indexing="bounding_box", anno_meta={}, normalize=False, use_eight_points=False):
    if indexing == "bounding_box" or indexing == "bbox":
        link_bbox_map = {link["link_name_status"]: get_scaled_rotated_box(link[indexing], str_rep=True) for link in link_info_list}
    elif indexing == "bbox_3d":
        link_bbox_map = {link["link_name_status"]: get_bbox_3d(link[indexing], str_rep=True, anno_meta=anno_meta, normalize=normalize, use_eight_points=use_eight_points) for link in link_info_list}
    new_actions = []
    action_str = ""
    for action_index, action in enumerate(actions):
        for link_name, bbox in link_bbox_map.items():
            if link_name in action:
                if "StatusComplete" in action:
                    action_str += "StatusComplete"
                    break
                action_type = action.split("[")[0]
                if indexing == "bbox_3d":
                    action_str += DELIMIMTER_BOX_3D_START + action_type + DELIMIMTER_BOX_3D_END + bbox
                elif indexing == "bounding_box" or indexing == "bbox":
                    action_str += DELIMIMTER_ROTATED_BOX_START + action_type + DELIMIMTER_ROTATED_BOX_END + bbox
                if action_index != len(actions) - 1:
                    action_str += ","
                break
            elif "StatusComplete" in action:
                action_str += "StatusComplete"
                break
        new_actions.append(action_str)
    return action_str


# Function to randomly select a task and replace link names with bounding boxes
def select_random_task(categories, link_info_list, indexing="bounding_box", anno_meta={}, normalize=False, use_eight_points=False):
    category = random.choice(list(categories.keys()))
    task_name = random.choice(list(categories[category].keys()))
    task = categories[category][task_name]
    description = task["description"]
    actions = replace_link_with_bbox(task["actions"], link_info_list, indexing, anno_meta=anno_meta, normalize=normalize, use_eight_points=use_eight_points)
    return description, actions


def create_grounding_task(possible_tasks, link_info_list, img_full_path, indexing="bounding_box", anno_meta={}, normalize=False, use_eight_points=False):
    description, actions = select_random_task(possible_tasks, link_info_list, indexing, anno_meta=anno_meta, normalize=normalize, use_eight_points=use_eight_points)
    if len(actions) < 5:
        # if actions is not valid, just retry
        description, actions = select_random_task(possible_tasks, link_info_list, indexing, anno_meta=anno_meta, normalize=normalize, use_eight_points=use_eight_points)
    if len(actions) < 5:
        return None
    question = GROUNDING_ACTIONS_INSTRUCT + description
    vqa_task = {
        "image": img_full_path,
        "conversations": [
            {
                "from": "human",
                "value": question,
            },
            {"from": "gpt", "value": actions},
        ],
    }
    return vqa_task


def parse_info_from_string(input_string):
    if "[" not in input_string:
        # pure string without bounding box
        # return string
        return 0, input_string
    elif "<rp>" not in input_string:
        # return direct box
        if "None" in input_string:
            # the Perpendicular box
            box_w_angle = extract_info_from_string(input_string.replace(",None", ""))
            box_w_angle.append("None")
            return 1, box_w_angle
        return 1, extract_info_from_string(input_string)
    else:
        # return list(str, box)
        return 2, extract_info_from_string(input_string)


def visualize_single_anno(image_full_path, answer, color=colors["red"]):
    image = cv2.imread(image_full_path)
    height, width, channels = image.shape
    anser_type, valid_answer = parse_info_from_string(answer)
    if anser_type == 0:
        # direct put text on the image
        image = cv2.putText(image, valid_answer, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    elif anser_type == 1:
        # draw the box on the image
        pad_x0, pad_y0 = get_pad_value(height, width)
        if len(valid_answer) != 5:
            image = cv2.putText(image, "FailToParse", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            return image
        actual_box = get_actual_rotated_box(valid_answer, max(width, height), max(width, height), pad_x0=pad_x0, pad_y0=pad_y0)
        bbox = actual_box[:4]
        angle = actual_box[4]
        image = draw_rotating_bbox(image, bbox, angle, color=color)
    elif anser_type == 2:
        answers = []
        for box_name in valid_answer:
            actual_box = get_actual_rotated_box(box_name[1])
            answers.append((box_name[0], actual_box))
        image = draw_rotating_bboxs_with_text(image, answers)

    return image


def visualize_pred(json_path, sample_num=20, save_path="visualization"):
    with open(json_path, "r") as f:
        data = json.load(f)
    data = random.sample(data, sample_num)
    totoal_index = 0
    visual_save_folder = os.path.join(save_path, json_path.split("/")[-1].split(".")[0])
    if not os.path.exists(visual_save_folder):
        os.makedirs(visual_save_folder, exist_ok=True)
    for item in data:
        anno = item["annotation"]
        ans = item["answer"]
        question = item["question"].split("Human")[-1]
        image_full_path = item["image"]

        question = question.split(" the following instruction:")[-1]
        if "Detect all" in question:
            question = question.replace("and provide their rotated bounding boxes.", f"{str(random.randint(0, 2*sample_num))}")
        # dt
        image = visualize_single_anno(image_full_path, ans, color=colors["green"])
        cv2.imwrite(os.path.join(visual_save_folder, str(totoal_index) + "_" + question + ".jpg"), image)
        # gt
        image = visualize_single_anno(image_full_path, anno, color=colors["green"])
        cv2.imwrite(os.path.join(visual_save_folder, str(totoal_index) + "_" + question + "_gt.jpg"), image)
        totoal_index += 1


############################################# 3D VQA #############################################
def normalize_val(val, min_val=-1.0, max_val=1.0, scale=100.0):
    if isinstance(val, list):
        val = np.array(val)
    val = (val - min_val) / (max_val - min_val) * scale
    # round to closest integer
    val = np.round(val).astype(int)
    return val


def unnormalize_val(val, min_val=-1.0, max_val=1.0, scale=100.0):
    if isinstance(val, list):
        val = np.array(val)
    val = val / scale * (max_val - min_val) + min_val
    return val


def get_bbox_3d(bbox_3d, str_rep=True, anno_meta={}, normalize=False, use_eight_points=False):
    if not use_eight_points:
        center = bbox_3d[:3]
        size = bbox_3d[3:6]
        orientation = bbox_3d[6:]
        # normalize output
        if normalize:
            center = normalize_val(center, min_val=-1.0, max_val=1.0, scale=100.0)
            size = normalize_val(size, min_val=0.0, max_val=2.0, scale=100.0)
            orientation = normalize_val(orientation, min_val=-np.pi, max_val=np.pi, scale=100.0)
        if str_rep:
            return f"[{center[0]:.2f},{center[1]:.2f},{center[2]:.2f},{size[0]:.2f},{size[1]:.2f},{size[2]:.2f},{orientation[0]:.2f},{orientation[1]:.2f},{orientation[2]:.2f}]"
        else:
            return np.concatenate([center, size, orientation])
    else:
        _bbox_3d = BBox3D(bbox_3d[:3], bbox_3d[3:6], bbox_3d[6:])
        # bbox_points = _bbox_3d.get_points()
        bbox_points = _bbox_3d.get_bbox_3d_proj(anno_meta["intrinsics"], anno_meta["camera_pose"], anno_meta["depth_min"], anno_meta["depth_max"], anno_meta["img_width"], anno_meta["img_height"])
        # normalize output
        if normalize:
            bbox_points = normalize_val(bbox_points, min_val=-1.0, max_val=1.0, scale=100.0)
        if str_rep:
            return "[[{:.2f},{:.2f},{:.2f}],[{:.2f},{:.2f},{:.2f}],[{:.2f},{:.2f},{:.2f}],[{:.2f},{:.2f},{:.2f}],[{:.2f},{:.2f},{:.2f}],[{:.2f},{:.2f},{:.2f}],[{:.2f},{:.2f},{:.2f}],[{:.2f},{:.2f},{:.2f}]]".format(
                bbox_points[0][0],
                bbox_points[0][1],
                bbox_points[0][2],
                bbox_points[1][0],
                bbox_points[1][1],
                bbox_points[1][2],
                bbox_points[2][0],
                bbox_points[2][1],
                bbox_points[2][2],
                bbox_points[3][0],
                bbox_points[3][1],
                bbox_points[3][2],
                bbox_points[4][0],
                bbox_points[4][1],
                bbox_points[4][2],
                bbox_points[5][0],
                bbox_points[5][1],
                bbox_points[5][2],
                bbox_points[6][0],
                bbox_points[6][1],
                bbox_points[6][2],
                bbox_points[7][0],
                bbox_points[7][1],
                bbox_points[7][2],
            )
            # return f"[[{bbox_points[0][0]},{bbox_points[0][1]},{bbox_points[0][2]}],[{bbox_points[1][0]},{bbox_points[1][1]},{bbox_points[1][2]}],[{bbox_points[2][0]},{bbox_points[2][1]},{bbox_points[2][2]}],[{bbox_points[3][0]},{bbox_points[3][1]},{bbox_points[3][2]}],[{bbox_points[4][0]},{bbox_points[4][1]},{bbox_points[4][2]}],[{bbox_points[5][0]},{bbox_points[5][1]},{bbox_points[5][2]}],[{bbox_points[6][0]},{bbox_points[6][1]},{bbox_points[6][2]}],[{bbox_points[7][0]},{bbox_points[7][1]},{bbox_points[7][2]}]]"
        else:
            return bbox_points


def get_axis_3d(axis_3d, str_rep=True, anno_meta={}, normalize=True):
    if normalize:
        axis_3d = normalize_val(axis_3d, min_val=-1.0, max_val=1.0, scale=100.0)
    if str_rep:
        axis_3d = BBox3D.project_points(axis_3d, anno_meta["intrinsics"], anno_meta["camera_pose"], anno_meta["depth_min"], anno_meta["depth_max"], anno_meta["img_width"], anno_meta["img_height"])
        return f"[{axis_3d[0][0]:.2f},{axis_3d[0][1]:.2f},{axis_3d[0][2]:.2f},{axis_3d[1][0]:.2f},{axis_3d[1][1]:.2f},{axis_3d[1][2]:.2f}]"
    else:
        return axis_3d


def get_axis_proj(axis_3d_proj, str_rep=True):
    if str_rep:
        return f"[{axis_3d_proj[0]:.2f},{axis_3d_proj[1]:.2f}]"
    else:
        return axis_3d_proj


def create_single_link_3d_rec_task(link_name, bbox_3d, pcd_full_path, anno_meta={}, normalize=False, use_eight_points=False):
    question = REC_SINGLE_LINK_3D_INSTRUCT + link_name
    bbox_3d = get_bbox_3d(bbox_3d, str_rep=True, anno_meta=anno_meta, normalize=normalize, use_eight_points=use_eight_points)
    vqa_task = {"image": pcd_full_path, "conversations": [{"from": "human", "value": question}, {"from": "gpt", "value": bbox_3d}]}
    return vqa_task


def create_3d_rec_joint_task(link_info_3d, axis_3d, joint_type, pcd_full_path, anno_meta={}, normalize=False, use_eight_points=False, axis_3d_proj=None):
    """
    The link info could either be a 3d-axis or the link name
    """
    if type(link_info_3d) == str:
        question = REC_JOINT_3D_INSTRUCT.format(REF=link_info_3d)
    else:
        link_info_3d = get_bbox_3d(link_info_3d, str_rep=True, anno_meta=anno_meta, normalize=normalize, use_eight_points=use_eight_points)
        question = REC_JOINT_3D_INSTRUCT.format(REF=link_info_3d)

    if axis_3d_proj is None:
        axis_3d = get_axis_3d(axis_3d, str_rep=True, anno_meta=anno_meta, normalize=normalize)
    else:
        axis_3d = get_axis_proj(axis_3d_proj, str_rep=True)
    vqa_task = {
        "image": pcd_full_path,
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": DELIMIMTER_AXIS_3D_START + joint_type + DELIMIMTER_AXIS_3D_END + axis_3d},
        ],
    }
    return vqa_task


def create_3d_reg_status_qa_task(link_info, status, pcd_full_path, anno_meta={}, normalize=False, use_eight_points=False):
    """
    The link info could either be a bounding box or the link name.
    """
    if type(link_info) == str:
        question = REC_JOINT_3D_INSTRUCT.format(REF=link_info)
    else:
        link_info = get_bbox_3d(link_info, str_rep=True, anno_meta=anno_meta, normalize=normalize, use_eight_points=use_eight_points)
        question = REC_JOINT_3D_INSTRUCT.format(REF=link_info)

    if status:
        answer = "Closed"
    else:
        answer = "Opened"

    vqa_task = {
        "image": pcd_full_path,
        "conversations": [
            {
                "from": "human",
                "value": question,
            },
            {"from": "gpt", "value": answer},
        ],
    }
    return vqa_task


def create_det_all_bbox_3d_task(list_object_3d_box_name, pcd_full_path, max_det=10, anno_meta={}, normalize=False, use_eight_points=False):
    question = DET_ALL_BBOX_3D_INSTRUCT
    if len(list_object_3d_box_name) > max_det:
        # get the biggest 10; the list is sorted by the area of the 2D bounding box
        list_object_3d_box_name = sorted(list_object_3d_box_name, key=lambda k: k["bbox"][2] * k["bbox"][3], reverse=True)[:max_det]
    object_num_in_words = number_words_map_dict[len(list_object_3d_box_name)]
    if object_num_in_words == "one":
        answer_str = f"There is one manipulable object part with its 3d bounding box: "
    else:
        answer_str = f"There are {object_num_in_words} manipulable object parts with their 3d bounding boxes: "
    for index, object_rot_box_name in enumerate(list_object_3d_box_name):
        object_bbox_3d = object_rot_box_name["bbox_3d"]
        object_name = object_rot_box_name["link_name"]

        bbox_3d = get_bbox_3d(object_bbox_3d, str_rep=True, anno_meta=anno_meta, normalize=normalize, use_eight_points=use_eight_points)
        if index == len(list_object_3d_box_name) - 1:
            answer_str += DELIMIMTER_BOX_3D_START + object_name + DELIMIMTER_BOX_3D_END + bbox_3d + "."
        else:
            answer_str += DELIMIMTER_BOX_3D_START + object_name + DELIMIMTER_BOX_3D_END + bbox_3d + ","

    if answer_str[-1] == ",":
        answer_str[-1] = "."

    vqa_task = {"image": pcd_full_path, "conversations": [{"from": "human", "value": question}, {"from": "gpt", "value": answer_str}]}
    return vqa_task


def create_3d_grounding_task(possible_tasks, link_info_list, pcd_full_path, indexing="bbox_3d", anno_meta={}, normalize=False, use_eight_points=False):
    description, actions = select_random_task(possible_tasks, link_info_list, indexing, anno_meta=anno_meta, normalize=normalize, use_eight_points=use_eight_points)
    if len(actions) < 5:
        # if actions is not valid, just retry
        description, actions = select_random_task(possible_tasks, link_info_list, indexing, anno_meta=anno_meta, normalize=normalize, use_eight_points=use_eight_points)
    if len(actions) < 5:
        return None
    question = GROUNDING_ACTIONS_3D_INSTRUCT + description
    vqa_task = {
        "image": pcd_full_path,
        "conversations": [
            {
                "from": "human",
                "value": question,
            },
            {"from": "gpt", "value": actions},
        ],
    }
    return vqa_task


if __name__ == "__main__":
    json_folder_name = "affordance_v1_epoch2"
    visual_save_folder = f"{json_folder_name}_visualization"
    if not os.path.exists(visual_save_folder):
        os.makedirs(visual_save_folder, exist_ok=True)
    else:
        shutil.rmtree(visual_save_folder)
        os.makedirs(visual_save_folder, exist_ok=True)
    json_folder = f"/mnt/petrelfs/huangsiyuan/LLaMA2-Accessory/accessory/vqa_logs/{json_folder_name}"
    jsons = os.listdir(json_folder)
    sample_num = 20
    totoal_index = 0
    for json_file in jsons:
        visualize_pred(os.path.join(json_folder, json_file), sample_num, visual_save_folder)
