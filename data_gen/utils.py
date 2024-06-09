import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate

import random
import cv2
import numpy as np

colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
    "cyan": (255, 255, 0),
    "magenta": (255, 0, 255),
    "gray": (128, 128, 128),
    "dark_red": (0, 0, 128),
    "dark_green": (0, 128, 0),
    "dark_blue": (128, 0, 0),
    "light_red": (0, 0, 255),
    "light_green": (0, 255, 0),
    "light_blue": (255, 0, 0),
    "orange": (0, 165, 255),
    "purple": (128, 0, 128),
    "brown": (19, 69, 139),
    "pink": (147, 20, 255),
}


def get_rotated_box(cx, cy, w, h, angle):
    # Create a rectangle centered at the origin
    rect = Polygon([(-w / 2, -h / 2), (-w / 2, h / 2), (w / 2, h / 2), (w / 2, -h / 2)])
    # Rotate the rectangle around the origin
    rect = rotate(rect, angle, origin=(0, 0), use_radians=False)
    # Translate the rectangle to the center point (cx, cy)
    rect = translate(rect, cx, cy)
    return rect


def calculate_iou(box1, box2):
    # Calculate intersection area
    intersection_area = box1.intersection(box2).area
    # Calculate union area
    union_area = box1.area + box2.area - intersection_area
    # Compute IoU
    iou = intersection_area / union_area
    return iou


def draw_rotating_bbox(img, bbox, angle, color=(0, 255, 0), thickness=2, text=None):
    """
    Draws a rotated bounding box on the image.

    Parameters:
    - img: The image on which to draw.
    - bbox: The bounding box specification as (center_x, center_y, width, height).
    - angle: The angle of rotation in degrees.
    - color: The color of the bounding box (default is green).
    - thickness: The thickness of the bounding box lines.
    """
    center, size = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
    # Calculate the coordinates of the rectangle's corners after rotation
    if angle is None or angle == "none" or angle == "None":
        # when the angle is None, it is pre-defined Perpendicular to the image
        angle = 0
        if text is None:
            text = "None Angle"
        else:
            text = "Perpendicular_" + text
    # Angle convert to degree
    angle = angle * 180 / np.pi
    rect_coords = cv2.boxPoints(((center[0], center[1]), (size[0], size[1]), angle))
    rect_coords = np.int0(rect_coords)
    # Draw the rotated rectangle
    cv2.drawContours(img, [rect_coords], 0, color, thickness)
    if text is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottom_left = (int(bbox[0] - bbox[2] / 2 - 5), int(bbox[1] + bbox[3] / 2 + 5))
        cv2.putText(img, text, bottom_left, font, 2.5, color, 2, cv2.LINE_AA)
    return img


def draw_rotating_bboxs_with_text(img, list_bbox_name, thickness=2):
    for bbox_name in list_bbox_name:
        color = random.choice(list(colors.values()))
        name = bbox_name[0]
        bbox = bbox_name[1][:4]
        angle = bbox_name[1][4]
        img = draw_rotating_bbox(img, bbox, angle, color, thickness, text=name)
    return img


def intersect_line_bbox(origin, direction, bbox):
    # Unpack the bounding box
    x_center, y_center, width, height = bbox
    x_min = x_center - width / 2
    x_max = x_center + width / 2
    y_min = y_center - height / 2
    y_max = y_center + height / 2

    # Initialize variables
    tmin = float("-inf")
    tmax = float("inf")

    # Check for intersection with each bbox side
    for i in range(2):
        if direction[i] != 0:
            t1 = (x_min - origin[i]) / direction[i]
            t2 = (x_max - origin[i]) / direction[i]

            tmin = max(tmin, min(t1, t2))
            tmax = min(tmax, max(t1, t2))
        elif origin[i] < x_min or origin[i] > x_max:
            return None  # Line parallel to slab, no intersection

    if tmin > tmax:
        return None  # No intersection

    # Calculate the intersection point
    intersection = origin + tmin * direction

    # Check if the intersection is within the y bounds
    if intersection[1] < y_min or intersection[1] > y_max:
        return None

    return intersection


def convert_depth_to_color(depth_img, maintain_ratio=False):
    zero_mask = depth_img == 0
    depth_min = np.min(depth_img[~zero_mask])
    if maintain_ratio:
        depth_max = depth_min + 2000.0
    else:
        depth_max = np.max(depth_img[~zero_mask])
    norm_depth_img = (depth_img - depth_min) / (depth_max - depth_min + 1e-6)
    norm_depth_img[zero_mask] = 0
    norm_depth_img = np.clip(norm_depth_img, 0, 1)
    # Convert the depth image to a color image
    color = cv2.applyColorMap((norm_depth_img * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return color


########################################### 3D Utils ###########################################
def read_ply_ascii(filename):
    with open(filename) as f:
        # Check if the file starts with ply
        if "ply" not in f.readline():
            raise ValueError("This file is not a valid PLY file.")

        # Skip the header, finding where it ends
        line = ""
        while "end_header" not in line:
            line = f.readline()
            if "element vertex" in line:
                num_vertices = int(line.split()[-1])

        # Read the vertex data
        data = np.zeros((num_vertices, 3), dtype=float)
        for i in range(num_vertices):
            line = f.readline().split()
            data[i] = np.array([float(line[0]), float(line[1]), float(line[2])])

    return data


colors_hex = [
    "#FF0000",  # Red
    "#FF7F00",  # Orange
    "#FFFF00",  # Yellow
    "#7FFF00",  # Chartreuse Green
    "#00FF00",  # Green
    "#00FF7F",  # Spring Green
    "#00FFFF",  # Cyan
    "#007FFF",  # Azure
    "#0000FF",  # Blue
    "#7F00FF",  # Violet
    "#FF00FF",  # Magenta
    "#FF007F",  # Rose
    "#7F3F00",  # Chocolate
    "#007F3F",  # Teal
    "#3F007F",  # Indigo
    "#7F007F",  # Purple
    "#7F0000",  # Maroon
    "#003F7F",  # Navy
    "#000000",
]
