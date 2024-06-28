import cv2
import random
import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import rotate, translate
from shapely.geometry import MultiPoint
from scipy.spatial.transform import Rotation as R


####################################### Constants #######################################
eps = 1e-6

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


####################################### 3D Structure ########################################
class AxisBBox3D:
    """Axis 3D BBox."""

    def __init__(self, center=None, extent=None, rot_vec=None, axis=None) -> None:
        self.extent = np.ones(3) if extent is None else np.array(extent)
        self.center = np.zeros(3) if center is None else np.array(center)
        self.R = np.eye(3) if rot_vec is None else R.from_rotvec(rot_vec).as_matrix()
        self.axis = np.zeros([2, 3]) if axis is None else axis

    def create_axis_aligned_from_points(self, points):
        min_bound = np.min(points, axis=0)
        max_bound = np.max(points, axis=0)
        self.center = (min_bound + max_bound) / 2
        self.extent = max_bound - min_bound
        self.R = np.eye(3)
        self.axis = np.array([[0, 0, max_bound[2]], [0, 0, min_bound[2]]])

    def create_minimum_axis_aligned_bbox(self, points):
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        obb = pcd.get_minimal_oriented_bounding_box()
        self.center = np.asarray(obb.center)
        self.extent = np.asarray(obb.extent)
        self.R = np.asarray(obb.R)

    def create_minimum_projected_bbox(self, points):
        points_xy = points[:, :2]
        # Minimum bounding box in 2D
        multipoint = MultiPoint(points_xy)
        min_rect = multipoint.minimum_rotated_rectangle
        rect_coords = list(min_rect.exterior.coords)
        rect_coords = np.array(rect_coords)[:, :2]
        edges = [
            rect_coords[i + 1] - rect_coords[i] for i in range(len(rect_coords) - 1)
        ]
        longest_edge = max(edges, key=lambda x: np.linalg.norm(x))  # Use this as x-axis
        shortest_edge = min(edges, key=lambda x: np.linalg.norm(x))
        longest_edge_len = np.linalg.norm(longest_edge)
        shortest_edge_len = np.linalg.norm(shortest_edge)
        center_xy = np.mean(rect_coords[:4, :], axis=0)
        min_z = np.min(points[:, 2])
        max_z = np.max(points[:, 2])
        if np.abs(max_z - min_z) < 0.01:
            max_z += 0.01
            min_z -= 0.01
        center = np.array([center_xy[0], center_xy[1], (min_z + max_z) / 2])
        x_axis = np.array([longest_edge[0], longest_edge[1], 0])
        z_axis = np.array([0, 0, max_z - min_z])
        x_axis = x_axis / (np.linalg.norm(x_axis) + eps)
        z_axis = z_axis / (np.linalg.norm(z_axis) + eps)
        y_axis = np.cross(z_axis, x_axis)

        if (longest_edge_len - shortest_edge_len) / (shortest_edge_len + eps) < 0.1:
            # Could be a circle
            min_bound = np.min(points, axis=0)
            max_bound = np.max(points, axis=0)
            axis_aligned_extent = max_bound - min_bound
            longest_edge_len_aa = np.max(axis_aligned_extent[:2])
            shortest_edge_len_aa = np.min(axis_aligned_extent[:2])
            if (
                np.abs(longest_edge_len_aa - longest_edge_len)
                / (longest_edge_len + eps)
                < 0.1
            ) and (
                np.abs(shortest_edge_len_aa - shortest_edge_len)
                / (shortest_edge_len + eps)
                < 0.1
            ):
                # aa box is similar to box
                return self.create_axis_aligned_from_points(points)

        self.center = np.array(center)
        self.extent = np.array([longest_edge_len, shortest_edge_len, max_z - min_z])
        self.R = np.array([x_axis, y_axis, z_axis]).T
        self.axis = np.array([[0, 0, max_z], [0, 0, min_z]])

    def create_joint_aligned_bbox(self, points, joint_origin, joint_axis):
        # Rotate the points to the joint axis
        joint_axis = joint_axis / (np.linalg.norm(joint_axis) + eps)
        joint_T = np.eye(4)
        joint_T[:3, 3] = joint_origin
        joint_T_x_axis = np.array([1, 0, 0])
        if np.abs(np.dot(joint_axis, joint_T_x_axis)) > 0.9:
            joint_T_x_axis = np.array([0, 1, 0])
        joint_T_y_axis = np.cross(joint_axis, joint_T_x_axis)
        joint_T_y_axis = joint_T_y_axis / (np.linalg.norm(joint_T_y_axis) + eps)
        joint_T_x_axis = np.cross(joint_T_y_axis, joint_axis)
        joint_T_x_axis = joint_T_x_axis / (np.linalg.norm(joint_T_x_axis) + eps)
        joint_T[:3, :3] = np.array([joint_T_x_axis, joint_T_y_axis, joint_axis]).T
        joint_T_inv = np.linalg.inv(joint_T)
        points = points @ joint_T_inv[:3, :3].T + joint_T_inv[:3, 3]
        # Minimum bounding box in 3D
        self.create_minimum_projected_bbox(points)
        # Rotate the bbox to the joint axis
        self.rotate(joint_T[:3, :3], (0, 0, 0))
        self.translate(joint_T[:3, 3])

    def get_min_bound(self):
        return self.center - self.extent / 2

    def get_max_bound(self):
        return self.center + self.extent / 2

    def rotate(self, R, center=np.array([0, 0, 0])):
        self.center = R @ (self.center - center) + center
        self.R = R @ self.R
        self.axis = self.axis @ R.T

    def translate(self, T):
        self.center += T
        self.axis += T

    def transform(self, T):
        self.center = T[:3, :3] @ self.center + T[:3, 3]
        self.R = T[:3, :3] @ self.R
        self.axis = self.axis @ T[:3, :3].T + T[:3, 3]

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

    def get_bbox_array(self):
        return np.concatenate(
            [self.center, self.extent, R.from_matrix(self.R).as_rotvec()]
        )

    def get_axis_array(self):
        return self.axis.flatten()

    def get_pose(self):
        pose = np.eye(4)
        pose[:3, :3] = self.R
        pose[:3, 3] = self.center
        return pose

    ########## Annotation tools ##########
    def get_bbox_3d_proj(
        self, intrinsics, camera_pose, depth_min, depth_max, img_width, img_height
    ):
        """BBox 3d projected to pixel space."""
        # Get the 3D bbox points
        bbox_points = self.get_points()
        return AxisBBox3D.project_points(
            bbox_points,
            intrinsics,
            camera_pose,
            depth_min,
            depth_max,
            img_width,
            img_height,
        )

    def get_axis_3d_proj(
        self, intrinsics, camera_pose, depth_min, depth_max, img_width, img_height
    ):
        """Axis 3d projected to pixel space."""
        return AxisBBox3D.project_points(
            self.axis,
            intrinsics,
            camera_pose,
            depth_min,
            depth_max,
            img_width,
            img_height,
        )

    @staticmethod
    def project_points(
        points, intrinsics, camera_pose, depth_min, depth_max, img_width, img_height
    ):
        proj_points = []
        for point in points:
            point_cam = point @ camera_pose[:3, :3].T + camera_pose[:3, 3]
            point_2d = [-point_cam[0] / point_cam[2], point_cam[1] / point_cam[2]]
            pixel_x = (point_2d[0] * intrinsics[0, 0] + intrinsics[0, 2]) / img_width
            pixel_y = (point_2d[1] * intrinsics[1, 1] + intrinsics[1, 2]) / img_height
            pixel_z = (np.abs(point_cam[2]) - depth_min) / (
                depth_max - depth_min + 1e-6
            )
            pixel = np.array([pixel_x, pixel_y, pixel_z])
            proj_points.append(pixel)
        # proj_points = np.clip(proj_points, 0, 1)
        return np.array(proj_points)

    def get_bbox_o3d(self, min_length=0.05):
        import open3d as o3d

        bbox = o3d.geometry.OrientedBoundingBox()
        bbox.center = self.center
        bbox_extent = np.copy(self.extent)
        bbox_extent = np.clip(bbox_extent, min_length, None)
        bbox.extent = bbox_extent
        bbox.R = self.R
        bbox.color = [1, 0, 0]
        return bbox


def calculate_zy_rotation_for_arrow(vec):
    gamma = np.arctan2(vec[1], vec[0])
    Rz = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1],
        ]
    )
    vec = Rz.T @ vec
    beta = np.arctan2(vec[0], vec[2])
    Ry = np.array(
        [[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]]
    )
    return Rz, Ry


def get_arrow(end, origin=np.array([0, 0, 0]), scale=1.0, color=np.array([1, 0, 0])):
    import open3d as o3d

    if np.all(end == origin):
        assert False, "Arrow end and origin are the same."
    vec = end - origin
    size = np.sqrt(np.sum(vec**2))

    Rz, Ry = calculate_zy_rotation_for_arrow(vec)
    mesh = o3d.geometry.TriangleMesh.create_arrow(
        cone_radius=size / 17.5 * scale,
        cone_height=size * 0.2 * scale,
        cylinder_radius=size / 30 * scale,
        cylinder_height=size * (1 - 0.2 * scale),
    )
    mesh.rotate(Ry, center=np.array([0, 0, 0]))
    mesh.rotate(Rz, center=np.array([0, 0, 0]))
    mesh.translate(origin)
    mesh.paint_uniform_color(color)
    return mesh


def check_annotations_3d(points, bbox_3d, axis_3d, points_mask, title=""):
    import open3d as o3d
    from matplotlib import pyplot as plt

    bbox_3d = np.array(bbox_3d)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    points_color = plt.get_cmap("tab20")(points_mask % 20)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(points_color)
    bbox = o3d.geometry.OrientedBoundingBox()
    bbox.center = bbox_3d[0:3]
    bbox.R = R.from_rotvec(bbox_3d[6:9]).as_matrix()
    bbox.extent = bbox_3d[3:6]
    bbox.color = [1, 0, 0]

    axis_points = np.array(axis_3d).reshape(2, 3)
    arrow_o3d = get_arrow(axis_points[1], axis_points[0], scale=1, color=[1, 0, 0])
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=[0, 0, 0]
    )
    print(title)
    o3d.visualization.draw_geometries([pcd, bbox, arrow_o3d, origin])


def check_annotations_2d(image, bbox_3d, axis_3d, intrinsic, title=""):
    image = image.copy()
    # Get project
    bbox = AxisBBox3D(
        center=bbox_3d[0:3],
        extent=bbox_3d[3:6],
        rot_vec=bbox_3d[6:9],
        axis=axis_3d.reshape(2, 3),
    )
    bbox_points = bbox.get_bbox_3d_proj(
        intrinsic, np.eye(4), 0, 1, image.shape[1], image.shape[0]
    )
    axis_points = bbox.get_axis_3d_proj(
        intrinsic, np.eye(4), 0, 1, image.shape[1], image.shape[0]
    )
    # Draw the bbox
    for i, j in zip(
        [0, 0, 0, 1, 1, 2, 2, 6, 5, 4, 3, 3], [1, 2, 3, 6, 7, 7, 5, 4, 4, 7, 6, 5]
    ):
        if i == 0 and j == 1:
            color = (0, 0, 255)
        elif i == 0 and j == 2:
            color = (0, 255, 0)
        elif i == 0 and j == 3:
            color = (255, 0, 0)
        else:
            color = (200, 200, 200)
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
            color,
            3,
            lineType=cv2.LINE_8,
        )
    # Draw the lines
    for i, j in zip([0], [1]):
        cv2.arrowedLine(
            image,
            (
                int(axis_points[i][0] * image.shape[1]),
                int(axis_points[i][1] * image.shape[0]),
            ),
            (
                int(axis_points[j][0] * image.shape[1]),
                int(axis_points[j][1] * image.shape[0]),
            ),
            (0, 200, 200),
            3,
        )
    # Reshape the image to height 480
    resize_shape = (480, int(480 / image.shape[0] * image.shape[1]))
    image = cv2.resize(image, resize_shape)
    return image


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


####################################### 3D Utils ########################################
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


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point
