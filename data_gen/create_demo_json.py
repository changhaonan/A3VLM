import os
import re
import json


def normalize_points(points_str):
    # Remove the outer brackets and split by '],[' to separate each point
    points = points_str.strip("[]").split("],[")

    # Create a list to hold the normalized tuples
    normalized_points = []

    # Iterate through each point, convert to integers, normalize, and store
    for point in points:
        # Split by ',' and convert each string number to an integer
        x, y, z = map(int, point.split(","))

        # Normalize by dividing each coordinate by 100
        normalized_point = [x / 100, y / 100, z / 100]

        # Append the normalized tuple to the list
        normalized_points.append(normalized_point)

    return normalized_points


# Function to parse the string into a dictionary
def parse_string_to_dict(input_str):
    # Regex to find all matches
    pattern = r"<box>(.*?)<\/box>\[(\[[\d,]+?\](?:,\[[\d,]+?\])*)\]"
    matches = re.findall(pattern, input_str)

    bbox_list = []
    for name, points_str in matches:
        # Process points string into a list of tuples
        points = normalize_points(f"[{points_str}]")

        bbox_list.append(points)

    return bbox_list


def create_json_step1():
    root_dir = "/mnt/petrelfs/huangsiyuan/data/ManipVQA2/eval_demo/eval_prepare"
    all_items = os.listdir(root_dir)

    vqa_tasks = []
    for item_ in all_items:
        imaage_full_path = os.path.join(root_dir, item_, "color.png")
        if not os.path.exists(imaage_full_path):
            continue
        vqa_tasks.append(
            {"image": imaage_full_path, "conversations": [{"from": "human", "value": "Detect all manipulable object parts and provide their 3D bounding boxes."}, {"from": "gpt", "value": None}]}
        )

    demo_json_path = "/mnt/petrelfs/huangsiyuan/data/ManipVQA2/eval_demo/demo_det_all.json"
    with open(demo_json_path, "w") as f:
        json.dump(vqa_tasks, f, indent=4)


def create_json_step2():
    step1_infer_result = "/mnt/petrelfs/huangsiyuan/LLaMA2-Accessory/accessory/vqa_logs/affordance_v5_rgb_8points/demo.json"
    with open(step1_infer_result, "r") as f:
        vqa_tasks = json.load(f)

    base_question = "Please provide the joint's type and its 3D axis linked to the object part  "
    step2_tasks = []
    for vqa_res in vqa_tasks:
        image_path = vqa_res["image"]
        bboxs_list = parse_string_to_dict(vqa_res["answer"])

        for bbox_points in bboxs_list:
            bbox_str = "[[{:.2f},{:.2f},{:.2f}],[{:.2f},{:.2f},{:.2f}],[{:.2f},{:.2f},{:.2f}],[{:.2f},{:.2f},{:.2f}],[{:.2f},{:.2f},{:.2f}],[{:.2f},{:.2f},{:.2f}],[{:.2f},{:.2f},{:.2f}],[{:.2f},{:.2f},{:.2f}]]".format(
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

            step2_tasks.append(
                {
                    "image": image_path,
                    "conversations": [
                        {"from": "human", "value": base_question + bbox_str},
                        {"from": "gpt", "value": None},
                    ],
                }
            )

    demo_json_path = "/mnt/petrelfs/huangsiyuan/data/ManipVQA2/eval_demo/demo_joint_rec.json"
    with open(demo_json_path, "w") as f:
        json.dump(step2_tasks, f, indent=4)


if __name__ == "__main__":
    create_json_step2()
