"""Generate grounding tasks for the given dataset using OpenAI API.
"""

import os
import json
from openai import OpenAI
import xml.etree.ElementTree as ET
from vqa_config import open_close_status
from tqdm import tqdm
import time

api_key = os.getenv("OPENAI_API_KEY")
assert api_key is not None, "Please set OPENAI_API_KEY environment variable first."
client = OpenAI(api_key=api_key)


def load_link_semantic(file_path, joint_types, open_close_link=None):
    parsed_data = []
    idx_str_list = []
    idx_str = ""
    with open(file_path, "r") as file:
        for line_idx, line in enumerate(file):
            parts = line.strip().split(" ")
            if len(parts) == 3:
                senmantic_name = parts[2]
                joint_type = parts[1]
                joint_type_from_urdf = joint_types[line_idx]
                cur_idx_str = f"{joint_type_from_urdf}_{senmantic_name}"
                if cur_idx_str in idx_str_list:
                    continue
                if open_close_link and senmantic_name in open_close_link:
                    parsed_data.append({"name": "opened_" + senmantic_name, "joint_type": joint_type_from_urdf, "status": "open"})
                    parsed_data.append({"name": "closed_" + senmantic_name, "joint_type": joint_type_from_urdf, "status": "close"})
                else:
                    parsed_data.append({"name": senmantic_name, "joint_type": joint_type_from_urdf, "status": "N.A."})
                idx_str_list.append(cur_idx_str)

    idx_str_list = list(set(sorted(idx_str_list)))
    for idx_str_ele in idx_str_list:
        idx_str += idx_str_ele + "_"
    return parsed_data, idx_str


def load_joint_type_semantic(urdf_file):
    """
    Load Joint Type from the URDF, with prefer for prismatic, revolute, and fixed joint types, etc
    """
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    # Find all joint elements in the document
    joints = root.findall("joint")

    # Iterate through each joint and extract its name and type
    joint_types = []
    for joint in joints:
        joint_type = joint.get("type")
        joint_types.append(joint_type)

    return joint_types


def load_history_generated(cato, idx_str):
    save_file = f"{save_path}/{cato}_{idx_str}.json"
    if os.path.exists(save_file):
        with open(save_file, "r") as f:
            data = json.load(f)

        task_num = 0
        tasks = data[cato]
        for link_part in tasks:
            task_num += len(tasks[link_part])
        return data, task_num
    else:
        return [], 0


def save_history_generated(cato, idx_str, data):
    save_file = f"{save_path}/{cato}_{idx_str}.json"
    # if exist, first load, then append the new ones to the same keys
    if os.path.exists(save_file):
        with open(save_file, "r") as f:
            history = json.load(f)

        history_tasks = history[cato]
        cur_tasks = data[cato]

        for link_part in history_tasks:
            if link_part not in cur_tasks:
                cur_tasks[link_part] = history_tasks[link_part]
            else:
                # merge the history and current tasks
                cur_link_tasks_dict = cur_tasks[link_part]
                history_link_tasks_dict = history_tasks[link_part]
                for task in history_link_tasks_dict:
                    if task not in cur_link_tasks_dict:
                        cur_link_tasks_dict[task] = history_link_tasks_dict[task]

    full_data = {cato: cur_tasks}
    with open(save_file, "w") as f:
        json.dump(full_data, f)


def get_openai_response(prompt, previous_description, class_name, link_info):
    prompt = prompt.replace("{OBJECT_CLASS}", class_name).replace("{LINK_INFO}", str(link_info)).replace("{HISTORY_GENERATION}", str(previous_description))
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a good assistant, skilled in creating a grounding training dataset for the given daily-use furniture class. Provide output in valid JSON.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return completion.choices[0].message.content


def get_response_and_save(prompt, dataset_idx=101564, max_tasks=20):
    dataset_path = os.path.join(dataset_root, str(dataset_idx))
    semantic_file = os.path.join(dataset_path, "semantics.txt")
    mata_json = os.path.join(dataset_path, "meta.json")
    if not os.path.exists(mata_json):
        return "SKIP"
    with open(mata_json, "r") as f:
        meta = json.load(f)
    cato = meta["model_cat"]
    joint_types = load_joint_type_semantic(os.path.join(dataset_path, "mobility.urdf"))
    link_semantics, idx_str = load_link_semantic(semantic_file, joint_types, open_close_status.get(cato))
    # print(link_semantics, idx_str)
    hist, task_num = load_history_generated(cato, idx_str)
    if task_num >= max_tasks:
        print(f"Already generated {task_num} tasks for {cato} {idx_str}")
        return "SKIP"
    response = get_openai_response(prompt, hist, cato, link_semantics)
    response = response.split("```json")[-1].split("```")[0].replace("```json", "").replace("```", "").strip().rstrip()
    try:
        result = json.loads(response)
        save_history_generated(cato, idx_str, result)
        return "Success"
    except Exception as e:
        file_save_name = os.path.join(save_failure_path, f"{cato}_{idx_str}.txt")
        with open(file_save_name, "w") as f:
            f.write(response)
        print(f"Error: {e}")
        # print(f"{response}")
        return "Fail"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_path", type=str, default="prompts/grounding_task_generation.txt")
    parser.add_argument("--dataset_root", type=str, required=True)
    args = parser.parse_args()

    prompt_path = args.prompt_path
    dataset_root = args.dataset_root

    with open(prompt_path, "r") as f:
        prompt = f.read()
    save_path = "openai_grounding_tasks"
    os.makedirs(save_path, exist_ok=True)
    save_failure_path = os.path.join(save_path, "failed_responses")
    os.makedirs(save_failure_path, exist_ok=True)

    all_sub_folders = os.listdir(dataset_root)
    all_sub_folders_indexs = [int(folder) for folder in all_sub_folders]
    all_sub_folders_indexs.sort()

    processed_dataset_idx = []
    processed_dataset_text_file = "openai_grounding_idxes.txt"
    if os.path.exists(processed_dataset_text_file):
        with open(processed_dataset_text_file, "r") as f:
            for line in f:
                processed_dataset_idx.append(int(line.strip()))
    print(f"Processed {len(processed_dataset_idx)} datasets")

    for index in tqdm(all_sub_folders_indexs):
        if index in processed_dataset_idx:
            continue
        status = get_response_and_save(prompt, index, max_tasks=30)
        if status == "Success":
            with open(processed_dataset_text_file, "a") as f:
                f.write(f"{index}\n")
            time.sleep(5)
        elif status == "SKIP":
            with open(processed_dataset_text_file, "a") as f:
                f.write(f"{index}\n")
        else:
            time.sleep(5)
            continue
