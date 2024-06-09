import re
import os
import argparse
from tqdm import tqdm


def modify_urdf(file_path, version_id=0):
    try:
        with open(file_path, "r") as file:
            modified_lines = []
            for line in file:
                if line.strip().startswith("<limit"):
                    # Add 'effort' and 'velocity' attribute if not present
                    if ("effort=" not in line) or ("velocity=" not in line):
                        line = re.sub(r"(<limit)(.*?>)", r'\1 effort="30" velocity="1.0"\2', line)
                    modified_lines.append(line)
                else:
                    modified_lines.append(line)
            # Replace the None with 0
            modified_lines = [re.sub(r"None", r"0", x) for x in modified_lines]
        # Write the modified lines to a new file
        with open(file_path, "w") as file:
            file.writelines(modified_lines)
        return True

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_dir", type=str, default="test_data")
    argparser.add_argument("--data_name", type=str, default="")
    argparser.add_argument("--version", type=int, default=0, help="Version of the dataset")
    args = argparser.parse_args()
    folder_path = args.data_dir
    version_id = int(args.version)

    if args.data_name:
        modify_urdf(f"{folder_path}/{args.data_name}/mobility.urdf", version_id=version_id)
    else:
        valid_dataset_idx_file = f"valid_dataset_idx_v{version_id}.txt"
        valid_dataset_idx_file_full_path = os.path.join(folder_path, valid_dataset_idx_file)
        if os.path.isfile(valid_dataset_idx_file_full_path):
            with open(valid_dataset_idx_file_full_path, "r") as file:
                valid_dataset_idx = file.read().splitlines()

            valid_dataset_idx = [int(x) for x in valid_dataset_idx if x]
        else:
            valid_dataset_idx = []

        dataset_idxs = [int(x) for x in os.listdir(folder_path) if x.isdigit()]

        # avoid duplicates
        dataset_idxs = [str(x) for x in dataset_idxs if x not in valid_dataset_idx]

        for data_name in tqdm(dataset_idxs):
            if os.path.isdir(os.path.join(folder_path, data_name)):
                status = modify_urdf(f"{folder_path}/{data_name}/mobility.urdf")
            else:
                continue
            if status:
                valid_dataset_idx.append(data_name)

        print(f"Valid dataset size: {len(valid_dataset_idx)}")
        with open(valid_dataset_idx_file_full_path, "w") as file:
            file.write("\n".join([str(x) for x in valid_dataset_idx]))
