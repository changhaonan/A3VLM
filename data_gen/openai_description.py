import os
import json
import re
import ast
import argparse
import time

from tqdm import tqdm

os.environ["http_proxy"] = "http://127.0.0.1:15732"
os.environ["https_proxy"] = "http://127.0.0.1:15732"

from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
assert api_key is not None, "Please set OPENAI_API_KEY environment variable first."
client = OpenAI(api_key=api_key)


def get_openai_response(model_cat, previous_description=None):
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are a good assistant, skilled in providing accurate prompt for stable diffusion.",
            },
            {
                "role": "user",
                "content": "I want to use stable diffusion to draw a table, please give me a prompt. Only give me the prompt, and nothing else",
            },
            {
                "role": "assistant",
                "content": "A wooden table, with black and yellow stripes.",
            },
            {
                "role": "user",
                "content": "I want to use stable diffusion to draw a cup, please give me two prompts with different styles. Note that the target object is the daily-use item. Only give me the prompt in the python list format, and nothing else",
            },
            {
                "role": "assistant",
                "content": "['A glass cup, which is purple and have a golden edge.', 'A wooden cup, with black and yellow stripes.']",
            },
            {
                "role": "user",
                "content": f"I want to use stable diffusion to draw a {model_cat}, please give me ten prompts with different styles. Note that the target object is the daily-use item. I already have the descriptions like {previous_description}, please avoid repeatness. Only give me the newly-generated prompts in a list, and nothing else.",
            },
        ],
    )

    return completion.choices[0].message.content


def get_generated_prompts(cato, prompts_folder):
    if os.path.exists(f"{prompts_folder}/{cato}.text"):
        with open(f"{prompts_folder}/{cato}.text", "r") as f:
            prompts = f.read().splitlines()

        prompts_all_in_string = ""
        for idx, prompt in enumerate(prompts):
            prompts_all_in_string += f"{idx+1}. {prompt}\n"

        prompts_all_in_string = prompts_all_in_string.join(prompts)
        return prompts
    else:
        return None


def extract_descriptions(response):
    try:
        descriptions_list = ast.literal_eval(response.strip())
    except (ValueError, SyntaxError) as e:
        descriptions_list = []
    return descriptions_list


def save_prompts(cato, prompts, prompts_folder):
    file_path = f"{prompts_folder}/{cato}.text"
    # if exist, use append mode
    if os.path.exists(file_path):
        mode = "a"
    else:
        mode = "w"

    with open(file_path, mode) as file:
        for prompt in prompts:
            file.write(prompt + "\n")


def get_response_and_save(target_cato, num_prompts=20, prompts_folder="texture_prompts"):
    if os.path.exists(f"{prompts_folder}/{cato}.text"):
        with open(f"{prompts_folder}/{cato}.text", "r") as f:
            prompts = f.read().splitlines()
        if len(prompts) >= num_prompts:
            # print("Enough prompts generated.")
            return False

    previous_description = get_generated_prompts(target_cato, prompts_folder)
    generated_description = get_openai_response(target_cato, previous_description)
    prompts = extract_descriptions(generated_description)
    if len(prompts) > 0:
        save_prompts(target_cato, prompts, prompts_folder)
    else:
        print(f"No prompts generated for {target_cato}.")
        print(f"Response: {generated_description}")


def get_model_cates(dataset_path):
    # get all meta.json with os.walk
    all_cato = set()
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file == "meta.json":
                with open(os.path.join(root, file), "r") as f:
                    meta = json.load(f)
                model_cate = meta["model_cat"]
                all_cato.add(model_cate)
    return all_cato


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_cat", type=str, default=None)
    argparser.add_argument("--num_prompts", type=int, default=20)
    argparser.add_argument("--dataset_path", type=str, default="dataset")
    argparser.add_argument("--output_dir", type=str, default="output")
    args = argparser.parse_args()

    prompts_folder = os.path.join(args.output_dir, "texture_prompts")
    if not os.path.exists(prompts_folder):
        os.makedirs(prompts_folder)

    if args.model_cat is None:
        model_cates = list(get_model_cates(args.dataset_path))
        print(f"Generating prompts for {len(model_cates)} categories.")
        for cato in tqdm(model_cates):
            generate = get_response_and_save(cato, args.num_prompts, prompts_folder)
            if generate:
                time.sleep(5)
    else:
        get_response_and_save(args.model_cat, args.num_prompts, prompts_folder)
