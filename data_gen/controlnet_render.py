# use batch from:https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching

import torch
import os
import csv
from huggingface_hub import HfApi
from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from transformers import pipeline
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from controlnet_tools import palette

import argparse
import time
from tqdm import tqdm
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

# items: "dataset_id","img_index", "img_name", "img_path", "depth_img_path", "texture_description","category":
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

processed_imgs_track_folder = "processed_imgs_track_v4"
os.makedirs(processed_imgs_track_folder, exist_ok=True)


class DepthControl(Dataset):
    def __init__(
        self, dataset_csv, splits=None, split_idx=None, reuse=True, modality="depth"
    ) -> None:
        assert modality in ["depth", "seg"]
        self.modality = modality
        with open(dataset_csv, "r") as f:
            reader = csv.DictReader(f)
            self.dataset = list(reader)

        keys = {
            "depth": "depth_img_path",
            "seg": "mask_img_path",
        }
        self.seg_category = ["Phone", "Laptop", "Keyboard", "Clock", "Safe"]
        self.img_fetch_key = keys[modality]

        print(f"Use {modality} modality.")
        # sort the csv by dataset_id and img_index
        self.dataset = sorted(
            self.dataset, key=lambda x: (int(x["dataset_id"]), int(x["img_index"]))
        )

        if splits is not None:
            assert split_idx is not None
            self.dataset = self.dataset[
                len(self.dataset)
                * split_idx
                // splits : len(self.dataset)
                * (split_idx + 1)
                // splits
            ]

        print(f"Complete Dataset length: {len(self.dataset)}")
        processed_imgs_track_split = os.path.join(
            processed_imgs_track_folder,
            f"processed_imgs_track_{split_idx}_{modality}.txt",
        )
        if os.path.exists(processed_imgs_track_split):
            with open(processed_imgs_track_split, "r") as f:
                processed_images = set(line.strip() for line in f)
        else:
            processed_images = set()

        if reuse:
            self.filter_dataset(processed_images)
            print("Already processed images: ", len(processed_images))

        self.transform = transform
        print(f"After filtering Dataset length: {len(self.dataset)}")
        print(f"The sample data path: {self.dataset[0]}")

    def filter_dataset(self, processed_images):
        filtered_dataset = []
        for item in self.dataset:
            depth_img_path = item[self.img_fetch_key]
            if depth_img_path in processed_images:
                continue

            if self.modality == "seg":
                item_class = item["category"]
                if item_class not in self.seg_category:
                    continue

            img_save_path = depth_img_path.replace(
                "depth_images", f"controlnet_images_{self.modality}"
            )
            # when already rendered, skip
            img_save_folder = Path(img_save_path).parent
            if (
                os.path.exists(img_save_folder)
                and len(os.listdir(img_save_folder)) > 155
            ):
                print(f"Skip {img_save_folder}")
                continue
            filtered_dataset.append(item)
        self.dataset = filtered_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        dataset_id = item["dataset_id"]
        img_index = item["img_index"]
        depth_img_path = item[self.img_fetch_key]
        texture_description = item["texture_description"]
        depth_img = load_image(depth_img_path)
        if self.modality == "seg":
            depth_img = self.add_color_to_mask(depth_img)

        if self.transform:
            depth_img = self.transform(depth_img)
        return dataset_id, img_index, depth_img, texture_description, depth_img_path

    @staticmethod
    def add_color_to_mask(mask_img, color_palette=palette):
        mask = np.asarray(mask_img)[:, :, 0]
        color_seg = np.zeros(
            (mask.shape[0], mask.shape[1], 3), dtype=np.uint8
        )  # height, width, 3
        for label, color in enumerate(palette):
            color_seg[mask == label, :] = color
        return color_seg


def get_controlnet_pipeline(checkpoint="lllyasviel/control_v11f1p_sd15_depth"):
    print(f"Loading checkpoint: {checkpoint}")
    controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    return pipe


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--modality", type=str, default="seg", choices=["depth", "seg"]
    )
    argparser.add_argument("--batch_size", type=int, default=2)
    argparser.add_argument("--num_images_per_prompt", type=int, default=4)
    argparser.add_argument(
        "--split_idx", type=int, default=0, help="The split index of the dataset"
    )
    argparser.add_argument(
        "--splits", type=int, default=12, help="The number of splits of the dataset"
    )
    argparser.add_argument(
        "--dataset_csv", type=str, default="partnet_pyrender_dataset_v4.csv"
    )
    argparser.add_argument(
        "--reuse", action="store_true", help="Whether to reuse the processed images"
    )
    args = argparser.parse_args()

    checkpoints = {
        "depth": "lllyasviel/control_v11f1p_sd15_depth",
        "seg": "lllyasviel/control_v11p_sd15_seg",
    }
    checkpoint = checkpoints[args.modality]

    replace_path = {"depth": "depth_images", "seg": "mask"}
    original_path = replace_path[args.modality]

    target_path = (
        "controlnet_images"
        if args.modality == "depth"
        else f"controlnet_images_{args.modality}"
    )

    batch_size = args.batch_size
    num_workers = 4
    num_images_per_prompt = args.num_images_per_prompt
    split_idx = args.split_idx
    processed_imgs_track_split = os.path.join(
        processed_imgs_track_folder,
        f"processed_imgs_track_{split_idx}_{args.modality}.txt",
    )
    pipe = get_controlnet_pipeline(checkpoint)

    dataset_csv = args.dataset_csv
    dataset = DepthControl(
        dataset_csv,
        split_idx=split_idx,
        reuse=args.reuse,
        splits=args.splits,
        modality=args.modality,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    generator = torch.manual_seed(0)

    save_interval = 10  # Save captions every 10 batches
    batch_counter = 0
    processed_imgs = []
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        dataset_id, img_index, depth_img, texture_description, img_path = data
        images = pipe(
            list(texture_description),
            num_inference_steps=30,
            generator=generator,
            image=depth_img,
            num_images_per_prompt=num_images_per_prompt,
            batch_size=batch_size,
        ).images
        images_per_depth_image = len(images) // len(depth_img)
        for p in img_path:
            processed_imgs.append(p)

        batch_idx = -1
        for idx_, image in enumerate(images):
            depth_image_idx = idx_ // images_per_depth_image
            render_img_idx = idx_ % images_per_depth_image

            img_save_path = img_path[depth_image_idx].replace(
                original_path, target_path
            )
            #
            img_save_path = Path(img_save_path).parent
            if not os.path.exists(img_save_path):
                os.makedirs(img_save_path)
            if i == 0:
                print(f"DEBUG: Saving to {img_save_path}")
            image.save(
                f"{img_save_path}/{img_index[depth_image_idx]}_{render_img_idx}.png"
            )

        batch_counter += 1
        if batch_counter % save_interval == 0:
            with open(processed_imgs_track_split, "a") as f:
                for img_path_ in processed_imgs:
                    f.write(f"{img_path_}\n")
            processed_imgs = []
