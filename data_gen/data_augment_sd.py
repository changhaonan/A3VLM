import os
import json
import numpy as np
import torch
import cv2
import requests
from PIL import Image
from diffusers import StableDiffusionDepth2ImgPipeline
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.utils import load_image


############################################# SD1.5-depth #############################################
def get_controlnet_pipeline(
    model="sd1.5-depth", checkpoint="lllyasviel/control_v11f1p_sd15_depth"
):
    if model == "sd1.5-depth":
        print(f"Loading checkpoint: {checkpoint}")
        controlnet = ControlNetModel.from_pretrained(
            checkpoint, torch_dtype=torch.float16
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
    elif model == "sd2-depth":
        pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-depth",
            torch_dtype=torch.float16,
            use_safetensors=True,
        ).to("cuda")
    elif model == "sd3-edge":
        controlnet = SD3ControlNetModel.from_pretrained("InstantX/SD3-Controlnet-Canny")
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", controlnet=controlnet
        )
        pipe.to("cuda", torch.float16)
    else:
        raise ValueError(f"Invalid model: {model}")
    return pipe


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    data_dir = "/home/harvey/Project/A3VLM/dataset/"
    output_dir = "/home/harvey/Project/A3VLM/output/"
    data_id = "46944"
    model = "sd1.5"
    num_poses = 2
    modality = "depth"
    materials = ["wood", "metal", "plastic", "marble"]
    statuses = ["new", "old", "clean", "dirty"]
    model_cat = "wardrobe"
    pipeline = get_controlnet_pipeline(model=f"{model}-{modality}")

    # Get all infos
    info_files = os.listdir(os.path.join(output_dir, data_id))
    info_files = [info for info in info_files if "info" in info]
    num_videos = len(info_files) * num_poses
    for joint_select_idx in range(len(info_files)):
        info_file = os.path.join(output_dir, data_id, f"info_{joint_select_idx}.json")
        with open(info_file, "r") as f:
            info = json.load(f)
        data_ids = info["data_ids"][0]
        bg_data_id = data_ids[-1]
        with open(os.path.join(data_dir, bg_data_id, "meta.json"), "r") as f:
            bg_data_meta = json.load(f)
        bg_model_cat = bg_data_meta["model_cat"]
        for pose_idx in range(num_poses):
            control_image = load_image(
                os.path.join(
                    output_dir,
                    data_id,
                    f"video/bg_{modality}_{joint_select_idx}_{pose_idx}.png",
                )
            )
            color_image = load_image(
                os.path.join(
                    output_dir,
                    data_id,
                    f"video/bg_color_{joint_select_idx}_{pose_idx}.png",
                )
            )
            material = rng.choice(materials)
            status = rng.choice(statuses)
            prompt = f"{status} {bg_model_cat}. Realistic render with high quality texture and lighting."
            n_propmt = "bad, deformed, borken."
            if model == "sd3":
                image = pipeline(
                    prompt=prompt,
                    negative_prompt="",
                    num_inference_steps=28,
                    controlnet_conditioning_scale=0.5,
                    control_image=control_image,
                ).images[0]
            elif model == "sd2":
                # Convert to depth to tensor
                control_image = cv2.imread(
                    os.path.join(
                        output_dir,
                        data_id,
                        f"video/bg_depth_{joint_select_idx}_{pose_idx}.png",
                    ),
                    cv2.IMREAD_UNCHANGED,
                )
                control_image = torch.from_numpy(control_image).to("cuda").unsqueeze(0)
                print(prompt)
                image = pipeline(
                    prompt=prompt,
                    image=color_image,
                    negative_prompt=n_propmt,
                    strength=0.7,
                    depth_map=control_image,
                ).images[0]
            elif model == "sd1.5":
                image = pipeline(
                    prompt=prompt,
                    image=control_image,
                    negative_prompt=n_propmt,
                    strength=0.7,
                ).images[0]
            # Save the image
            image.save(
                os.path.join(
                    output_dir,
                    data_id,
                    f"video/bg_sd_{joint_select_idx}_{pose_idx}.png",
                )
            )
