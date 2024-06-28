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
    model = "sd2"
    modality = "depth"
    materials = ["wood", "metal", "plastic", "marble", "china"]
    statuses = ["new", "old", "clean", "dirty"]
    pipeline = get_controlnet_pipeline(model=f"{model}-{modality}")
    for i in range(2):
        depth_image = load_image(
            f"/home/harvey/Project/A3VLM/output/3596/video/bg_{modality}_{i}.png"
        )
        color_image = load_image(
            f"/home/harvey/Project/A3VLM/output/3596/video/bg_color_{i}.png"
        )
        material = rng.choice(materials)
        status = rng.choice(statuses)
        prompt = f"{status} {material} cabinet. Realistic render with high quality texture and lighting."
        n_propmt = "bad, deformed, borken."
        if model == "sd3":
            image = pipeline(
                prompt=prompt,
                negative_prompt="",
                num_inference_steps=28,
                controlnet_conditioning_scale=0.5,
                control_image=depth_image,
            ).images[0]
        elif model == "sd2":
            # Convert to depth to tensor
            depth_image = cv2.imread(
                f"/home/harvey/Project/A3VLM/output/3596/video/bg_depth_{i}.png",
                cv2.IMREAD_UNCHANGED,
            )
            depth_image = torch.from_numpy(depth_image).to("cuda").unsqueeze(0)
            image = pipeline(
                prompt=prompt,
                image=color_image,
                negative_prompt=n_propmt,
                strength=0.7,
                depth_map=depth_image,
            ).images[0]
        elif model == "sd1.5":
            image = pipeline(
                prompt=prompt,
                image=depth_image,
                negative_prompt=n_propmt,
                strength=0.7,
            ).images[0]
        # Save the image
        image.save(f"/home/harvey/Project/A3VLM/output/3596/video/bg_sd_0_{i}.png")
