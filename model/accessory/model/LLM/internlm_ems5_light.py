from dataclasses import dataclass
from importlib import resources as impresources
from typing import Optional, List

import accessory
import fairscale.nn.model_parallel.initialize as fs_init
import open_clip
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
from transformers import Blip2Model, Blip2Config, AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class ModelArgs:

    load_pretrained_llm: bool = True
    load_pretrained_visual_encoder: bool = True
    max_seq_len: int = 4096

    max_batch_size: int = 32


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs, with_visual=False):
        super().__init__()
        self.args = args

        self._build_llm()

        self.image_words = 0
        self.past_key_values = None  # for inference
        if with_visual:
            self._build_visual()
            
    def _build_llm(self):
        if self.args.load_pretrained_llm:
            print("loading pretrained llm from HF InternLM 7B")

            try:
                # when setting the HF offline
                self.llm = AutoModelForCausalLM.from_pretrained(
                    # NOTE You need to change this!
                    # "Path_TO/internlm2-7b-original", trust_remote_code=True,
                    "/mnt/petrelfs/huangsiyuan/ckpts/internlm2-7b-original", trust_remote_code=True,
                    attn_implementation="flash_attention_2"
                )
            except Exception as e:
                            # when not setting the HF Offline:
                self.llm = AutoModelForCausalLM.from_pretrained("internlm/internlm2-7b", torch_dtype=torch.float16, trust_remote_code=True)
        else:
            print("Currently, only support the pre-trained version of interlm")
            
    def _build_visual(self):
        example_t = torch.rand(1)  # an example tensor for probing the current default dtype and device
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)

        print("build llama model with openclip")
        if self.args.load_pretrained_visual_encoder:
            self.openclip_convnext_xxl, _, _ = open_clip.create_model_and_transforms(
                "convnext_xxlarge", pretrained="laion2b_s34b_b82k_augreg_soup"
            )
        else:
            self.openclip_convnext_xxl, _, _ = open_clip.create_model_and_transforms(
                "convnext_xxlarge", pretrained=None
            )
        self.openclip_convnext_xxl = self.openclip_convnext_xxl.visual.trunk
        self.openclip_convnext_xxl.head.global_pool = nn.Identity()
        self.openclip_convnext_xxl.head.flatten = nn.Identity()
        self.openclip_convnext_xxl.to(example_t)

        print("build llama model with dinov2")
        if self.args.load_pretrained_visual_encoder:
            self.dinov2_vitg14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14", pretrained=True)
        else:
            self.dinov2_vitg14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14", pretrained=False)
        self.dinov2_vitg14.to(example_t)

        torch.set_default_dtype(default_dtype)

        self.visual_proj = nn.Sequential(
            nn.Linear(3072 + 1536, self.llm.config.hidden_size),
            nn.LayerNorm(self.llm.config.hidden_size),
        )

        self.image_words = (257 + 2) * 5
        self.image_size = 1024
        # add image tags
        self.start_img = nn.Parameter(torch.rand(1, 1, self.llm.config.hidden_size))
        self.end_img = nn.Parameter(torch.rand(1, 1, self.llm.config.hidden_size))

    def get_trainable_params(self, pretrain_stage=False):
        trainable = {}
        no_train_prefix = ["qformer.", "openclip_convnext_xxl.", "clip.", "dinov2_vitg14."]
        for name, para in self.named_parameters():
            if not any([name.startswith(_) for _ in no_train_prefix]):
                trainable[name] = para

        return trainable

    @torch.no_grad()
    def clip_encode_image(self, x):
        # modified from CLIP
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1,
                                                                                  x.shape[-1], dtype=x.dtype,
                                                                                  device=x.device), x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # preserve all spatial tokens
        x = self.clip.visual.ln_post(x[:, :, :])

        return x

    def encode_image(self, image):
        # images should be of size [bsz, 1024, 1024]
        self.openclip_convnext_xxl.eval()
        self.dinov2_vitg14.eval()

        image_bs = image.size(0)
        mp_world_size = fs_init.get_model_parallel_world_size()
        mp_rank = fs_init.get_model_parallel_rank()
        # assert image_bs % mp_world_size == 0

        n_pad_items = (mp_world_size - image_bs % mp_world_size) % mp_world_size
        padded_image = torch.cat([image, image[:1].expand(n_pad_items, *image.size()[1:])], dim=0)
        padded_image_bs = padded_image.shape[0]

        local_image_bs = padded_image_bs // mp_world_size
        local_image = padded_image[local_image_bs * mp_rank: local_image_bs * (mp_rank + 1)]
        with torch.no_grad():
            local_image_224 = F.interpolate(local_image.half(), size=(224,224), mode="bicubic").to(local_image)
            local_image_448 = F.interpolate(local_image.half(), size=(448,448), mode="bicubic").to(local_image)
            local_parts_224 = [
                local_image_448[..., :224, :224], local_image_448[..., :224, 224:],
                local_image_448[..., 224:, :224], local_image_448[..., 224:, 224:]
            ]
            local_224 = torch.stack([local_image_224] + local_parts_224, dim=1)
            local_224 = local_224.view(-1, *local_224.shape[2:])

            local_image_512 = F.interpolate(local_image.half(), size=(512,512), mode="bicubic").to(local_image)
            local_parts_512 = [
                local_image[..., :512, :512], local_image[..., :512, 512:],
                local_image[..., 512:, :512], local_image[..., 512:, 512:]
            ]
            local_512 = torch.stack([local_image_512] + local_parts_512, dim=1)
            local_512 = local_512.view(-1, *local_512.shape[2:])

            local_convnext_image_feats = self.openclip_convnext_xxl(local_512)
            assert local_convnext_image_feats.size()[1:] == (3072, 16, 16)
            local_convnext_image_feats = local_convnext_image_feats.flatten(-2).permute(0, 2, 1)  # (*, 256, 3072)
            local_convnext_image_feats = torch.cat([
                local_convnext_image_feats.mean(dim=1, keepdim=True),  # add gap as cls token
                local_convnext_image_feats,
            ], dim=1)  # (*, 257, 3072)

            clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
            clip_mean = clip_mean.to(local_image, non_blocking=True).view(3, 1, 1)
            clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
            clip_std = clip_std.to(local_image, non_blocking=True).view(3, 1, 1)
            dinov2_mean = torch.Tensor([0.485, 0.456, 0.406]).to(local_image, non_blocking=True).view(3, 1, 1)
            dinov2_std = torch.Tensor([0.229, 0.224, 0.225]).to(local_image, non_blocking=True).view(3, 1, 1)
            local_dinov2_image_feats = self.dinov2_vitg14.forward_features(
                local_224 * (clip_std / dinov2_std) + (clip_mean - dinov2_mean) / dinov2_std
            )
            local_dinov2_image_feats = torch.cat([
                local_dinov2_image_feats["x_norm_clstoken"].unsqueeze(1),
                local_dinov2_image_feats["x_norm_patchtokens"],
            ], dim=1)
            local_ens_image_feats = torch.cat([
                local_convnext_image_feats,
                local_dinov2_image_feats,
            ], dim=2)  # (*, 257, 4608)

            ens_image_feats = torch.zeros([padded_image_bs*5, *local_ens_image_feats.size()[1:]],
                                          device=local_ens_image_feats.device, dtype=local_ens_image_feats.dtype)
            dist.all_gather_into_tensor(ens_image_feats, local_ens_image_feats,
                                        group=fs_init.get_model_parallel_group())

            ens_image_feats = ens_image_feats[:image_bs*5]

        ens_image_feats = self.visual_proj(ens_image_feats)

        ens_image_feats = ens_image_feats.view(image_bs, 5, *ens_image_feats.shape[1:])
        ens_image_feats = list(torch.unbind(ens_image_feats, dim=1))
        return ens_image_feats

    def forward(self, examples, image=None):
        self.past_key_values = None

        _bsz = examples.shape[0]
        h = self.llm.get_input_embeddings()(examples)
        image_words = 0
        if image is not None:
            h_bos, h_caption = h[:, :1], h[:, 1:]
            l_image_tokens: List = self.encode_image(image)
            for i, image_tokens in enumerate(l_image_tokens):
                image_tokens = torch.cat((self.start_img.expand(_bsz, -1, -1),
                                          image_tokens,
                                          self.end_img.expand(_bsz, -1, -1)), dim=1)
                l_image_tokens[i] = image_tokens
            image_tokens = torch.cat(l_image_tokens, dim=1)
            image_words = image_tokens.shape[1]
            assert image_words == self.image_words, (
                f"{image_words} v.s. {self.image_words}, {[_.shape for _ in l_image_tokens]}"
            )
            h = torch.cat((h_bos, image_tokens, h_caption), dim=1)

        llm_output: CausalLMOutputWithPast = self.llm(inputs_embeds=h, labels=None)
        logits = llm_output.logits
        logits = logits[:, image_words:, :]
        return logits


    @torch.inference_mode()
    def forward_inference(self, examples: torch.Tensor, start_pos: int, image=None):
        _bsz = examples.shape[0]
        h = self.llm.get_input_embeddings()(examples)
        if image is not None:
            assert start_pos == 0
            h_bos, h_caption = h[:, :1], h[:, 1:]
            l_image_tokens: List = self.encode_image(image)
            for i, image_tokens in enumerate(l_image_tokens):
                image_tokens = torch.cat((self.start_img.expand(_bsz, -1, -1),
                                          image_tokens,
                                          self.end_img.expand(_bsz, -1, -1)), dim=1)
                l_image_tokens[i] = image_tokens
            image_tokens = torch.cat(l_image_tokens, dim=1)
            image_words = image_tokens.shape[1]
            assert image_words == self.image_words, (
                f"{image_words} v.s. {self.image_words}, {[_.shape for _ in l_image_tokens]}"
            )
            h = torch.cat((h_bos, image_tokens, h_caption), dim=1)
            self.past_key_values = None
        else:
            if start_pos == 0:
                self.past_key_values = None


        llm_output: CausalLMOutputWithPast = self.llm(
            inputs_embeds=h,
            labels=None,
            past_key_values=self.past_key_values,
            use_cache=True)
        self.past_key_values = llm_output.past_key_values
        logits = llm_output.logits[:, -1].float()
        return logits

    def get_basic_block_classes(self):
        return [type(self.llm.model.layers[0])]

    def get_quant_blocklist(self) -> List[str]:
        vision_prefixes = [
            "clip.", "openclip_convnext_xxl.", "dinov2_vitg14.", "qformer.",
            "visual_proj.", "qformer_proj.",
        ]
        blocklist = []
        for n, m in self.named_modules():
            if any(n.startswith(x) for x in vision_prefixes):
                blocklist.append(n)
        return blocklist
