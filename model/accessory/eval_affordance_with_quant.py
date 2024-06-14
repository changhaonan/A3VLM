import argparse
import itertools
import json
import os
import random
import time
from functools import partial
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

import sys
import os

sys.path.append(os.path.abspath(__file__).rsplit('/', 2)[0])
from model.meta import MetaModel
from data.conversation.lib import conv_templates
from accessory.data.conversation import default_conversation
from accessory.data.transform import load_objaverse_point_cloud
from accessory.util.tensor_type import default_tensor_type, promote_trainable_params_to_fp32


import argparse
import torch
import torch.distributed as dist
import gradio as gr

from PIL import Image
import PIL.ImageFile as ImageFile

# Increase the limit for decompression
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None  # Disable the decompression bomb limit

from util import misc
from fairscale.nn.model_parallel import initialize as fs_init

from util.tensor_parallel import load_tensor_parallel_model_list
# from util.quant import quantize
from data.bbox_util import Expand2square, denorm_bboxes
import re
import json
import numpy as np
import cv2

global_config = {
    'temperature': 0.1,
    'top_p': 0.75
}

partnet_dataset_root = '/mnt/petrelfs/XXXXX/data/ManipVQA2/jsons_vqa_tasks_fix_angle/'
partnet_dataset_root_depth = "/mnt/petrelfs/XXXXX/data/ManipVQA2/vqa_tasks_v11_0508"
partnet_dataset_root_8points = "/mnt/petrelfs/XXXXX/data/ManipVQA2/vqa_tasks_v15_521_3d"

ds_collections = {
    "demo": {
        "train": "/mnt/petrelfs/XXXXX/data/ManipVQA2/eval_demo/demo_det_all.json",
        "test": "/mnt/petrelfs/XXXXX/data/ManipVQA2/eval_demo/demo_det_all.json",
        "max_new_tokens": 2048,
        "use_answer_extractor": True,
    },
    
    "demo2": {
       "train": "/mnt/petrelfs/XXXXX/data/ManipVQA2/eval_demo/demo_joint_rec.json",
        "test": "/mnt/petrelfs/XXXXX/data/ManipVQA2/eval_demo/demo_joint_rec.json",
        "max_new_tokens": 1024,
        "use_answer_extractor": True,
    }
}

def collate_fn(batches):
    questions = [_['question'] for _ in batches]
    if 'question_id' in batches[0]:
        question_ids = [_['question_id'] for _ in batches]
    else:
        question_ids = [idx for idx, _ in enumerate(questions)]
        
    annotations = [_['annotation'] for _ in batches]
    input_image = torch.cat([_['image'] for _ in batches])
    image_paths = [_['image_path'] for _ in batches]

    # input_ids = tokenizer(questions, return_tensors='pt', padding='longest')

    return input_image, question_ids, questions, annotations, image_paths

import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class PadToSquare:
    def __init__(self, background_color):
        """
        pad an image to squre (borrowed from LLAVA, thx)
        :param background_color: rgb values for padded pixels, normalized to [0, 1]
        """
        self.bg_color = tuple(int(x * 255) for x in background_color)

    def __call__(self, img: Image.Image):
        width, height = img.size
        if width == height:
            return img
        elif width > height:
            result = Image.new(img.mode, (width, width), self.bg_color)
            result.paste(img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(img.mode, (height, height), self.bg_color)
            result.paste(img, ((height - width) // 2, 0))
            return result


def T_padded_resize(size=224):
    t = transforms.Compose([
        PadToSquare(background_color=(0.48145466, 0.4578275, 0.40821073)),
        transforms.Resize(
            size, interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])
    return t
import torchvision.transforms as transforms


class VQADataset(torch.utils.data.Dataset):

    def __init__(self, train, test, prompt, img_root, img_size=224, remove_space=False, sampled_num=5000, result=None):

        with open(test, "r") as f:
            self.test = json.load(f)
        if "train" in test or len(self.test) > sampled_num:
            # first shuffle, then sample
            random.shuffle(self.test)
            sampled_num = min(len(self.test), sampled_num)
            self.test = self.test[:sampled_num]
        
        if result is not None:
            # when image_path and question is the same, contine
            print(f"before remove, test length: {len(self.test)}")
            for test_item in self.test:
                img_path = test_item["image"]                    
                for result_item in result:
                    if img_path == result_item["image"]:
                        self.test.remove(test_item)
                        break    
            print(f"after remove, test length: {len(self.test)}")
        
        # self.test = open(test).readlines()
        self.prompt = prompt
        self.img_root = img_root
        self.print = False
        self.transform_val = T_padded_resize(img_size)
        self.remove_space = remove_space

    def __len__(self):
        return len(self.test)

    def __getitem__(self, idx):
        # print(self.test[idx].strip())
        data = self.test[idx]
        image_path = data["image"]
        question = data["conversations"][0]["value"]
        question_id = idx
        annotation = data["conversations"][1]["value"]
        
        if ".npy" not in image_path:
            try:
                image = Image.open(image_path).convert('RGB')
            except OSError as e:
                tmp_idx = random.randint(0, len(self.test) - 1)
                print(f"opening {image_path} failed with error {e} and randomly sample a new one")
                data = json.loads(self.test[tmp_idx].strip())
                image_path, _, _, _, _ = data['image'], data[
                    'question'], data['question_id'], data.get('answer', None), data.get('ocr_tokens', '')
                image = Image.open(image_path).convert('RGB')
                question_id = 99999 # fake question id to indicate this is a fake image
    
            image = self.transform_val(image).unsqueeze(0)
        else:
            image = load_objaverse_point_cloud(image_path)
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        
        conv = default_conversation()
        conv.load_qas([[question, None]])
        prompt = conv.get_prompt()

        if self.remove_space:
            question = question.replace("###Assistant: ", "###Assistant:")
            
        item_dict =  {
            'question': prompt,
            'question_id': question_id,
            'annotation': annotation,
            'image': image,
            "image_path": image_path
        }
        if idx < 2:
            print(f"question: {question}")
            print(f"prompt: {prompt}")
            print(f"annotation: {annotation}")
            print(f"image_path: {image_path}")
            print(f"image: {image.shape}")
            
        return item_dict


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)

def normalize_number(x):
    if x > 100:
        return x / 1000
    elif x > 10:
        return x / 100
    elif x >= 1:
        return x / 10
    else:
        return x

def format_bounding_box(answer):
    # Remove any non-numeric and non-comma characters, clean extra whitespace
    cleaned_answer = re.sub(r'[^\d,]', '', answer.replace(" ", ""))

    # Function to insert dot before the last three digits of a number
    def insert_dot(match):
        number = match.group(0)
        return number[:-3] + '.' + number[-3:]
    
    # Apply the function to all numbers in the string
    formatted_answer = re.sub(r'\d{4,}', insert_dot, cleaned_answer)
    
    # Split into individual numbers and convert to float, assuming they are now correctly formatted
    bbox = [float(n) for n in formatted_answer.split(',') if n]
    bbox = [normalize_number(x) for x in bbox]
    return bbox

if __name__ == '__main__':

    def get_args_parser():
        parser = argparse.ArgumentParser('Single-turn (conversation) demo', add_help=False)
        # Model parameters
        parser.add_argument('--llama_type', default='llama_qformerv2', type=str, metavar='MODEL',
                            help='type of llama')
        parser.add_argument('--llama_config', default='/path/to/params.json', type=str, nargs="+",
                            help='Path to llama model config')
        parser.add_argument('--tokenizer_path', type=str, default="../tokenizer.model",
                            help='path to tokenizer.model')
        parser.add_argument('--img_root', type=str, default="./data/nocaps/images",
                            help='path to tokenizer.model')
        parser.add_argument('--annotation_path', type=str, default="./data/nocaps/nocap_val.json",
                            help='path to tokenizer.model')

        parser.add_argument('--pretrained_path', default='/path/to/pretrained', type=str, nargs="+",
                            help='directory containing pre-trained checkpoints')

        parser.add_argument('--device', default='cuda',
                            help='device for inference')
        parser.add_argument('--model_parallel_size', default=1, type=int)

        parser.add_argument('--world_size', default=1, type=int,
                            help='number of distributed processes')
        parser.add_argument('--batch_size', default=4, type=int)
        parser.add_argument('--num_workers', default=4, type=int)
        parser.add_argument('--local_rank', default=-1, type=int)
        parser.add_argument('--seed', default=1, type=int)
        parser.add_argument('--dist_on_itp', action='store_true')
        parser.add_argument('--dist_url', default='env://',
                            help='url used to set up distributed training')
        parser.add_argument('--quant', action="store_true", default=False,
                            help="enable quantization")
        parser.add_argument('--dataset', default='vqav2_val', type=str)
        parser.add_argument("--input_size", type=int, default=224)
        parser.add_argument('--ocr_question', action='store_true')
        parser.add_argument('--prompt', default='vqav2_val', type=str)
        parser.add_argument('--addition_flag', default=None, type=str)
        parser.add_argument("--remove_space", action="store_true", default=False)
        parser.add_argument("--sampled_num", type=int, default=200)
        

        return parser

    args = get_args_parser().parse_args()
    add_flag = args.addition_flag
    remove_space = args.remove_space
    

    misc.init_distributed_mode(args)
    fs_init.initialize_model_parallel(args.model_parallel_size)
    model = MetaModel(args.llama_type, args.llama_config, args.tokenizer_path, with_visual=True)

    print(f"load pretrained from {args.pretrained_path}")
    load_tensor_parallel_model_list(model, args.pretrained_path)
    tokenizer = model.tokenizer
    # print("Model = %s" % str(model))
    # model enabled done
    if args.quant:
        from accessory.util.quant import quantize
        print("Quantizing model to 4bit!")
        from transformers.utils.quantization_config import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig.from_dict(
            config_dict={
                "load_in_8bit": False,
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
            },
            return_unused_kwargs=False,
        )
        quantize(model, quantization_config)
        model.cuda()
    else:
        model.bfloat16().cuda()
    model.eval()
    dataset_name = args.dataset
    if dataset_name == "all":
        dataset_names = list(ds_collections.keys())
    else:
        dataset_names = [dataset_name]
    
    result = None
    for dataset_name in dataset_names:
        save_path = f'vqa_logs/{add_flag}'
        os.makedirs(save_path, exist_ok=True)
        results_file = f'{save_path}/{dataset_name}.json'
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                result = json.load(f)
        
        
        print(f"evaluating on {dataset_name}")
        log_path = f'results/{args.pretrained_path[0].split("ckpts")[-1].replace("/", "_")}.txt'
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                content = f.read()
            if f'dataset config: {dataset_name}' in content:
                exit(0)
        prompt = ''

        if 'prompt' in ds_collections[dataset_name]:
            prompt = ds_collections[dataset_name]['prompt']
        else:
            prompt = prompt
        
        random.seed(args.seed)
        dataset = VQADataset(
            train=ds_collections[dataset_name]['train'],
            test=ds_collections[dataset_name]['test'],
            img_root=args.img_root,
            # tokenizer=tokenizer,
            prompt=prompt,
            img_size=args.input_size,
            remove_space=remove_space,
            sampled_num=args.sampled_num,
            result=result
        )

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            # sampler=InferenceSampler(len(dataset)),
            sampler=torch.utils.data.SequentialSampler(dataset),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=partial(collate_fn),
        )

        outputs = []
        max_gen_len = ds_collections[dataset_name]['max_new_tokens']
        gen_t = global_config['temperature']
        top_p = global_config['top_p']
        answer_extractor = ds_collections[dataset_name].get('use_answer_extractor', False)
        failed_tasks = []
        with torch.no_grad():
            for image, question_ids, _prompt, annotations, image_paths in tqdm(dataloader):
                if dist.get_rank() == 0:
                    dist.barrier()
                    dist.broadcast_object_list([_prompt, image, max_gen_len, gen_t, top_p])

                    image = image.cuda()
                    # print(f'\ninput: {_prompt[0]}\n')
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        results = model.generate(_prompt, image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)

                    for question_id, answer, annotation, question, image_path in zip(question_ids, results,
                                                            annotations, _prompt, image_paths):
                        if answer_extractor:
                            # remove symbol
                            answer = answer.split('###')[0]
                            answer = answer.replace('.', '').strip()
                            if len(answer.strip().split(' ')) > 0:
                                ans_pattern = ['answer is']
                                for a_p in ans_pattern:
                                    if a_p in answer:
                                        try:
                                            answer_extracted = re.findall(f'{a_p}[ ]*[a-zA-Z0-9.]+', answer)[0]
                                            answer_extracted = re.sub(a_p, '', answer_extracted)
                                            answer = answer_extracted.strip()
                                        except Exception as e:
                                            print(e)
                                            print(answer)
                                            answer = answer.strip()

                        failed_flag = False
                        dt_bbox = format_bounding_box(answer)
                        if len(dt_bbox) != 4:
                            failed_flag = True
                        elif dt_bbox[0] > dt_bbox[2] or dt_bbox[1] > dt_bbox[3]:
                            failed_flag = True
                        outputs.append({
                            'answer': answer,
                            "format_answer": dt_bbox,
                            'annotation': annotation,
                            "question": question,
                            "image": image_path,
                            "fail": failed_flag
                        })
                        if failed_flag:
                            failed_tasks.append(
                                [image, question_ids, _prompt, annotations, image_paths]
                            )
                else:
                    dist.barrier()
                    input_data = [None for _ in range(5)]
                    dist.broadcast_object_list(input_data)
                    _prompt, image, max_gen_len, gen_t, top_p = input_data
                    image = image.cuda()
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        _ = model.generate(_prompt, image, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p)

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_outputs = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_outputs, outputs)

        merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]
        
        if torch.distributed.get_rank() == 0:
            
            if result:
                merged_outputs.extend(result)
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            time_prefix_2 = time.strftime('%y%m%d%H', time.localtime())
            json.dump(merged_outputs, open(results_file, 'w'),
                    ensure_ascii=False)  # save to results
        