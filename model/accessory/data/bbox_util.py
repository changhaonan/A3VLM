from PIL import Image
from typing import Dict, Any, Tuple, Optional, List, Union
import re

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
# from functools import partial
# from enum import auto, Enum
# from mmengine import DATASETS, TRANSFORMS, METRICS, FUNCTIONS, Registry
import copy
import torch
# from transformers import TrainingArguments, PreTrainedTokenizer
from typing import Dict, Any, Callable, List, Optional, Tuple, Type
import warnings


# bbox settings
PHRASE_ST_PLACEHOLDER = '<ph_st>'
PHRASE_ED_PLACEHOLDER = '<ph_ed>'
IMAGE_PLACEHOLDER = '<image>'
BOXES_PLACEHOLDER = '<boxes>'
EXPR_PLACEHOLDER = '<expr>'
OBJS_PLACEHOLDER = '<objs>'
QUESTION_PLACEHOLDER = '<question>'
POINTS_PLACEHOLDER = '<points>'
Box = List[Union[float, int]]
Boxes = List[Box]
BoxesSeq = List[Boxes]


def expand2square(pil_img, background_color=(255, 255, 255)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def box_xyxy_expand2square(box, *, w, h):
    if w == h:
        return box
    if w > h:
        x1, y1, x2, y2 = box
        y1 += (w - h) // 2
        y2 += (w - h) // 2
        box = x1, y1, x2, y2
        return box
    assert w < h
    x1, y1, x2, y2 = box
    x1 += (h - w) // 2
    x2 += (h - w) // 2
    box = x1, y1, x2, y2
    return box


class Expand2square:
    def __init__(self, background_color=(255, 255, 255)):
        self.background_color = background_color

    def __call__(self, image: Image.Image, labels: Dict[str, Any] = None) -> Tuple[
        Image.Image, Optional[Dict[str, Any]]]:
        width, height = image.size
        processed_image = expand2square(image, background_color=self.background_color)
        if labels is None:
            return processed_image, labels
        if 'boxes' in labels:
            bboxes = [box_xyxy_expand2square(bbox, w=width, h=height) for bbox in labels['boxes']]
            labels['boxes'] = bboxes
        # if 'points' in labels:
        #     points = [point_xy_expand2square(point, w=width, h=height) for point in labels['points']]
        #     labels['points'] = points
        return processed_image, labels


class BoxFormatProcess():
    def __init__(self, box_formatter):
        self.box_formatter = box_formatter

    def map_obj(self, boxes_value: List[List[float]], boxes_seq: List[List[int]]) -> List[List[List[float]]]:
        """
        >>> normalized_boxes = [[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.3, 0.3, 0.3, 0.3]]
        >>> boxes_seq_ = [[3, 1], [2]]
        >>> var = map_obj(normalized_boxes, boxes_seq_)
        >>> assert var == [[[0.3,0.3,0.3,0.3], [0.1,0.1,0.1,0.1]], [0.2,0.2,0.2,0.2]]
        """
        try:
            ret = []
            for boxes in boxes_seq:
                boxes_ret = []
                for box_index in boxes:
                    if isinstance(box_index, (list, tuple)):
                        boxes_ret.append(boxes_value[box_index[0]][box_index[1]])
                    else:
                        boxes_ret.append(boxes_value[box_index])
                ret.append(boxes_ret)
            return ret
        except:
            raise SystemExit(f"error: map obj {boxes_value} {boxes_seq}")

    def norm_box_xyxy(self, box, *, w, h):
        x1, y1, x2, y2 = box

        # Calculate the normalized coordinates with min-max clamping
        norm_x1 = max(0.0, min(x1 / w, 1.0))
        norm_y1 = max(0.0, min(y1 / h, 1.0))
        norm_x2 = max(0.0, min(x2 / w, 1.0))
        norm_y2 = max(0.0, min(y2 / h, 1.0))

        # Return the normalized box coordinates
        normalized_box = (round(norm_x1, 3), round(norm_y1, 3), round(norm_x2, 3), round(norm_y2, 3))
        return normalized_box

    def norm_point_xyxy(self, point, *, w, h):
        x, y = point
        norm_x = max(0.0, min(x / w, 1.0))
        norm_y = max(0.0, min(y / h, 1.0))
        point = norm_x, norm_y
        return point

    def __call__(self, sentence: Dict[str, Any], target: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # box_formatter = preprocessor['target']['boxes']

        normalized_boxes = []
        if target is not None and 'boxes' in target:
            for box in target['boxes']:
                normalized_boxes.append(
                    self.norm_box_xyxy(box, w=target['width'], h=target['height'])
                )
        normalized_points = []
        if target is not None and 'points' in target:
            for point in target['points']:
                normalized_points.append(
                    self.norm_point_xyxy(point, w=target['width'], h=target['height'])
                )

        # convert bboxes_seq
        words: str = sentence['value']
        boxes_seq: List[List[int]] = sentence.get('boxes_seq', None)
        if boxes_seq is not None:
            # map box seq
            boxes_seq = self.map_obj(normalized_boxes, boxes_seq)
            # reformat; replace <boxes> placeholder
            converted = self.box_formatter(words, boxes_seq)
            words = converted
        points_seq: List[List[int]] = sentence.get('points_seq', None)
        if points_seq is not None:
            # map point seq
            points_seq: List[Boxes] = self.map_obj(normalized_points, points_seq)
            # reformat; replace <points> placeholder
            converted = self.box_formatter.call_on_point(words, points_seq)
            words = converted

        if boxes_seq is not None or points_seq is not None:
            sentence['raw_value'] = sentence['value']
            sentence['value'] = words
        return sentence, target


class BoxFormatter:
    def __init__(self, bboxes_token=BOXES_PLACEHOLDER, points_token=POINTS_PLACEHOLDER):
        self.bboxes_token = bboxes_token
        self.points_token = points_token
        # normally the bboxes_token_pat is the same as bboxes_token if u not use some weird token
        self.bboxes_token_pat = re.compile(bboxes_token)
        self.points_token_pat = re.compile(points_token)

    def __call__(self, sentence: str, bboxes_seq: BoxesSeq) -> str:
        all_box = self.bboxes_token_pat.findall(sentence)
        assert len(all_box) == len(bboxes_seq), f"not match. sentence: {sentence}. boxes:{bboxes_seq}"
        if len(all_box) == 0:
            return sentence
        bboxes_strs = [self.format_box(bboxes) for bboxes in bboxes_seq]
        converted = sentence.replace(self.bboxes_token, '{}').format(*bboxes_strs)
        return converted

    def call_on_point(self, sentence: str, points_seq: BoxesSeq) -> str:
        all_box = self.points_token_pat.findall(sentence)
        assert len(all_box) == len(points_seq), f"not match. sentence: {sentence}. boxes:{points_seq}"
        if len(all_box) == 0:
            return sentence
        bboxes_strs = [self.format_point(bboxes) for bboxes in points_seq]
        converted = sentence.replace(self.points_token, '{}').format(*bboxes_strs)
        return converted

    def format_point(self, points) -> str:
        raise NotImplementedError

    def format_box(self, bboxes: Boxes) -> str:
        raise NotImplementedError

    def extract(self, string: str) -> List[Boxes]:
        raise NotImplementedError

    def extract_point(self, string: str) -> List[Boxes]:
        raise NotImplementedError


class PlainBoxFormatter(BoxFormatter):

    def __init__(self, *args, precision=3, use_small_brackets=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.precision = precision
        self.use_small_brackets = use_small_brackets

        small_brackets_pat = re.compile(r'\(\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3}(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3})*\)')
        small_brackets_point_pat = re.compile(r'\(\d(?:\.\d*)?(?:,\d(?:\.\d*)?)(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?))*\)')

        middle_brackets_pat = re.compile(r'\[\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3}(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?){3})*\]')
        middle_brackets_point_pat = re.compile(r'\[\d(?:\.\d*)?(?:,\d(?:\.\d*)?)(?:;\d(?:\.\d*)?(?:,\d(?:\.\d*)?))*\]')

        self.pat = small_brackets_pat if use_small_brackets else middle_brackets_pat
        self.point_pat = small_brackets_point_pat if use_small_brackets else middle_brackets_point_pat

    def format_box(self, boxes: Boxes) -> str:
        box_strs = []
        for box in boxes:
            box_strs.append(','.join([f"{elem:.{self.precision}f}" for elem in box]))
        box_str = ';'.join(box_strs)
        if self.use_small_brackets:
            return "(" + box_str + ")"
        return "[" + box_str + "]"

    def format_point(self, points) -> str:
        return self.format_box(points)

    def extract(self, string: str) -> List[Boxes]:
        """ balabala<boxes>balabala<boxes> -> [boxes, boxes] """
        ret = []
        for bboxes_str in self.pat.findall(string):
            bboxes = []
            bbox_strs = bboxes_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(";")
            for bbox_str in bbox_strs:
                bbox = list(map(float, bbox_str.split(',')))
                bboxes.append(bbox)
            ret.append(bboxes)
        return ret

    def extract_point(self, string: str) -> List[Boxes]:
        """ balabala<boxes>balabala<boxes> -> [boxes, boxes] """
        ret = []
        for bboxes_str in self.point_pat.findall(string):
            bboxes = []
            bbox_strs = bboxes_str.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(";")
            for bbox_str in bbox_strs:
                bbox = list(map(float, bbox_str.split(',')))
                bboxes.append(bbox)
            ret.append(bboxes)
        return ret

# def prepare_interactive(
#         model_args,
#         preprocessor: Dict[str, Any],
# ):
#     conv_args = model_args.conv_args
#     tokenize_kwargs = conv_args.get('tokenize_kwargs', {})
#     conv_template = conv_args.get('conv_template', 'vicuna_v1.1')
#     conv_template = partial(get_conv_template, name=conv_template)
#     transforms = conv_args.get('transforms', None)
#     if transforms is not None:
#         transforms = TRANSFORMS.build(transforms)
#     # process func
#     process_func = {}
#     for k, v in model_args.process_func_args.items():
#         process_func[k] = FUNCTIONS.build(cfg=v)
#
#     ds = SingleImageInteractive(
#         preprocessor=preprocessor,
#         process_func=process_func,
#         tokenize_kwargs=tokenize_kwargs,
#         conv_template=conv_template,
#         training_args=None,
#         transforms=transforms,
#         mode='test',
#     )
#     return ds
#
# class SeparatorStyle(Enum):
#     """Separator styles."""
#
#     ADD_COLON_SINGLE = auto()
#     ADD_COLON_TWO = auto()
#     ADD_SPACE_TWO = auto()
#     NO_COLON_SINGLE = auto()
#     BAIZE = auto()
#     DOLLY = auto()
#     RWKV = auto()
#     PHOENIX = auto()
#     NEW_LINE = auto()
#     BILLA = auto()
#
# class Conversation:
#     """A class that keeps all conversation history."""
#
#     # The name of this template
#     name: str
#     # System prompts
#     system: str
#     # Two roles
#     roles: List[str]
#     # All messages
#     messages: List[List[str]]
#     # Offset of few shot examples
#     offset: int
#     # Separators
#     sep_style: SeparatorStyle
#     sep: str
#     sep2: str = None
#     # Stop criteria (the default one is EOS token)
#     stop_str: str = None
#     # Stops generation if meeting any token in this list
#     stop_token_ids: List[int] = None
#
#     # Used for the state in the gradio servers.
#     # TODO(lmzheng): move this out of this class.
#     conv_id: Any = None
#     skip_next: bool = False
#     model_name: str = None
#
#     def get_prompt(self) -> str:
#         """Get the prompt for generation."""
#         if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
#             ret = self.system + self.sep
#             for role, message in self.messages:
#                 if message:
#                     ret += role + ": " + message + self.sep
#                 else:
#                     ret += role + ":"
#             return ret
#         elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
#             seps = [self.sep, self.sep2]
#             ret = self.system + seps[0]
#             for i, (role, message) in enumerate(self.messages):
#                 if message:
#                     ret += role + ": " + message + seps[i % 2]
#                 else:
#                     ret += role + ":"
#             return ret
#         elif self.sep_style == SeparatorStyle.ADD_SPACE_TWO:
#             seps = [self.sep, self.sep2]
#             ret = self.system + seps[0]
#             for i, (role, message) in enumerate(self.messages):
#                 if message:
#                     ret += role + " " + message + seps[i % 2]
#                 else:
#                     ret += role + ""
#             return ret
#         elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
#             ret = self.system
#             for role, message in self.messages:
#                 if message:
#                     ret += role + message + self.sep
#                 else:
#                     ret += role
#             return ret
#         elif self.sep_style == SeparatorStyle.BAIZE:
#             ret = self.system + "\n"
#             for role, message in self.messages:
#                 if message:
#                     ret += role + message + "\n"
#                 else:
#                     ret += role
#             return ret
#         elif self.sep_style == SeparatorStyle.DOLLY:
#             seps = [self.sep, self.sep2]
#             ret = self.system
#             for i, (role, message) in enumerate(self.messages):
#                 if message:
#                     ret += role + ":\n" + message + seps[i % 2]
#                     if i % 2 == 1:
#                         ret += "\n\n"
#                 else:
#                     ret += role + ":\n"
#             return ret
#         elif self.sep_style == SeparatorStyle.RWKV:
#             ret = self.system
#             for i, (role, message) in enumerate(self.messages):
#                 if message:
#                     ret += (
#                             role
#                             + ": "
#                             + message.replace("\r\n", "\n").replace("\n\n", "\n")
#                     )
#                     ret += "\n\n"
#                 else:
#                     ret += role + ":"
#             return ret
#         elif self.sep_style == SeparatorStyle.PHOENIX:
#             ret = self.system
#             for role, message in self.messages:
#                 if message:
#                     ret += role + ": " + "<s>" + message + "</s>"
#                 else:
#                     ret += role + ": " + "<s>"
#             return ret
#         elif self.sep_style == SeparatorStyle.NEW_LINE:
#             ret = self.system + self.sep
#             for role, message in self.messages:
#                 if message:
#                     ret += role + "\n" + message + self.sep
#                 else:
#                     ret += role + "\n"
#             return ret
#         elif self.sep_style == SeparatorStyle.BILLA:
#             ret = self.system + self.sep
#             for role, message in self.messages:
#                 if message:
#                     ret += role + ": " + message + self.sep
#                 else:
#                     ret += role + ": "  # must be end with a space
#             return ret
#         else:
#             raise ValueError(f"Invalid style: {self.sep_style}")
#
#     def append_message(self, role: str, message: str):
#         """Append a new message."""
#         self.messages.append([role, message])
#
#     def to_gradio_chatbot(self):
#         """Convert the history to gradio chatbot format"""
#         ret = []
#         for i, (role, msg) in enumerate(self.messages[self.offset:]):
#             if i % 2 == 0:
#                 ret.append([msg, None])
#             else:
#                 ret[-1][-1] = msg
#         return ret
#
#     def to_openai_api_messages(self):
#         """Convert the conversation to OpenAI chat completion format."""
#         ret = [{"role": "system", "content": self.system}]
#
#         for i, (_, msg) in enumerate(self.messages[self.offset:]):
#             if i % 2 == 0:
#                 ret.append({"role": "user", "content": msg})
#             else:
#                 if msg is not None:
#                     ret.append({"role": "assistant", "content": msg})
#         return ret
#
#     def copy(self):
#         return Conversation(
#             name=self.name,
#             system=self.system,
#             roles=self.roles,
#             messages=[[x, y] for x, y in self.messages],
#             offset=self.offset,
#             sep_style=self.sep_style,
#             sep=self.sep,
#             sep2=self.sep2,
#             stop_str=self.stop_str,
#             stop_token_ids=self.stop_token_ids,
#             conv_id=self.conv_id,
#             model_name=self.model_name,
#         )
#
#     def dict(self):
#         return {
#             "name": self.name,
#             "system": self.system,
#             "roles": self.roles,
#             "messages": self.messages,
#             "offset": self.offset,
#             "conv_id": self.conv_id,
#             "model_name": self.model_name,
#         }
#
# class SeparatorStyle(Enum):
#     """Separator styles."""
#
#     ADD_COLON_SINGLE = auto()
#     ADD_COLON_TWO = auto()
#     ADD_SPACE_TWO = auto()
#     NO_COLON_SINGLE = auto()
#     BAIZE = auto()
#     DOLLY = auto()
#     RWKV = auto()
#     PHOENIX = auto()
#     NEW_LINE = auto()
#     BILLA = auto()
#
# conv_templates: Dict[str, Conversation] = {}
# def get_conv_template(name: str) -> Conversation:
#     """Get a conversation template."""
#     return conv_templates[name].copy()
# class SingleImageConvDatasetMixin:
#
#     def __init__(
#             self,
#             *args,
#             preprocessor: Dict[str, Any],
#             process_func: Dict[str, Any],
#             conv_template: Callable[[], Conversation] = partial(get_conv_template, name='vicuna_v1.1'),
#             mode='train',
#             tokenize_kwargs: dict = None,
#             training_args: TrainingArguments = None,
#             transforms: Optional[Callable] = None,
#             **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         assert mode in ['train', 'validation', 'test']
#
#         self.preprocessor = preprocessor
#         self.process_func = process_func
#         self.conv_template = conv_template
#         self.mode = mode
#         self.tokenize_kwargs = tokenize_kwargs if tokenize_kwargs is not None else {}
#         self.training_args = training_args
#         self.transforms = transforms
#
#     def __getitem__(self, index, debug_mode=False, return_conv=False) -> Dict[str, Any]:
#         # getitem
#         item = self.get_raw_item(index)
#         image: Image.Image = item.get('image', None)
#         target: Dict[str, Any] = item.get('target', None)
#         raw_conv: List[Dict[str, Any]] = item['conversations']
#
#         # transform
#         assert isinstance(image, list) == isinstance(target, list)
#         multimage_mode = isinstance(image, list)
#         if isinstance(image, list):
#             # TODO: validate raw item
#             transformed_image, transformed_target = [], []
#             for img, tgt in zip(image, target):
#                 if self.transforms is not None and image is not None:
#                     img, tgt = self.transforms(img, tgt)
#                 if tgt is not None:
#                     tgt['width'], tgt['height'] = img.width, img.height
#                 transformed_image.append(img)
#                 transformed_target.append(tgt)
#             image, target = transformed_image, transformed_target
#         else:
#             self.validate_raw_item(item)  # only validate for single image.
#             if self.transforms is not None and image is not None:
#                 image, target = self.transforms(image, target)
#             has_image = 'image' in item and bool(item['image'])
#             has_target = 'target' in item and bool(item['target']) and any(bool(elem) for elem in item['target'].values())
#             if has_target and has_image:
#                 target['width'], target['height'] = image.width, image.height
#
#         # preprocess
#         raw_conv = self.process_conv(raw_conv)
#         raw_conv, image = self.process_conv_multimage(raw_conv, image)
#         raw_conv, _ = self.process_target(raw_conv, target, multimage_mode=multimage_mode)
#         conv = self.build_conv(raw_conv)
#         if return_conv:
#             # noinspection PyTypeChecker
#             return conv
#         text_dict = self.process_text(conv)
#         image_dict = self.process_image(image)
#
#         # return
#         ret_dict = {}
#         ret_dict.update(text_dict)
#         ret_dict.update(image_dict)
#         self._print_sample(ret_dict, raw_conv, conv)
#         if debug_mode:
#             return {'ret': ret_dict, 'raw_conv': raw_conv, 'conv': conv, 'image': image}
#         return ret_dict
#
#     def __len__(self):
#         raise NotImplementedError
#
#     # noinspection PyMethodMayBeStatic
#     def process_conv_multimage(self, raw_conv, image):
#         # re-sort multi image
#         if image is None:
#             return raw_conv, image
#         if not isinstance(image, (list, tuple)):
#             return raw_conv, image
#         image_seqs = []
#         for conv in raw_conv:
#             image_seqs.extend(conv['image_seq'] if 'image_seq' in conv else [])
#         images = []
#         for idx in image_seqs:
#             images.append(image[idx])
#         return raw_conv, images
#
#     def get_raw_item(self, index) -> Dict[str, Any]:
#         """
#         return item format like this.
#         item = {
#             'image': # PIL.Image.Image,
#             'target': {
#                 # xmin, ymin, xmax, ymax
#                 'boxes': [
#                     [10, 10, 256, 265],  # dog1
#                     [24, 18, 378, 768],  # dog2
#                     [100, 310, 670, 653],  # man
#                     [278, 320, 809, 673],  # rope
#                 ],
#             }
#
#             "conversations": [
#                 {
#                     'from': 'human',
#                     'value': 'What is the relation between the two dogs <boxes> and the man <boxes> in the image <image> ?',
#                     'boxes_seq': [[0, 1], [2], ],
#                 },
#                 {
#                     'from': 'gpt',
#                     'value': 'a rope <boxes> is connecting the left dog <boxes> with the man <boxes>. '
#                              'So the man <boxes> is walking the dog <boxes>.'
#                             'And the man <boxes> has no relationship with the right dog <boxes>',
#                     'boxes_seq': [[3], [0], [2], [2], [0], [2], [1]],
#                 }
#             ]
#         }
#         # placeholder: <image> <boxes>
#         """
#         raise NotImplementedError
#
#     # noinspection PyMethodMayBeStatic
#     def validate_raw_item(self, item):
#         has_image = 'image' in item and bool(item['image'])
#         has_target = 'target' in item and bool(item['target']) and any(bool(elem) for elem in item['target'].values())
#         has_target_boxes = 'boxes' in item['target'] if has_target else False
#         raw_conv: List[Dict[str, Any]] = item['conversations']
#
#         # check image
#         human_input_has_image_placeholder = any(
#             sentence['from'] == 'human' and IMAGE_PLACEHOLDER in sentence['value'] for sentence in raw_conv
#         )
#         if human_input_has_image_placeholder:
#             assert has_image
#         if has_image and (not human_input_has_image_placeholder):
#             warnings.warn(f'item has image but the question has no image placeholder.\n{item}')
#         gpt_input_has_image_placeholder = any(
#             sentence['from'] == 'gpt' and IMAGE_PLACEHOLDER in sentence['value'] for sentence in raw_conv
#         )
#         assert not gpt_input_has_image_placeholder
#
#         # check target
#         has_boxes_placeholder = any(
#             BOXES_PLACEHOLDER in sentence['value'] for sentence in raw_conv
#         )
#         if has_boxes_placeholder:
#             assert has_target_boxes
#         # not check box placeholder num this will be checked in format process
#
#     def build_conv(self, source: List[Dict[str, Any]]) -> Conversation:
#         conv = self.conv_template()
#         role_map = {"human": conv.roles[0], "gpt": conv.roles[1]}
#         assert len(source) > 0
#         assert source[0]['from'] == 'human'
#         for sentence in source:
#             role = role_map[sentence['from']]
#             conv.append_message(role, sentence['value'])
#         return conv
#
#     def process_conv(self, raw_conv: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """
#         some utils preprocess for raw_conv.
#             e.g. replace <image> placeholder to sequence <im_start> <im_patch>*256 <im_end>
#         """
#         return self.process_func['conv'](raw_conv, self.preprocessor, self.conv_template)
#
#     def process_target(self, raw_conv: List[Dict[str, Any]], target: Dict[str, Any], multimage_mode=False) -> Tuple[
#         List[Dict[str, Any]], Dict[str, Any]]:
#         """
#         convert target placeholder to actual information in raw_conv.
#             e.g. normalize bounding boxes; convert bounding boxes format; replace <boxes> placeholder
#         """
#         return self.process_func['target'](raw_conv, target, self.preprocessor, multimage_mode=multimage_mode)
#
#     def process_text(self, conv: Conversation) -> Dict[str, Any]:
#         """
#         convert Conversation object to torch.Tensor, e.g. input_ids, labels, attention_mask, etc.
#             self.tokenize_kwargs control something like padding/truncation behavior.
#         """
#         return self.process_func['text'](conv, self.preprocessor, self.mode, **self.tokenize_kwargs)
#
#     def process_image(self, image: Image.Image) -> Dict[str, Any]:
#         """
#         convert Image.Image object to torch.Tensor
#         """
#         return self.process_func['image'](image, self.preprocessor)
#
#     def _print_sample(self, ret_dict, raw_conv, conv):
#         if not hasattr(self, '_printed_sample'):
#             self._printed_sample = True
#             post_processed_labels = post_process_generate_ids(self.preprocessor['text'], ret_dict['labels'])
#             print(f"=================== {self.mode} sample ===================", flush=True)
#             print(f"        input_ids: {self.preprocessor['text'].convert_ids_to_tokens(ret_dict['input_ids'])}")
#             print(f"           labels: {self.preprocessor['text'].convert_ids_to_tokens(post_processed_labels)}")
#             print(f"decoded input_ids: {self.preprocessor['text'].decode(ret_dict['input_ids'])}")
#             print(f"decoded    labels: {self.preprocessor['text'].decode(post_processed_labels)}")
#             if 'image' in ret_dict and ret_dict['image'] is not None:
#                 image = ret_dict['image']
#                 if isinstance(image, torch.Tensor):
#                     print(f"            image: {image.shape}")
#                 elif isinstance(image, dict):
#                     print(f"            image: {image.keys()}")
#                 elif isinstance(image, list) and len(image) > 0:
#                     print(f"            image: {len(image)}, {type(image[0])}")
#                 else:
#                     print(f"            image: {type(image)}")
#             print("====================================================", flush=True)
#             try:
#                 if self.training_args is not None:
#                     _save_obj = {
#                         'ret_dict': ret_dict,
#                         'raw_conv': raw_conv,
#                         'conv': conv.get_prompt(),
#                     }
#                     from pathlib import Path
#                     output_dir = Path(self.training_args.output_dir)
#                     output_dir.mkdir(exist_ok=True, parents=True)
#                     _local_rank = self.training_args.local_rank
#                     _word_size = self.training_args.world_size
#                     _file_path = str(output_dir / f'sample_check_{self.mode}_{_local_rank}_{_word_size}.pt')
#                     print(f'saving some sample to {_file_path} for check.')
#                     torch.save(_save_obj, _file_path)
#             except Exception as e:
#                 warnings.warn(f'try to save samples but get exception: {e.args}. ignored.')
#
#
# class SingleImageInteractive(SingleImageConvDatasetMixin):
#     _printed_sample = True
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.image: Optional[Image.Image] = None
#         self.roles = ('human', 'gpt')
#         self.boxes = []
#         self.points = []
#         self.raw_conv = []
#         self.conversations = []
#
#     def set_image(self, image: Image.Image):
#         assert self.image is None, f"{image}"
#         self.image = image
#
#     def append_message(self, role: str, message: str, *, boxes=None, points=None, boxes_seq=None, points_seq=None):
#         """Append a new message."""
#         assert role in self.roles
#
#         def convert_idx(objs_seq, objs_value, get_obj_idx_func):
#             if objs_seq is None:
#                 return None
#             ret = []
#             for objs_idx in objs_seq:
#                 new_objs_idx = []
#                 for idx in objs_idx:
#                     new_idx = get_obj_idx_func(objs_value[idx])
#                     new_objs_idx.append(new_idx)
#                 ret.append(tuple(new_objs_idx))
#             return tuple(ret)
#
#         boxes_seq = convert_idx(boxes_seq, boxes, self._get_box_idx)
#         points_seq = convert_idx(points_seq, points, self._get_point_idx)
#
#         if self.image is not None:
#             previous_message_has_image_placeholder = any(
#                 '<image>' in item['value'] for item in self.conversations
#             )
#             if not previous_message_has_image_placeholder and '<image>' not in message:
#                 message = '<image> ' + message
#             if previous_message_has_image_placeholder and '<image>' in message:
#                 message = message.replace('<image>', '')
#
#         self.conversations.append(
#             {
#                 'from': role,
#                 'value': message,
#                 'boxes_seq': copy.deepcopy(boxes_seq),
#                 'points_seq': copy.deepcopy(points_seq),
#             }
#         )
#
#     def get_raw_item(self, index=None):
#         ret = copy.deepcopy({
#             'image': self.image,
#             'target': {
#                 'boxes': self.boxes,
#                 'points': self.points,
#             },
#             'conversations': self.conversations,
#         })
#         assert ret['conversations'][0]['from'] == self.roles[0]
#         if ret['conversations'][-1]['from'] == self.roles[0]:
#             ret['conversations'].append(
#                 {
#                     'from': self.roles[1],
#                     'value': '',
#                 }
#             )
#         return ret
#
#     def to_model_input(self):
#         item = self.__getitem__(0)
#         ret = {'input_ids': item['input_ids'].unsqueeze(0).cuda()}
#         if 'image' in item and item['image'] is not None:
#             ret['images'] = item['image'].unsqueeze(0).cuda()
#         else:
#             ret['images'] = None
#         return ret
#
#     def to_gradio_chatbot_new_messages(self):
#         conv = self.__getitem__(0, return_conv=True)
#         new_messages = conv.messages[-2:]
#         ret_messages = []
#         for r, m in new_messages:
#             nm = m.replace('<im_patch>', '').replace('<im_end>', '').replace('<im_start>', '<image>')
#             ret_messages.append((r, nm))
#         return ret_messages
#
#     def _get_box_idx(self, box):
#         assert isinstance(box, (tuple, list)), f"{type(box)}"
#         assert isinstance(box[0], (int, float)), f"{type(box[0])}"
#         assert len(box) == 4
#         box = tuple(box)
#         if box not in self.boxes:
#             self.boxes.append(box)
#             return len(self.boxes) - 1
#         else:
#             return self.boxes.index(box)
#
#     def _get_point_idx(self, point):
#         assert isinstance(point, (tuple, list))
#         assert isinstance(point[0], (int, float))
#         assert len(point) == 2
#         point = tuple(point)
#         if point not in self.points:
#             self.points.append(tuple(point))
#             return len(self.points) - 1
#         else:
#             return self.points.index(point)
#
#     def __len__(self):
#         return 1
#
#
# def post_process_generate_ids(tokenizer: PreTrainedTokenizer, ids: torch.Tensor):
#     ids = copy.deepcopy(ids)  # do not modify origin preds and targets
#     ids[ids < 0] = tokenizer.pad_token_id
#     return ids
#
#
# class SingleImageConvDatasetMixin:
#
#     def __init__(
#             self,
#             *args,
#             preprocessor: Dict[str, Any],
#             process_func: Dict[str, Any],
#             conv_template: Callable[[], Conversation] = partial(get_conv_template, name='vicuna_v1.1'),
#             mode='train',
#             tokenize_kwargs: dict = None,
#             training_args: TrainingArguments = None,
#             transforms: Optional[Callable] = None,
#             **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         assert mode in ['train', 'validation', 'test']
#
#         self.preprocessor = preprocessor
#         self.process_func = process_func
#         self.conv_template = conv_template
#         self.mode = mode
#         self.tokenize_kwargs = tokenize_kwargs if tokenize_kwargs is not None else {}
#         self.training_args = training_args
#         self.transforms = transforms
#
#     def __getitem__(self, index, debug_mode=False, return_conv=False) -> Dict[str, Any]:
#         # getitem
#         item = self.get_raw_item(index)
#         image: Image.Image = item.get('image', None)
#         target: Dict[str, Any] = item.get('target', None)
#         raw_conv: List[Dict[str, Any]] = item['conversations']
#
#         # transform
#         assert isinstance(image, list) == isinstance(target, list)
#         multimage_mode = isinstance(image, list)
#         if isinstance(image, list):
#             # TODO: validate raw item
#             transformed_image, transformed_target = [], []
#             for img, tgt in zip(image, target):
#                 if self.transforms is not None and image is not None:
#                     img, tgt = self.transforms(img, tgt)
#                 if tgt is not None:
#                     tgt['width'], tgt['height'] = img.width, img.height
#                 transformed_image.append(img)
#                 transformed_target.append(tgt)
#             image, target = transformed_image, transformed_target
#         else:
#             self.validate_raw_item(item)  # only validate for single image.
#             if self.transforms is not None and image is not None:
#                 image, target = self.transforms(image, target)
#             has_image = 'image' in item and bool(item['image'])
#             has_target = 'target' in item and bool(item['target']) and any(bool(elem) for elem in item['target'].values())
#             if has_target and has_image:
#                 target['width'], target['height'] = image.width, image.height
#
#         # preprocess
#         raw_conv = self.process_conv(raw_conv)
#         raw_conv, image = self.process_conv_multimage(raw_conv, image)
#         raw_conv, _ = self.process_target(raw_conv, target, multimage_mode=multimage_mode)
#         conv = self.build_conv(raw_conv)
#         if return_conv:
#             # noinspection PyTypeChecker
#             return conv
#         text_dict = self.process_text(conv)
#         image_dict = self.process_image(image)
#
#         # return
#         ret_dict = {}
#         ret_dict.update(text_dict)
#         ret_dict.update(image_dict)
#         self._print_sample(ret_dict, raw_conv, conv)
#         if debug_mode:
#             return {'ret': ret_dict, 'raw_conv': raw_conv, 'conv': conv, 'image': image}
#         return ret_dict
#
#     def __len__(self):
#         raise NotImplementedError
#
#     # noinspection PyMethodMayBeStatic
#     def process_conv_multimage(self, raw_conv, image):
#         # re-sort multi image
#         if image is None:
#             return raw_conv, image
#         if not isinstance(image, (list, tuple)):
#             return raw_conv, image
#         image_seqs = []
#         for conv in raw_conv:
#             image_seqs.extend(conv['image_seq'] if 'image_seq' in conv else [])
#         images = []
#         for idx in image_seqs:
#             images.append(image[idx])
#         return raw_conv, images
#
#     def get_raw_item(self, index) -> Dict[str, Any]:
#         """
#         return item format like this.
#         item = {
#             'image': # PIL.Image.Image,
#             'target': {
#                 # xmin, ymin, xmax, ymax
#                 'boxes': [
#                     [10, 10, 256, 265],  # dog1
#                     [24, 18, 378, 768],  # dog2
#                     [100, 310, 670, 653],  # man
#                     [278, 320, 809, 673],  # rope
#                 ],
#             }
#
#             "conversations": [
#                 {
#                     'from': 'human',
#                     'value': 'What is the relation between the two dogs <boxes> and the man <boxes> in the image <image> ?',
#                     'boxes_seq': [[0, 1], [2], ],
#                 },
#                 {
#                     'from': 'gpt',
#                     'value': 'a rope <boxes> is connecting the left dog <boxes> with the man <boxes>. '
#                              'So the man <boxes> is walking the dog <boxes>.'
#                             'And the man <boxes> has no relationship with the right dog <boxes>',
#                     'boxes_seq': [[3], [0], [2], [2], [0], [2], [1]],
#                 }
#             ]
#         }
#         # placeholder: <image> <boxes>
#         """
#         raise NotImplementedError
#
#     # noinspection PyMethodMayBeStatic
#     def validate_raw_item(self, item):
#         has_image = 'image' in item and bool(item['image'])
#         has_target = 'target' in item and bool(item['target']) and any(bool(elem) for elem in item['target'].values())
#         has_target_boxes = 'boxes' in item['target'] if has_target else False
#         raw_conv: List[Dict[str, Any]] = item['conversations']
#
#         # check image
#         human_input_has_image_placeholder = any(
#             sentence['from'] == 'human' and IMAGE_PLACEHOLDER in sentence['value'] for sentence in raw_conv
#         )
#         if human_input_has_image_placeholder:
#             assert has_image
#         if has_image and (not human_input_has_image_placeholder):
#             warnings.warn(f'item has image but the question has no image placeholder.\n{item}')
#         gpt_input_has_image_placeholder = any(
#             sentence['from'] == 'gpt' and IMAGE_PLACEHOLDER in sentence['value'] for sentence in raw_conv
#         )
#         assert not gpt_input_has_image_placeholder
#
#         # check target
#         has_boxes_placeholder = any(
#             BOXES_PLACEHOLDER in sentence['value'] for sentence in raw_conv
#         )
#         if has_boxes_placeholder:
#             assert has_target_boxes
#         # not check box placeholder num this will be checked in format process
#
#     def build_conv(self, source: List[Dict[str, Any]]) -> Conversation:
#         conv = self.conv_template()
#         role_map = {"human": conv.roles[0], "gpt": conv.roles[1]}
#         assert len(source) > 0
#         assert source[0]['from'] == 'human'
#         for sentence in source:
#             role = role_map[sentence['from']]
#             conv.append_message(role, sentence['value'])
#         return conv
#
#     def process_conv(self, raw_conv: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """
#         some utils preprocess for raw_conv.
#             e.g. replace <image> placeholder to sequence <im_start> <im_patch>*256 <im_end>
#         """
#         return self.process_func['conv'](raw_conv, self.preprocessor, self.conv_template)
#
#     def process_target(self, raw_conv: List[Dict[str, Any]], target: Dict[str, Any], multimage_mode=False) -> Tuple[
#         List[Dict[str, Any]], Dict[str, Any]]:
#         """
#         convert target placeholder to actual information in raw_conv.
#             e.g. normalize bounding boxes; convert bounding boxes format; replace <boxes> placeholder
#         """
#         return self.process_func['target'](raw_conv, target, self.preprocessor, multimage_mode=multimage_mode)
#
#     def process_text(self, conv: Conversation) -> Dict[str, Any]:
#         """
#         convert Conversation object to torch.Tensor, e.g. input_ids, labels, attention_mask, etc.
#             self.tokenize_kwargs control something like padding/truncation behavior.
#         """
#         return self.process_func['text'](conv, self.preprocessor, self.mode, **self.tokenize_kwargs)
#
#     def process_image(self, image: Image.Image) -> Dict[str, Any]:
#         """
#         convert Image.Image object to torch.Tensor
#         """
#         return self.process_func['image'](image, self.preprocessor)
#
#     def _print_sample(self, ret_dict, raw_conv, conv):
#         if not hasattr(self, '_printed_sample'):
#             self._printed_sample = True
#             post_processed_labels = post_process_generate_ids(self.preprocessor['text'], ret_dict['labels'])
#             print(f"=================== {self.mode} sample ===================", flush=True)
#             print(f"        input_ids: {self.preprocessor['text'].convert_ids_to_tokens(ret_dict['input_ids'])}")
#             print(f"           labels: {self.preprocessor['text'].convert_ids_to_tokens(post_processed_labels)}")
#             print(f"decoded input_ids: {self.preprocessor['text'].decode(ret_dict['input_ids'])}")
#             print(f"decoded    labels: {self.preprocessor['text'].decode(post_processed_labels)}")
#             if 'image' in ret_dict and ret_dict['image'] is not None:
#                 image = ret_dict['image']
#                 if isinstance(image, torch.Tensor):
#                     print(f"            image: {image.shape}")
#                 elif isinstance(image, dict):
#                     print(f"            image: {image.keys()}")
#                 elif isinstance(image, list) and len(image) > 0:
#                     print(f"            image: {len(image)}, {type(image[0])}")
#                 else:
#                     print(f"            image: {type(image)}")
#             print("====================================================", flush=True)
#             try:
#                 if self.training_args is not None:
#                     _save_obj = {
#                         'ret_dict': ret_dict,
#                         'raw_conv': raw_conv,
#                         'conv': conv.get_prompt(),
#                     }
#                     from pathlib import Path
#                     output_dir = Path(self.training_args.output_dir)
#                     output_dir.mkdir(exist_ok=True, parents=True)
#                     _local_rank = self.training_args.local_rank
#                     _word_size = self.training_args.world_size
#                     _file_path = str(output_dir / f'sample_check_{self.mode}_{_local_rank}_{_word_size}.pt')
#                     print(f'saving some sample to {_file_path} for check.')
#                     torch.save(_save_obj, _file_path)
#             except Exception as e:
#                 warnings.warn(f'try to save samples but get exception: {e.args}. ignored.')


import PIL.Image
import numpy as np
def draw_bounding_boxes(
        image: Union[torch.Tensor, PIL.Image.Image],
        boxes: Union[torch.Tensor, List, np.ndarray],
        **kwargs,
):
    if isinstance(image, PIL.Image.Image):
        from torchvision.transforms import PILToTensor
        image = PILToTensor()(image)
    assert isinstance(image, torch.Tensor), ""

    if not isinstance(boxes, torch.Tensor):
        boxes = torch.as_tensor(boxes)
    assert isinstance(boxes, torch.Tensor)

    from torchvision.utils import draw_bounding_boxes as _draw_bounding_boxes
    return _draw_bounding_boxes(image, boxes, **kwargs)


def denorm_bboxes(w, h, box):
    xmin, ymin, xmax, ymax = box
    # out_box = [ymin*w, xmin*h, ymax*w, xmax*h]
    out_box = [xmin * w, ymin * h, xmax * w, ymax * h]

    return out_box