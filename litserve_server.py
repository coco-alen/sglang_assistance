# import asyncio
# import base64
# import io
# import os
# from PIL import Image
# import torch

# os.environ["TORCHINDUCTOR_CACHE_DIR"] = "~/.triton"
# os.environ["CUDA_VISIBLE_DEVICES"] = '7'

# import litserve as ls
# from sglang.srt.parser.conversation import chat_templates
# from sglang import Engine

# from transformers import AutoProcessor
# from transformers import Qwen2_5_VLForConditionalGeneration

# class Qwen2_5_VL_API(ls.LitAPI):
#     def setup(self, device):
#         model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
#         chat_template = "qwen2-vl"

#         self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
#         self.vision = (
#         Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path).eval().visual.to(device)
#         )
#         self.llm = Engine(
#             model_path=model_path, chat_template=chat_template, mem_fraction_static=0.5,
#             disable_radix_cache=True
#         )
#         self.conv = chat_templates[chat_template].copy()

#     def decode_b64_image(self, b64_string):
#         image_data = base64.b64decode(b64_string)
#         image = Image.open(io.BytesIO(image_data)).convert("RGB")
#         return image

#     def get_image_features(self, images_tensor, image_grid_thw):
#         with torch.no_grad():
#             image_features = self.vision(images_tensor, image_grid_thw)
#         return image_features

#     async def decode_request(self, request):
#         text_input = getattr(request, "text", "")
#         image_data = getattr(request, "image", None)

#         if image_data is not None:
#             image = await asyncio.to_thread(self.decode_b64_image, image_data)
            
#             self.conv.append_message(self.conv.roles[0], f"{text_input}{self.conv.image_token}")
#             self.conv.image_data = [image]
#             processed_prompt = self.processor(
#                 images=[image], text=self.conv.get_prompt(), return_tensors="pt"
#             )
#             precomputed_embeddings = await asyncio.to_thread(
#                 self.get_image_features,
#                 processed_prompt["pixel_values"].to(self.vision.device),
#                 processed_prompt["image_grid_thw"].to(self.vision.device)
#                 )
            
#             input_ids = processed_prompt["input_ids"][0].detach().cpu().tolist()
#             mm_item = dict(
#                 modality="IMAGE",
#                 image_grid_thw=processed_prompt["image_grid_thw"],
#                 precomputed_embeddings=precomputed_embeddings,
#             )
#             return input_ids, mm_item
#         else:
#             self.conv.append_message(self.conv.roles[0], text_input)
#             processed_prompt = self.processor(
#                 text=self.conv.get_prompt(), return_tensors="pt"
#             )
#             input_ids = processed_prompt["input_ids"][0].detach().cpu().tolist()
#             mm_item = None
#             return input_ids, mm_item

#     async def predict(self, inputs, context):
#         input_ids, mm_item = inputs
#         if mm_item is not None:
#             out = await self.llm.async_generate(input_ids=input_ids, image_data=[mm_item])
#         else:
#             out = await self.llm.async_generate(input_ids=input_ids)
#         return out["text"]

#     async def encode_response(self, outputs):
#         return {"text": outputs}

# if __name__ == "__main__":
#     api = Qwen2_5_VL_API(enable_async=True, api_path="/v1")
#     server = ls.LitServer(api, accelerator="cuda")
#     server.run(port=23333)




import asyncio
import base64
import io
import os
import aiohttp
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "~/.triton"
os.environ["CUDA_VISIBLE_DEVICES"] = '7'

import litserve as ls
from sglang.srt.parser.conversation import chat_templates
from sglang import Engine

from transformers import AutoTokenizer, AutoModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class InternVL3_API(ls.LitAPI):
    def setup(self, device):
        model_path = "OpenGVLab/InternVL3-8B"
        chat_template = "internvl-2-5"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        vlm_model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True).eval()
        
        self.vision = vlm_model.vision_model.cuda()
        self.mlp = vlm_model.mlp1.cuda()
        self.conv = chat_templates[chat_template].copy()
        self.transform = self.build_transform(input_size=448)
        self.llm_url = "http://0.0.0.0:23333/generate"

    def build_transform(self, input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images


    def decode_b64_image(self, b64_string):
        image_data = base64.b64decode(b64_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image


    async def load_image(self, b64_string, input_size=448, max_num=12):
        image = await asyncio.to_thread(self.decode_b64_image, b64_string)
        images = await asyncio.to_thread(self.dynamic_preprocess, image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def get_image_features(self, images_tensor):
        with torch.no_grad():
            image_features = self.vision(images_tensor).last_hidden_state
            image_features = self.mlp(image_features)
        return image_features

    async def call_api(self, data):
        async with aiohttp.ClientSession() as session:
            async with session.post(self.llm_url, json=data) as response:
                # 获取返回内容
                result = await response.json()  # 或 await response.text()
                
                return result


    async def decode_request(self, request):
        text_input = getattr(request, "text", "")
        image_data = getattr(request, "image", None)

        if image_data is not None:
            self.conv.append_message(self.conv.roles[0], f"{text_input}{self.conv.image_token}")
            processed_prompt = self.tokenizer(
                    self.conv.get_prompt(), return_tensors="pt"
                ).to(torch.bfloat16)
            input_ids = processed_prompt["input_ids"][0].detach().cpu().tolist()

            image = await self.load_image(image_data)
            return {
                "input_ids": input_ids,
                "image" : image.to(torch.bfloat16).cuda()
            }
        
        else:
            self.conv.append_message(self.conv.roles[0], f"{text_input}{self.conv.image_token}")
            processed_prompt = self.tokenizer(
                    self.conv.get_prompt(), return_tensors="pt"
                ).to(torch.bfloat16)
            input_ids = processed_prompt["input_ids"][0].detach().cpu().tolist()
            return {
                "input_ids": input_ids,
                "image" : None,
            }

    async def predict(self, inputs):
        input_ids, image_tensor = inputs["input_ids"], inputs["image"]
        if image_tensor is not None:
            image_features = await asyncio.to_thread(self.get_image_features, image_tensor)

        data = {
            "input_ids": input_ids,
        }

        result = await self.call_api(data)
        return result
        

    async def encode_response(self, outputs):
        return outputs

if __name__ == "__main__":
    api = InternVL3_API(enable_async=True, api_path="/v1")
    server = ls.LitServer(api, accelerator="cuda")
    server.run(port=23334, num_api_servers=8)