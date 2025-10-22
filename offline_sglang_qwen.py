import nest_asyncio
import os

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "~/.triton"
nest_asyncio.apply()  # Run this first.

from io import BytesIO
import requests
from PIL import Image
import time
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from sglang.srt.parser.conversation import chat_templates
from sglang import Engine

from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration

if __name__ == '__main__':
    model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
    chat_template = "qwen2-vl"

    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
    vision = (
        Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path).eval().visual.cuda()
    )
    image = Image.open(
        BytesIO(
            requests.get(
                "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
            ).content
        )
    )

    conv = chat_templates[chat_template].copy()
    conv.append_message(conv.roles[0], f"What is in this image?{conv.image_token}")
    conv.image_data = [image]

    print(conv.get_prompt())

    llm = Engine(
        model_path=model_path, chat_template=chat_template, mem_fraction_static=0.5,
        disable_radix_cache=True
    )
    processed_prompt = processor(
        images=[image], text=conv.get_prompt(), return_tensors="pt"
    )
    input_ids = processed_prompt["input_ids"][0].detach().cpu().tolist()
    precomputed_embeddings = vision(
        processed_prompt["pixel_values"].cuda(), processed_prompt["image_grid_thw"].cuda()
    )

    mm_item = dict(
        modality="IMAGE",
        image_grid_thw=processed_prompt["image_grid_thw"],
        precomputed_embeddings=precomputed_embeddings,
    )
    # out = llm.generate(prompt=conv.get_prompt(), image_data=[image])
    # print("==== Output Normal ====")
    # print(out["text"])

    out = llm.generate(input_ids=input_ids, image_data=[mm_item])
    print("==== Output EP Disagg ====")
    print(out["text"])


    import concurrent.futures
    def calculate_llm():
        return llm.generate(input_ids=input_ids, image_data=[mm_item])
    def cal_vlm():
        precomputed_embeddings = vision(
            processed_prompt["pixel_values"].cuda(), processed_prompt["image_grid_thw"].cuda()
        )

        mm_item = dict(
            modality="IMAGE",
            image_grid_thw=processed_prompt["image_grid_thw"],
            precomputed_embeddings=precomputed_embeddings,
        )
        # out = llm.generate(prompt=conv.get_prompt(), image_data=[image])
        # print("==== Output Normal ====")
        # print(out["text"])

        return llm.generate(input_ids=input_ids, image_data=[mm_item])
    
    Total_calls = 200
    max_workers = 32

    # Start the timer
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(calculate_llm): i for i in range(Total_calls)}
        for future in concurrent.futures.as_completed(future_to_item):
            idx = future_to_item[future]
            try:
                future.result()
            except Exception as e:
                print(f"Request {idx} generated an exception: {e}")
    end_time = time.time()
    print(f"Total time taken for LLM: {end_time - start_time:.2f} seconds")
    request_throughput = Total_calls / (end_time - start_time)
    print(f"Request throughput for LLM: {request_throughput:.2f}requests/second")   

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(cal_vlm): i for i in range(Total_calls)}
        for future in concurrent.futures.as_completed(future_to_item):
            idx = future_to_item[future]
            try:
                future.result()
            except Exception as e:
                print(f"Request {idx} generated an exception: {e}")
    end_time = time.time()
    print(f"Total time taken for VLM: {end_time - start_time:.2f} seconds")
    request_throughput = Total_calls / (end_time - start_time)
    print(f"Request throughput for VLM: {request_throughput:.2f}requests/second")