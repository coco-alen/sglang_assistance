import torch
import requests
from openai import OpenAI

from transformers import AutoTokenizer, AutoModel
from transformers.models.qwen3 import Qwen3Model

import time
import concurrent.futures
from tqdm import tqdm

def call_openai_api():

    client = OpenAI(
        base_url=openai_url
    )
    response = client.chat.completions.create(
        model="OpenGVLab/InternVL3-8B",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                    },
                ],
            }
        ],
        max_tokens=300,
    )
    # print(response.choices[0].message.content)
    return response

def call_api():

    data = {
        # "text": text,
        # "input_ids": input_ids,
        "input_embeds": inputs_embeds,
    }
    response = requests.post(url, json=data)
    print(response.json())


def test_throughput():
    Total_calls = 200
    max_workers = 32

    # Start the timer
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(call_api): i for i in range(Total_calls)}
        for future in tqdm(concurrent.futures.as_completed(future_to_item), total=Total_calls):
            idx = future_to_item[future]
            try:
                future.result()  
            except Exception as exc:
                print(f'Error processing sample {idx}: {exc}')


    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    request_throughput = Total_calls / (end_time - start_time)
    print(f"Request throughput: {request_throughput:.2f} requests/second")

if __name__ == "__main__":
    model_name = "Qwen/Qwen3-1.7B-FP8"
    url = "http://0.0.0.0:23333/generate"
    openai_url = "http://0.0.0.0:23333/v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen3Model.from_pretrained(model_name, trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
    embedding = model.embed_tokens
    texts = [
        """
Resolution theorem proving for showing that a formula of propositional logic is not satisfiable has which of the following properties?
I. It is a sound proof system in the sense that there does not exist a proof of the unsatisfiability of a satisfiable formula of propositional logic.
II. It is a complete proof system in the sense that there is a proof of unsatisfiability for every unsa tisfiable formula of propositional logic.
III. It is a succinct proof system in the sense that whenever an unsatisfiable formula F of propositional logic has a resolution proof, F also has a proof whose length is polynomial in the length of F.
Here you can choose from: [ "I only", "III only", "I and II only", "I and III only" ].
             """
    ]
    text = texts[0]
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(model.device)
    print("input tokens:", len(input_ids[0]))

    inputs_embeds = embedding(input_ids)
    input_ids = input_ids.tolist()
    inputs_embeds = inputs_embeds.tolist()

    # call_api()

    # call_openai_api()

    test_throughput()