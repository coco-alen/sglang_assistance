import os
import cv2
import math
import requests
import base64
from PIL import Image
from io import BytesIO

# 配置参数
def extract_frames(video_path, min_frames=10, max_frames=30, out_dir='frames'):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    # 计算初始采样fps
    sample_fps = 1
    num_frames = int(duration * sample_fps)
    # 降低fps直到帧数在范围内
    while num_frames > max_frames:
        sample_fps /= 2
        num_frames = int(duration * sample_fps)
        if sample_fps < 0.1:
            break
    # 保证帧数不少于min_frames
    if num_frames < min_frames:
        sample_fps = min_frames / duration
        num_frames = min_frames
    interval = int(fps / sample_fps)
    saved = 0
    frame_idx = 0
    images = []
    while cap.isOpened() and saved < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            img_path = os.path.join(out_dir, f'frame_{saved:03d}.jpg')
            cv2.imwrite(img_path, frame)
            images.append(img_path)
            saved += 1
        frame_idx += 1
    cap.release()
    return images

def build_prompt(image_paths):
    # prompt = "Describe the following images: "
    # prompt += ", ".join([f"<image:{os.path.basename(p)}>" for p in image_paths])
    prompt = """
        **Task:** Analyze the video footage and provide information that user should know or will be interested in,
        give descriptions of activities to determine if any anomaly is present without referring to human gender.
        Describe objects and actions based on visible physical movements or interactions, ignoring changes in lighting, and highlight key events.
        Only describe the fact, don't make predictions. (e.g. If the cars are not moving, say 'Parked' instead of 'Arrived')
        Additionally, assign an importance score to the event.
        Ensure your output strictly adheres to the provided format. 

        **Key Instructions:**
        1. **Title:** 
           - The title should highlight the factual, concise key insights in 4 words or fewer (32 characters or fewer)(e.g. Fedex delivery).
           - **Must add only one attracting emoji to beginning of title that represents the event. 
               If the scenario is very dangerous or very urgent, use police car light emoji "🚨" instead.**
        
        2. **Video Description:**
           - **A factual, concise and accurate description in human-like natural language in 200 characters or fewer in English only**.
           - Describe objects in the scene and their state. Only describe object movements if they are clearly visible, 
               with the object's position or orientation changing over time. 
               Do not infer movement from lighting changes, shadows, or other unrelated effects.
           - Avoid opinions or subjective judgments, like 'messy kitchen','dirty box'.
           - Focus on details of objects (e.g. color and make of the car, package delivery business),
           - If the labels contain a name, a recognized face in the scene, you must add it. 
           - Use approximate terms like "a few" instead of specific numbers (e.g., "a few rabbits").
           - Use the timestamp footprint to help your understanding, but don't say the timestamp.
        
        3. **Importance Score:**
           - Provide an importance score from 1-10 based on the significance of the event.
           - 1-3: Routine activities
           - 4-6: Notable events worth attention
           - 7-9: Important events requiring action
           - 10: Emergency situations
        
        **Output Format:**
        Title: [emoji] [4 words or fewer title]
        Description: [32 words or fewer description]
        Importance: [1-10 score]
        """
    return prompt

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode('utf-8')
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
    }

def send_to_sglang(image_paths, prompt, api_url, api_key=None):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    # 构造 content 列表：所有图片 + prompt
    content = [encode_image_to_base64(p) for p in image_paths]
    content.append({"type": "text", "text": prompt})
    data = {
        "messages": [
            {"role": "user", "content": content}
        ]
    }
    response = requests.post(api_url, json=data, headers=headers)
    response.raise_for_status()
    return response.json()

def main():
    video_path = "example.mp4"
    api_url = "http://0.0.0.0:23333/v1/chat/completions"
    api_key = None
    print("正在抽取视频帧...")
    image_paths = extract_frames(video_path)
    print(f"抽取了{len(image_paths)}帧。")
    prompt = build_prompt(image_paths)
    # print("生成的prompt:", prompt)
    print("正在请求sglang...")
    result = send_to_sglang(image_paths, prompt, api_url, api_key)
    print("sglang返回结果:")
    print(result["choices"][0]["message"]["content"])

if __name__ == "__main__":
    main()
