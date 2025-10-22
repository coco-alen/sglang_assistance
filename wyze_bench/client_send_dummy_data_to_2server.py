import asyncio
import base64
import os
import argparse
from openai import AsyncOpenAI
import numpy as np
import cv2
from tqdm import tqdm
import time
import random

def generate_dummy_image(size=448, image_index=0):
    image = np.random.randint(0, 256, size=(size, size, 3), dtype=np.uint8)
    
    text = f"Image #{image_index}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (size - text_size[0]) // 2
    text_y = (size + text_size[1]) // 2
    
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    b64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
    }

def generate_image_batch(batch_id, num_images):
    images = []
    for i in range(num_images):
        image_index = batch_id * 1000 + i
        images.append(generate_dummy_image(448, image_index))
    return images

async def process_single_batch(client, batch_id, images, model_name, semaphore, base_url):
    async with semaphore:
        user_prompt = """
        **Task:** Analyze the provided images and identify key visual elements, patterns, and any notable features.
        Provide a comprehensive description of what you observe across all images.

        **Key Instructions:**
        1. **Summary:** 
           - Provide a brief title summarizing the main theme or pattern observed (4 words or fewer).
           - Add an appropriate emoji at the beginning of the title.
        
        2. **Image Analysis:**
           - Describe the visual elements present in the images (shapes, colors, text).
           - Note any patterns or commonalities across the images.
           - Mention specific details that stand out.
           - Keep the description concise but informative (under 200 characters).
        
        3. **Quality Score:**
           - Rate the overall visual quality and coherence of the image set from 1-10.
           - 1-3: Poor quality or incoherent
           - 4-6: Average quality with some interesting elements
           - 7-9: High quality with clear patterns or themes
           - 10: Exceptional quality and coherence
        
        **Output Format:**
        Summary: [emoji] [4 words or fewer title]
        Analysis: [Concise description under 200 characters]
        Quality: [1-10 score]
        """
        
        messages = [
            {"role": "system", "content": "You are an image analysis assistant specialized in identifying visual patterns and elements."},
            {"role": "user", "content": images + [{"type": "text", "text": user_prompt}]},
        ]
        
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=100,
                temperature=0.0,
                timeout=120,
            )
            
            analysis = response.choices[0].message.content.strip()
            return {
                'batch_id': batch_id,
                'num_images': len(images),
                'analysis': analysis,
                'base_url': base_url,
                'success': True
            }
        except Exception as e:
            return {
                'batch_id': batch_id,
                'num_images': len(images),
                'analysis': f"Error: {str(e)}",
                'base_url': base_url,
                'success': False
            }

async def main():
    parser = argparse.ArgumentParser(description='Send dummy images to API with configurable concurrency')
    parser.add_argument('--min-images-per', type=int, required=True, help='Minimum number of images per request')
    parser.add_argument('--max-images-per', type=int, required=True, help='Maximum number of images per request')
    parser.add_argument('--max-concurrency', type=int, required=True, help='Maximum number of concurrent API calls')
    parser.add_argument('--base-url', type=str, required=True, help='Base URL for the first API endpoint')
    parser.add_argument('--base-url-2', type=str, required=True, help='Base URL for the second API endpoint')
    parser.add_argument('--num-requests', type=int, required=True, help='Total number of requests to send')
    parser.add_argument('--model', type=str, default='OpenGVLab/InternVL3-8B', help='Model name to use')
    parser.add_argument('--api-key', type=str, default='EMPTY', help='API key (default: EMPTY)')
    args = parser.parse_args()
    
    if args.min_images_per > args.max_images_per:
        print("Error: min-images-per must be less than or equal to max-images-per")
        return
    
    api_key = os.getenv("OPENAI_API_KEY", args.api_key)
    client1 = AsyncOpenAI(api_key=api_key, base_url=args.base_url)
    client2 = AsyncOpenAI(api_key=api_key, base_url=args.base_url_2)
    semaphore = asyncio.Semaphore(args.max_concurrency)
    
    print(f"Configuration:")
    print(f"  Min images per request: {args.min_images_per}")
    print(f"  Max images per request: {args.max_images_per}")
    print(f"  Max concurrency: {args.max_concurrency}")
    print(f"  Base URL 1: {args.base_url}")
    print(f"  Base URL 2: {args.base_url_2}")
    print(f"  Number of requests: {args.num_requests}")
    print(f"  Model: {args.model}")
    print()
    
    print("Phase 1: Generating image batches...")
    batches = []
    total_images = 0
    
    for batch_id in tqdm(range(args.num_requests), desc="Creating batches"):
        num_images = random.randint(args.min_images_per, args.max_images_per)
        images = generate_image_batch(batch_id, num_images)
        use_first_url = random.choice([True, False])
        batches.append({
            'batch_id': batch_id,
            'images': images,
            'num_images': num_images,
            'client': client1 if use_first_url else client2,
            'base_url': args.base_url if use_first_url else args.base_url_2
        })
        total_images += num_images
    
    url1_count = sum(1 for b in batches if b['base_url'] == args.base_url)
    url2_count = len(batches) - url1_count
    
    print(f"\nGenerated {args.num_requests} batches with {total_images} total images")
    print(f"Average images per batch: {total_images / args.num_requests:.1f}")
    print(f"Distribution: {url1_count} requests to URL1, {url2_count} requests to URL2")
    
    print("\nPhase 2: Sending API requests...")
    
    tasks = [
        process_single_batch(batch['client'], batch['batch_id'], batch['images'], 
                           args.model, semaphore, batch['base_url'])
        for batch in batches
    ]
    
    start_time = time.time()
    results = []
    
    with tqdm(total=len(tasks), desc="Processing batches") as pbar:
        for coro in asyncio.as_completed(tasks):
            result_data = await coro
            results.append(result_data)
            pbar.update(1)
    
    end_time = time.time()
    duration = end_time - start_time
    
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    throughput = len(results) / duration if duration > 0 else 0
    
    url1_success = sum(1 for r in successful_results if r['base_url'] == args.base_url)
    url2_success = sum(1 for r in successful_results if r['base_url'] == args.base_url_2)
    url1_failed = sum(1 for r in failed_results if r['base_url'] == args.base_url)
    url2_failed = sum(1 for r in failed_results if r['base_url'] == args.base_url_2)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total requests sent: {len(results)}")
    print(f"Successful: {len(successful_results)} (URL1: {url1_success}, URL2: {url2_success})")
    print(f"Failed: {len(failed_results)} (URL1: {url1_failed}, URL2: {url2_failed})")
    print(f"Total time: {duration:.2f} seconds")
    print(f"Throughput: {throughput:.2f} requests/second")
    print(f"Total images processed: {total_images}")
    print(f"Images/second: {total_images/duration:.2f}")
    print(f"{'='*60}")
    
    print("\nExample outputs:")
    for i, result in enumerate(successful_results[:2]):
        print(f"\nBatch ID: {result['batch_id']}")
        print(f"Target URL: {result['base_url']}")
        print(f"Number of images: {result['num_images']}")
        print(f"Analysis: {result['analysis']}")
        print("-" * 60)

if __name__ == "__main__":
    asyncio.run(main())