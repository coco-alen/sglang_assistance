import asyncio
import base64
import os
import argparse
import csv
from pathlib import Path
from decord import VideoReader, cpu
from openai import AsyncOpenAI
import numpy as np
import cv2
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def extract_frames_from_video_sync(video_path, target_size=448):
    """Extract frames from a single video at 1 FPS (synchronous version for multiprocessing)."""
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        
        # Calculate frame indices for 1 FPS extraction
        # If video FPS is 30, we take every 30th frame
        frame_interval = int(round(fps))
        frame_indices = list(range(0, total_frames, frame_interval))
        
        frames = []
        
        for idx in frame_indices:
            # Get frame and convert to numpy array
            frame = vr[idx].asnumpy()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Downsample frame to target_size x target_size
            frame_resized = cv2.resize(frame_bgr, (target_size, target_size), interpolation=cv2.INTER_AREA)
            
            _, buffer = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            b64 = base64.b64encode(buffer).decode('utf-8')
            frames.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        
        return {
            'video_path': str(video_path),
            'frames': frames,
            'total_frames': total_frames,
            'fps': fps,
            'extracted_frames': len(frames),
            'success': True
        }
    except Exception as e:
        return {
            'video_path': str(video_path),
            'error': str(e),
            'success': False
        }

async def extract_frames_from_video(video_path, target_size=448):
    """Extract frames from a single video at 1 FPS."""
    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        
        # Calculate frame indices for 1 FPS extraction
        # If video FPS is 30, we take every 30th frame
        frame_interval = int(round(fps))
        frame_indices = list(range(0, total_frames, frame_interval))
        
        frames = []
        
        for idx in frame_indices:
            # Get frame and convert to numpy array
            frame = vr[idx].asnumpy()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            b64 = base64.b64encode(buffer).decode('utf-8')
            frames.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        
        return {
            'video_path': video_path,
            'frames': frames,
            'total_frames': total_frames,
            'fps': fps,
            'extracted_frames': len(frames)
        }
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

async def process_single_video(client, video_data, model_name, semaphore):
    """Process a single video with API call."""
    async with semaphore:
        user_prompt = """
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
        
        messages = [
            {"role": "system", "content": "You are a video analysis assistant specialized in detecting anomalies and important events."},
            {"role": "user", "content": video_data['frames'] + [{"type": "text", "text": user_prompt}]},
        ]
        
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=64,
                temperature=0.0,
                timeout=120,
            )
            
            caption = response.choices[0].message.content.strip()
            return {
                'video_path': str(video_data['video_path']),
                'caption': caption,
                'success': True
            }
        except Exception as e:
            return {
                'video_path': str(video_data['video_path']),
                'caption': f"Error: {str(e)}",
                'success': False
            }

async def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Process videos with concurrent API calls')
    parser.add_argument('category', type=str, help='Category folder name (e.g., "all" or "baby")')
    parser.add_argument('concurrency', type=int, help='Number of concurrent API calls')
    parser.add_argument('--decode-workers', type=int, default=16, help='Number of parallel workers for video decoding (default: 16)')
    parser.add_argument('--port', type=int, default=30000, help='port of the server')
    parser.add_argument('--data_path', type=str, default="/mnt/raid0/datasets/wyze_benchmark/videos", help='root directory of the dataset')
    parser.add_argument("--infinite_loop", action="store_true")
    args = parser.parse_args()
    
    # Configuration
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    base_url = f"http://localhost:{args.port}/v1"
    model_name = "OpenGVLab/InternVL3-8B"
    base_path = Path(args.data_path)
    category_path = base_path / args.category
    
    # Check if category folder exists
    if not category_path.exists():
        print(f"Error: Category folder '{category_path}' does not exist!")
        return
    
    # Get all video files
    video_files = list(category_path.glob("*.mp4"))
    if not video_files:
        print(f"No .mp4 files found in {category_path}")
        return
    
    print(f"Found {len(video_files)} videos in category '{args.category}'")
    print(f"Using concurrency level: {args.concurrency}")
    print(f"Using decode workers: {args.decode_workers}")
    
    # Phase 1: Preprocess all videos
    print("\nPhase 1: Preprocessing videos (extracting frames at 1 FPS)...")
    print(f"Using {args.decode_workers} parallel workers for video decoding")
    
    preprocessed_videos = []
    failed_videos = []
    
    # Use ProcessPoolExecutor for parallel video decoding
    with ProcessPoolExecutor(max_workers=args.decode_workers) as executor:
        # Submit all video processing tasks
        future_to_video = {
            executor.submit(extract_frames_from_video_sync, video_file): video_file 
            for video_file in video_files
        }
        
        # Process results with progress bar
        for future in tqdm(future_to_video, desc="Extracting frames", total=len(video_files)):
            video_file = future_to_video[future]
            try:
                result = future.result()
                if result['success']:
                    preprocessed_videos.append(result)
                else:
                    print(f"\nError processing {video_file.name}: {result.get('error', 'Unknown error')}")
                    failed_videos.append(video_file.name)
            except Exception as e:
                print(f"\nUnexpected error processing {video_file.name}: {e}")
                failed_videos.append(video_file.name)
    
    print(f"\nSuccessfully preprocessed {len(preprocessed_videos)} videos")
    if failed_videos:
        print(f"Failed to preprocess {len(failed_videos)} videos: {', '.join(failed_videos[:5])}{'...' if len(failed_videos) > 5 else ''}")
    
    # Phase 2: Send concurrent API requests
    print("\nPhase 2: Sending API requests...")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    semaphore = asyncio.Semaphore(args.concurrency)
    
    # Process with progress bar and timing
    start_time = time.time()
    results = []
    while True:
        # Create tasks for all videos
        tasks = [
            process_single_video(client, video_data, model_name, semaphore)
            for video_data in preprocessed_videos
        ]
        # Use tqdm for progress tracking with asyncio.as_completed
        with tqdm(total=len(tasks), desc="Processing videos") as pbar:
            for coro in asyncio.as_completed(tasks):
                result_data = await coro
                results.append(result_data)
                pbar.update(1)
        if not args.infinite_loop:
            break

    end_time = time.time()
    duration = end_time - start_time
    
    # Calculate statistics
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    throughput = len(results) / duration if duration > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Total videos processed: {len(results)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(failed_results)}")
    print(f"Total time: {duration:.2f} seconds")
    print(f"Throughput: {throughput:.2f} requests/second")
    print(f"{'='*60}")
    
    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"video_captions_{args.category}_{timestamp}.csv"
    
    with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['video_filename', 'caption', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow({
                'video_filename': Path(result['video_path']).name,
                'caption': result['caption'],
                'status': 'success' if result['success'] else 'failed'
            })
    
    print(f"\nResults saved to: {output_filename}")
    
    # Print some sample results
    print("\nSample results:")
    for i, result in enumerate(successful_results[:3]):
        print(f"\nVideo: {Path(result['video_path']).name}")
        print(f"Caption: {result['caption']}")
        print("-" * 60)

if __name__ == "__main__":
    asyncio.run(main())