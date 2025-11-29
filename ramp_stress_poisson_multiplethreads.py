#!/usr/bin/env python3
"""
Continuous multithreaded load test for an OpenAI-style VLM endpoint.

- Each thread continuously sends new requests as soon as previous response is received.
- Measures maximum sustainable throughput over specified duration.
- Each request uses unique frame variants to avoid backend cache reuse.
- Designed to run standalone. Prints progress and final RPS measurement.
"""

import argparse
import asyncio
import base64
import random
import time
from pathlib import Path
import statistics
from collections import defaultdict
from typing import List, Dict, Tuple
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import logging
import json
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from kneed import KneeLocator


VIDEO_DIR = Path("/home/azureuser/docker_tps_test/media_files/wyzie_250")
DEFAULT_DURATION = 120
DEFAULT_MAX_FRAMES = 10
DEFAULT_MAX_TOKENS = 8192
DEFAULT_MAX_OUTPUT_TOKENS = 60
DEFAULT_THREAD_COUNT = 64
TARGET_SIZE = 448  # Resize all images to 448x448


def setup_response_logging(log_file_path):
    """Setup logging for VLM responses."""
    # Create a logger specifically for VLM responses
    vlm_logger = logging.getLogger('vlm_responses')
    vlm_logger.setLevel(logging.INFO)

    # Remove any existing handlers
    for handler in vlm_logger.handlers[:]:
        vlm_logger.removeHandler(handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    file_handler.setFormatter(formatter)

    vlm_logger.addHandler(file_handler)
    vlm_logger.propagate = False  # Don't propagate to root logger

    return vlm_logger


def create_controlled_prompt(target_tokens):
    """Create a prompt that very explicitly requests a specific token count."""
    if target_tokens <= 30:
        prompt = (
            f"IMPORTANT: Respond with EXACTLY {target_tokens} tokens or fewer. "
            f"Provide a very brief summary of what happened in this video. "
            f"Count your words carefully and stop at {target_tokens} tokens maximum. "
            f"Focus only on the main actions and objects shown."
        )
    elif target_tokens <= 60:
        prompt = (
            f"IMPORTANT: Your response must be EXACTLY {target_tokens} tokens. "
            f"Provide a concise summary of what happened in this video in precisely {target_tokens} tokens. "
            f"Count each word as you write and ensure your total response is {target_tokens} tokens. "
            f"Describe the main actions, objects, and events shown across the video frames. "
            f"Stop immediately when you reach {target_tokens} tokens."
        )
    elif target_tokens <= 100:
        prompt = (
            f"CRITICAL: Write EXACTLY {target_tokens} tokens in your response. "
            f"Provide a moderately detailed summary of what happened in this video using precisely {target_tokens} tokens. "
            f"Count every single token/word and ensure your response totals exactly {target_tokens} tokens. "
            f"Describe the main actions, objects, events, and context shown across the video frames. "
            f"Your response must be exactly {target_tokens} tokens - no more, no less."
        )
    else:
        prompt = (
            f"MANDATORY: Your response length must be EXACTLY {target_tokens} tokens. "
            f"Provide a detailed summary of what happened in this video using precisely {target_tokens} tokens. "
            f"Count each word/token carefully as you write and ensure your total response equals {target_tokens} tokens. "
            f"Include detailed descriptions of actions, objects, events, context, and progression shown across the video frames. "
            f"Stop writing immediately when you reach exactly {target_tokens} tokens."
        )

    return prompt


def load_videos(max_count=50):
    """Load video file paths only - actual processing happens during prep phase."""
    video_files = list(sorted(VIDEO_DIR.glob("*.mp4")))
    if not video_files:
        raise RuntimeError(f"No mp4 files found in {VIDEO_DIR}")
    return video_files[:max_count]


def prepare_video_dataset(video_files: List[Path], max_frames: int, variants_per_video: int = 3) -> List[Dict]:
    """Pre-process all videos with multiple variants to avoid runtime processing delays."""
    print(f"Pre-processing {len(video_files)} videos with {variants_per_video} variants each...")
    processed_videos = []

    for video_idx, video_path in enumerate(video_files):
        try:
            for variant in range(variants_per_video):
                # Create unique ID for this variant
                unique_id = hash(f"{video_path.name}_{variant}_{int(time.time())}") % 1000000

                frames_b64 = extract_frames_to_base64(video_path, max_frames, unique_id)
                if frames_b64:
                    processed_videos.append({
                        'id': f"{video_path.stem}_v{variant}",
                        'original_path': str(video_path),
                        'frames_b64': frames_b64,
                        'frame_count': len(frames_b64),
                        'size_mb': video_path.stat().st_size / (1024 * 1024),
                        'variant': variant,
                        'unique_id': unique_id
                    })

            print(f"✓ Processed {video_path.name} ({variants_per_video} variants, {len(frames_b64)} frames each)")

        except Exception as e:
            print(f"✗ Failed to process {video_path.name}: {e}")

    print(f"Pre-processing complete: {len(processed_videos)} video variants ready")
    return processed_videos


def extract_frames_to_base64(video_path: Path, max_frames: int, unique_id: int):
    """Extract frames with per-request noise to avoid cache collisions."""
    cap = cv2.VideoCapture(str(video_path))
    frames_b64 = []
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return []

        uniq_random = random.Random(unique_id)
        frame_indices = []
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            for _ in range(max_frames):
                frame_indices.append(uniq_random.randint(0, total_frames - 1))

        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Resize to exactly 448x448 for consistent input size and optimal performance
            frame = cv2.resize(frame, (TARGET_SIZE, TARGET_SIZE))

            # Add small noise based on unique_id to avoid identical hashes
            noise = np.random.default_rng(unique_id + i).standard_normal(frame.shape, dtype=np.float32) * 1.5
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if success:
                frames_b64.append(base64.b64encode(buffer).decode("utf-8"))
    finally:
        cap.release()
    return frames_b64




def send_request_sync_optimized(session, base_url, video_data, request_id, max_tokens, max_output_tokens, vlm_logger=None):
    """Optimized synchronous request with persistent session to minimize latency."""
    # Use pre-processed frames directly - no runtime processing
    frames_b64 = video_data['frames_b64']
    unique_id = video_data['unique_id'] + random.randint(0, 1000)  # Add small random component

    # Create controlled prompt based on target output tokens
    controlled_prompt = create_controlled_prompt(max_output_tokens)

    content = [{"type": "text", "text": controlled_prompt}]
    for fb64 in frames_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{fb64}"}})

    payload = {
        "model": "OpenGVLab/InternVL3-8B",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "seed": unique_id,
    }

    # Add max_completion_tokens if supported by the API
    if max_output_tokens and max_output_tokens != max_tokens:
        payload["max_completion_tokens"] = max_output_tokens

    start = time.time()
    try:
        # Use persistent session for connection reuse and reduced latency
        response = session.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=60
        )
        elapsed = time.time() - start
        ok = response.status_code == 200
        detail = ""
        output_tokens = 0
        full_response = ""

        if ok:
            try:
                data = response.json()
                full_response = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                detail = full_response[:80] + ("..." if len(full_response) > 80 else "")

                # Try to get token usage info
                usage = data.get("usage", {})
                output_tokens = usage.get("completion_tokens", len(full_response.split()) if full_response else 0)

                # Log the full response if logger is provided
                if vlm_logger and full_response:
                    log_entry = {
                        "request_id": request_id,
                        "video_id": video_data['id'],
                        "prompt_tokens": max_output_tokens,
                        "actual_tokens": output_tokens,
                        "latency": elapsed,
                        "response": full_response,
                        "frame_count": len(frames_b64)
                    }
                    vlm_logger.info(json.dumps(log_entry, ensure_ascii=False))

            except Exception as e:
                ok = False
                detail = f"json_err:{e}"
        else:
            detail = f"HTTP {response.status_code}"

        return {
            "id": request_id,
            "ok": ok,
            "status": response.status_code,
            "latency": elapsed,
            "detail": detail,
            "output_tokens": output_tokens,
            "video_id": video_data['id'],
            "full_response": full_response
        }
    except Exception as e:
        return {
            "id": request_id,
            "ok": False,
            "status": 0,
            "latency": time.time() - start,
            "error": str(e),
            "output_tokens": 0,
            "video_id": video_data.get('id', 'unknown'),
            "full_response": ""
        }


def send_continuous_requests_synchronized(base_url, processed_videos, max_tokens, max_output_tokens, results_list, results_lock, stop_flag, start_barrier, test_start_time, vlm_logger=None):
    """Continuously send requests with synchronized start and optimized latency."""
    request_count = 0

    # Create persistent session for this thread to reduce connection overhead
    session = requests.Session()

    # Configure session for optimal performance
    session.headers.update({
        'Connection': 'keep-alive',
        'Content-Type': 'application/json'
    })

    # Pre-select videos for this thread to minimize choice overhead
    thread_videos = [random.choice(processed_videos) for _ in range(1000)]  # Pre-select 1000 videos
    video_index = 0

    try:
        # Wait for all threads to be ready, then start simultaneously
        start_barrier.wait()

        # Record actual start time after barrier
        thread_start_time = time.time()

        while not stop_flag.is_set():
            request_count += 1

            # Use pre-selected video to minimize selection overhead
            video_data = thread_videos[video_index % len(thread_videos)]
            video_index += 1

            request_id = f"req_{threading.current_thread().name}_{request_count}"

            # Send request with optimized session (this will block until response received)
            result = send_request_sync_optimized(session, base_url, video_data, request_id, max_tokens, max_output_tokens, vlm_logger)

            # Add thread-specific info
            result['thread_id'] = threading.current_thread().name
            result['thread_request_count'] = request_count
            result['test_time'] = time.time() - test_start_time
            result['request_start_offset'] = time.time() - thread_start_time

            # Thread-safely append result
            with results_lock:
                results_list.append(result)

            # Print progress occasionally (every 10th request per thread)
            if request_count % 10 == 0:
                status = "✓" if result.get('ok') else "✗"
                elapsed_time = time.time() - thread_start_time
                rps = request_count / elapsed_time if elapsed_time > 0 else 0
                actual_tokens = result.get('output_tokens', 0)

                # Highlight token control effectiveness
                token_status = ""
                if actual_tokens > 0:
                    deviation = abs(actual_tokens - max_output_tokens)
                    if deviation <= 2:
                        token_status = f"tokens: {actual_tokens}/{max_output_tokens} ✓EXACT"
                    elif deviation <= 5:
                        token_status = f"tokens: {actual_tokens}/{max_output_tokens} ~close"
                    else:
                        token_status = f"tokens: {actual_tokens}/{max_output_tokens} !off"
                else:
                    token_status = f"tokens: 0/{max_output_tokens} ✗fail"

                print(f"{status} {threading.current_thread().name}: {request_count} requests, {rps:.2f} RPS, latency: {result.get('latency', 0):.3f}s, {token_status}")

    finally:
        session.close()
        print(f"Thread {threading.current_thread().name} completed {request_count} requests")


def analyze_latency_vs_rps(results: List[Dict], target_rps: float, duration: float) -> Dict:
    """Analyze latency patterns vs RPS - showing accdtual individual latencies instead of smoothed averages."""
    if not results:
        return {}

    # Add timestamps relative to start
    start_time = min(r.get('start_time', 0) for r in results if 'start_time' in r)
    if start_time == 0:
        # If no start_time, estimate from request order
        for i, r in enumerate(results):
            r['estimated_time'] = (i / len(results)) * duration

    # Collect individual request data points (no windowing for actual latencies)
    individual_points = []
    for r in results:
        if not r.get('ok'):
            continue

        # Use actual timestamp if available, otherwise estimate
        t = r.get('start_time', r.get('estimated_time', 0))
        if start_time > 0:
            t -= start_time

        # Calculate instantaneous RPS based on time in ramp
        instantaneous_rps = min(target_rps, (t / duration) * target_rps)

        individual_points.append({
            'latency': r.get('latency', 0),
            'time': t,
            'rps': instantaneous_rps,
            'request_id': r.get('id', 'unknown')
        })

    # Sort by RPS for analysis
    individual_points.sort(key=lambda x: x['rps'])

    # Create smaller windows only for summary statistics, but keep individual points
    window_size = max(2.0, duration / 20)  # Adaptive window size, minimum 2 seconds
    time_windows = defaultdict(list)

    for point in individual_points:
        window_idx = int(point['time'] // window_size)
        time_windows[window_idx].append(point)

    # Calculate windowed statistics (for summary only)
    windowed_stats = []
    for window_idx in sorted(time_windows.keys()):
        window_data = time_windows[window_idx]
        if len(window_data) < 1:  # Allow single data points
            continue

        latencies = [d['latency'] for d in window_data]
        avg_rps = statistics.mean(d['rps'] for d in window_data)

        windowed_stats.append({
            'window_start': window_idx * window_size,
            'rps': avg_rps,
            'latency_mean': statistics.mean(latencies),
            'latency_median': statistics.median(latencies) if len(latencies) > 1 else latencies[0],
            'latency_p95': np.percentile(latencies, 95) if len(latencies) > 1 else latencies[0],
            'latency_p99': np.percentile(latencies, 99) if len(latencies) > 1 else latencies[0],
            'latency_std': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'request_count': len(window_data)
        })

    # Detect knee point using individual data points (more accurate)
    knee_point = None
    if len(individual_points) > 10:
        # Use individual points for knee detection instead of windowed averages
        rps_values = [p['rps'] for p in individual_points]
        latency_values = [p['latency'] for p in individual_points]

        try:
            kneedle = KneeLocator(
                rps_values, latency_values,
                curve="convex", direction="increasing"
            )
            if kneedle.knee:
                knee_point = {
                    'rps': kneedle.knee,
                    'latency': kneedle.knee_y,
                    'detected': True
                }
        except Exception as e:
            print(f"Knee detection failed: {e}")

    # Calculate latency increase analysis using individual points
    baseline_latency = None
    degradation_points = []

    if len(individual_points) >= 10:
        # Use first 10% of requests as baseline (low RPS period)
        baseline_count = max(5, len(individual_points) // 10)
        baseline_latencies = [p['latency'] for p in individual_points[:baseline_count]]
        baseline_latency = statistics.mean(baseline_latencies)

        # Look for degradation in individual points
        for i, point in enumerate(individual_points):
            if i < baseline_count:
                continue

            increase_ratio = point['latency'] / baseline_latency if baseline_latency > 0 else 1
            if increase_ratio > 2.0:  # 100% increase threshold for individual points
                degradation_points.append({
                    'rps': point['rps'],
                    'latency': point['latency'],
                    'increase_ratio': increase_ratio,
                    'time': point['time'],
                    'request_id': point['request_id']
                })
                # Only report first few degradation points to avoid spam
                if len(degradation_points) >= 5:
                    break

    # Calculate average output tokens
    avg_output_tokens = 0
    if results:
        valid_token_counts = [r.get('output_tokens', 0) for r in results if r.get('ok') and r.get('output_tokens', 0) > 0]
        if valid_token_counts:
            avg_output_tokens = sum(valid_token_counts) / len(valid_token_counts)

    return {
        'individual_points': individual_points,  # Add individual data points
        'windowed_stats': windowed_stats,
        'knee_point': knee_point,
        'baseline_latency': baseline_latency,
        'degradation_points': degradation_points,
        'total_requests': len(results),
        'successful_requests': len([r for r in results if r.get('ok')]),
        'avg_output_tokens': avg_output_tokens
    }


def plot_latency_analysis(analysis: Dict, target_rps: float, save_path: str = None, config: Dict = None):
    """Create comprehensive latency vs RPS plots showing actual individual latencies."""
    if not analysis or (not analysis.get('individual_points') and not analysis.get('windowed_stats')):
        print("No data to plot")
        return

    if config is None:
        config = {}  # Fallback for backward compatibility

    individual_points = analysis.get('individual_points', [])
    windowed_stats = analysis.get('windowed_stats', [])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Latency vs RPS Analysis - Individual Request Data', fontsize=16, fontweight='bold')

    # Plot 1: Individual latency points vs RPS (main analysis)
    if individual_points:
        individual_rps = [p['rps'] for p in individual_points]
        individual_latencies = [p['latency'] for p in individual_points]

        # Scatter plot of individual requests
        ax1.scatter(individual_rps, individual_latencies, alpha=0.6, s=20, c='blue', label='Individual Requests')

        # Add trend line if we have windowed stats for reference
        if windowed_stats:
            rps_values = [w['rps'] for w in windowed_stats]
            latency_means = [w['latency_mean'] for w in windowed_stats]
            ax1.plot(rps_values, latency_means, 'r-', label='Mean Trend', linewidth=2, alpha=0.8)
    else:
        # Fallback to windowed stats if individual points not available
        rps_values = [w['rps'] for w in windowed_stats]
        latency_means = [w['latency_mean'] for w in windowed_stats]
        latency_p95 = [w['latency_p95'] for w in windowed_stats]
        latency_p99 = [w['latency_p99'] for w in windowed_stats]

        ax1.plot(rps_values, latency_means, 'b-o', label='Mean Latency', linewidth=2, markersize=4)
        ax1.plot(rps_values, latency_p95, 'r-s', label='P95 Latency', linewidth=2, markersize=4)
        ax1.plot(rps_values, latency_p99, 'orange', marker='^', label='P99 Latency', linewidth=2, markersize=4)

    # Mark knee point if detected
    if analysis.get('knee_point') and analysis['knee_point'].get('detected'):
        knee = analysis['knee_point']
        ax1.axvline(x=knee['rps'], color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax1.text(knee['rps'], knee['latency'], f'Knee: {knee["rps"]:.1f} RPS',
                rotation=90, verticalalignment='bottom', fontweight='bold')

    # Mark degradation points
    degradation_points = analysis.get('degradation_points', [])
    if degradation_points:
        deg_rps = [deg['rps'] for deg in degradation_points[:5]]  # Limit to first 5
        deg_latencies = [deg['latency'] for deg in degradation_points[:5]]
        ax1.scatter(deg_rps, deg_latencies, c='red', s=100, marker='x', linewidth=3,
                   label=f'Degradation Points ({len(degradation_points)} total)', zorder=5)

    ax1.set_xlabel('RPS (Requests per Second)')
    ax1.set_ylabel('Latency (seconds)')
    ax1.set_title('Actual Latency vs RPS (Individual Requests)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Time series of RPS and individual latencies
    if individual_points:
        times_individual = [p['time'] for p in individual_points]
        rps_individual = [p['rps'] for p in individual_points]
        ax2.scatter(times_individual, rps_individual, alpha=0.6, s=10, c='green', label='Request RPS')
        ax2.plot([0, max(times_individual)], [0, target_rps], 'r--', label='Target Ramp', linewidth=2)
    elif windowed_stats:
        times = [w['window_start'] for w in windowed_stats]
        rps_values = [w['rps'] for w in windowed_stats]
        ax2.plot(times, rps_values, 'g-', linewidth=2)

    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('RPS')
    ax2.set_title('RPS Ramp Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Individual latency over time
    if individual_points:
        times_individual = [p['time'] for p in individual_points]
        latencies_individual = [p['latency'] for p in individual_points]
        ax3.scatter(times_individual, latencies_individual, alpha=0.6, s=20, c='blue', label='Individual Requests')

        # Add windowed trend if available
        if windowed_stats:
            times = [w['window_start'] for w in windowed_stats]
            latency_means = [w['latency_mean'] for w in windowed_stats]
            ax3.plot(times, latency_means, 'r-', label='Mean Trend', linewidth=2, alpha=0.8)
    else:
        # Fallback to windowed stats
        times = [w['window_start'] for w in windowed_stats]
        latency_means = [w['latency_mean'] for w in windowed_stats]
        latency_p95 = [w['latency_p95'] for w in windowed_stats]
        ax3.plot(times, latency_means, 'b-', label='Mean', linewidth=2)
        ax3.fill_between(times,
                         [w['latency_mean'] - w['latency_std'] for w in windowed_stats],
                         [w['latency_mean'] + w['latency_std'] for w in windowed_stats],
                         alpha=0.3, color='blue', label='±1 Std Dev')
        ax3.plot(times, latency_p95, 'r--', label='P95', linewidth=2)

    # Mark baseline if available
    if analysis.get('baseline_latency'):
        ax3.axhline(y=analysis['baseline_latency'], color='green', linestyle=':',
                   label=f'Baseline: {analysis["baseline_latency"]:.2f}s')
        ax3.axhline(y=analysis['baseline_latency'] * 1.5, color='orange', linestyle=':',
                   label='1.5x Baseline')

    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Latency (seconds)')
    ax3.set_title('Individual Request Latencies Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Request rate and latency distribution
    if windowed_stats:
        times = [w['window_start'] for w in windowed_stats]
        request_counts = [w['request_count'] for w in windowed_stats]
        ax4.bar(times, request_counts, width=max(2.0, (max(times) if times else 60) / 20),
                alpha=0.6, color='skyblue', label='Requests/Window')
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Requests per Window')
        ax4.set_title('Request Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        # Show latency histogram if no windowed stats
        if individual_points:
            latencies = [p['latency'] for p in individual_points]
            ax4.hist(latencies, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            ax4.set_xlabel('Latency (seconds)')
            ax4.set_ylabel('Request Count')
            ax4.set_title('Latency Distribution')
            ax4.grid(True, alpha=0.3)

    # Add configuration text box to the plot
    config_text = f"Configuration:\n"
    config_text += f"• Target RPS: {target_rps:.1f}\n"
    config_text += f"• Frames/Video: {config.get('max_frames', 'N/A')}\n"
    config_text += f"• Max Tokens: {config.get('max_tokens', 'N/A')}\n"
    config_text += f"• Max Output Tokens: {config.get('max_output_tokens', 'N/A')}\n"
    config_text += f"• Video Variants: {config.get('video_count', 'N/A')}\n"
    config_text += f"• Test Duration: {config.get('duration', 'N/A')}s\n"
    config_text += f"• Threads: {config.get('thread_count', 'N/A')}"

    # Add text box to bottom right plot
    ax4.text(0.02, 0.98, config_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Add note about individual data points
    note_text = f"Note: Showing {len(individual_points) if individual_points else 0} individual request latencies.\nNo smoothing or averaging applied to main plot."
    ax4.text(0.02, 0.35, note_text, transform=ax4.transAxes, fontsize=8,
             verticalalignment='top', style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()
    return fig


def print_analysis_summary(analysis: Dict, target_rps: float):
    """Print a detailed analysis summary with individual request data."""
    print("\n" + "="*60)
    print("LATENCY ANALYSIS SUMMARY - INDIVIDUAL REQUEST DATA")
    print("="*60)

    if not analysis:
        print("No analysis data available")
        return

    print(f"Target RPS: {target_rps:.2f}")
    print(f"Total Requests: {analysis.get('total_requests', 0)}")
    print(f"Successful Requests: {analysis.get('successful_requests', 0)}")
    print(f"Individual Data Points: {len(analysis.get('individual_points', []))}")
    print(f"Analysis Method: Individual request latencies (no smoothing/averaging)")

    if analysis.get('avg_output_tokens', 0) > 0:
        print(f"Average Output Tokens: {analysis['avg_output_tokens']:.0f}")

    if analysis.get('baseline_latency'):
        print(f"Baseline Latency: {analysis['baseline_latency']:.3f}s")

    # Knee point analysis
    knee = analysis.get('knee_point')
    if knee and knee.get('detected'):
        print(f"\n🔍 KNEE POINT DETECTED:")
        print(f"  RPS: {knee['rps']:.2f}")
        print(f"  Latency: {knee['latency']:.3f}s")
        print(f"  Recommendation: Optimal operating point is around {knee['rps']:.1f} RPS")
    else:
        print("\n🔍 No clear knee point detected")

    # Degradation analysis
    degradation_points = analysis.get('degradation_points', [])
    if degradation_points:
        print(f"\n⚠️  PERFORMANCE DEGRADATION DETECTED ({len(degradation_points)} requests):")
        for i, deg in enumerate(degradation_points[:3]):  # Show first 3
            req_id = deg.get('request_id', 'unknown')
            print(f"  Request {req_id}: {deg['rps']:.2f} RPS, {deg['latency']:.3f}s ({deg['increase_ratio']:.1f}x baseline)")
        if len(degradation_points) > 3:
            print(f"  ... and {len(degradation_points) - 3} more requests")
    else:
        print("\n✅ No significant performance degradation detected")

    # Individual request statistics summary
    individual_points = analysis.get('individual_points', [])
    if individual_points:
        # Calculate stats from individual points at high RPS
        high_rps_points = [p for p in individual_points if p['rps'] > target_rps * 0.8]
        if high_rps_points:
            high_latencies = [p['latency'] for p in high_rps_points]
            print(f"\n📊 HIGH RPS PERFORMANCE (>{target_rps * 0.8:.1f} RPS):")
            print(f"  Requests at High RPS: {len(high_rps_points)}")
            print(f"  Mean Latency: {statistics.mean(high_latencies):.3f}s")
            print(f"  Median Latency: {statistics.median(high_latencies):.3f}s")
            print(f"  Min/Max Latency: {min(high_latencies):.3f}s / {max(high_latencies):.3f}s")
            if len(high_latencies) > 1:
                print(f"  P95 Latency: {np.percentile(high_latencies, 95):.3f}s")
                print(f"  P99 Latency: {np.percentile(high_latencies, 99):.3f}s")

        print(f"\n📊 OVERALL LATENCY PATTERN:")
        all_latencies = [p['latency'] for p in individual_points]
        print(f"  Total Latency Range: {min(all_latencies):.3f}s - {max(all_latencies):.3f}s")
        print(f"  Overall Mean: {statistics.mean(all_latencies):.3f}s")
        print(f"  Note: Individual request data, no temporal averaging applied")
    elif analysis.get('windowed_stats'):
        # Fallback to windowed stats
        stats = analysis['windowed_stats']
        final_stats = stats[-1] if stats else None
        if final_stats:
            print(f"\n📊 FINAL PERFORMANCE (Windowed):")
            print(f"  Final RPS: {final_stats['rps']:.2f}")
            print(f"  Final Mean Latency: {final_stats['latency_mean']:.3f}s")
            print(f"  Final P95 Latency: {final_stats['latency_p95']:.3f}s")
            print(f"  Final P99 Latency: {final_stats['latency_p99']:.3f}s")


def run_continuous_load_test(base_url: str, duration: float, max_frames: int, max_tokens: int, max_output_tokens: int, thread_count: int = DEFAULT_THREAD_COUNT):
    # Pre-process all videos before starting the test to avoid runtime delays
    print(f"Preparing video dataset (frames: {max_frames}, tokens: {max_tokens}, output: {max_output_tokens})...")
    video_files = load_videos()
    processed_videos = prepare_video_dataset(video_files, max_frames)

    if not processed_videos:
        raise RuntimeError("No processed videos available for testing")

    # Setup response logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = f"vlm_responses_{thread_count}threads_{timestamp}.jsonl"
    vlm_logger = setup_response_logging(log_file_path)

    print(f"Starting continuous load test with {len(processed_videos)} pre-processed video variants")
    print(f"Using {thread_count} threads for maximum throughput")
    print(f"Test duration: {duration:.1f}s")
    print()
    print(f"🎯 TOKEN CONTROL CONFIGURATION:")
    print(f"   • Target: EXACTLY {max_output_tokens} tokens per response")
    print(f"   • Prompt: Explicitly demands {max_output_tokens} tokens")
    print(f"   • Monitoring: Real-time token compliance tracking")
    print()
    print(f"📝 Response logging: {log_file_path}")
    print(f"🚀 Each thread will continuously send requests until test ends")
    print()

    # Shared resources
    results_list = []
    results_lock = threading.Lock()
    stop_flag = threading.Event()

    # Barrier to synchronize thread startup - all threads wait until all are ready
    start_barrier = threading.Barrier(thread_count + 1)  # +1 for main thread

    print(f"Launching {thread_count} threads...")

    # Launch worker threads
    threads = []
    for i in range(thread_count):
        thread = threading.Thread(
            target=send_continuous_requests_synchronized,
            args=(base_url, processed_videos, max_tokens, max_output_tokens, results_list, results_lock, stop_flag, start_barrier, time.time(), vlm_logger),
            name=f"worker_{i}"
        )
        thread.start()
        threads.append(thread)

    print(f"All threads launched. Waiting for synchronization...")

    # Wait for all threads to be ready, then release them simultaneously
    start_barrier.wait()

    # Record actual test start time (after synchronization)
    start_time = time.time()

    print(f"🚀 All {thread_count} threads started simultaneously!")
    print(f"Test running for {duration:.1f}s...")
    print()

    # Let threads run for the specified duration
    time.sleep(duration)

    # Signal all threads to stop
    stop_flag.set()

    # Wait for all threads to finish
    for thread in threads:
        thread.join(timeout=5.0)  # Give threads 5 seconds to finish current requests

    # Calculate actual test duration
    end_time = time.time()
    actual_duration = end_time - start_time

    # Process results
    ok = [r for r in results_list if r.get("ok")]
    errs = [r for r in results_list if not r.get("ok")]

    if not results_list:
        print("No requests completed!")
        return [], {}

    # Calculate metrics
    avg_lat = sum(r.get("latency", 0) for r in ok) / len(ok) if ok else 0
    median_lat = sorted([r.get("latency", 0) for r in ok])[len(ok)//2] if ok else 0
    avg_output_tokens = sum(r.get("output_tokens", 0) for r in ok) / len(ok) if ok else 0

    # Calculate actual measured RPS
    actual_rps = len(ok) / actual_duration if actual_duration > 0 else 0.0
    total_rps = len(results_list) / actual_duration if actual_duration > 0 else 0.0

    # Print results
    print(f"\n" + "="*60)
    print("CONTINUOUS LOAD TEST RESULTS")
    print("="*60)
    print(f"Test Configuration:")
    print(f"  • Duration: {actual_duration:.2f}s")
    print(f"  • Thread Count: {thread_count}")
    print(f"  • Video Variants: {len(processed_videos)}")
    print(f"  • Image Size: {TARGET_SIZE}x{TARGET_SIZE} (optimized)")
    print(f"  • Synchronized Start: Yes")
    print(f"  • Persistent Sessions: Yes")
    print(f"")
    print(f"Request Results:")
    print(f"  • Total Requests: {len(results_list)}")
    print(f"  • Successful: {len(ok)}")
    print(f"  • Failed: {len(errs)}")
    print(f"  • Success Rate: {(len(ok)/len(results_list)*100) if results_list else 0:.1f}%")
    print(f"")
    print(f"Throughput:")
    print(f"  • Actual RPS (successful): {actual_rps:.2f}")
    print(f"  • Total RPS (all requests): {total_rps:.2f}")
    print(f"")
    print(f"Latency (successful requests):")
    print(f"  • Average: {avg_lat:.3f}s")
    print(f"  • Median: {median_lat:.3f}s")
    if ok:
        latencies = [r.get("latency", 0) for r in ok]
        print(f"  • Min: {min(latencies):.3f}s")
        print(f"  • Max: {max(latencies):.3f}s")

        # Calculate first request timing spread (synchronization effectiveness)
        first_requests = [r for r in ok if r.get('thread_request_count') == 1]
        if len(first_requests) > 1:
            start_times = [r.get('request_start_offset', 0) for r in first_requests]
            sync_spread = max(start_times) - min(start_times) if start_times else 0
            print(f"  • First Request Sync Spread: {sync_spread*1000:.1f}ms")

    print(f"")
    print(f"TOKEN CONTROL ANALYSIS:")
    print(f"  🎯 TARGET: {max_output_tokens} tokens per response")
    print(f"  📊 ACTUAL: {avg_output_tokens:.1f} average tokens")

    # Calculate token control effectiveness with clear categorization
    if ok:
        token_counts = [r.get('output_tokens', 0) for r in ok if r.get('output_tokens', 0) > 0]
        if token_counts:
            min_tokens = min(token_counts)
            max_tokens = max(token_counts)
            token_deviation = abs(avg_output_tokens - max_output_tokens)
            deviation_pct = abs(token_deviation/max_output_tokens*100)

            # Categorize control effectiveness
            if deviation_pct <= 5:
                control_grade = "EXCELLENT ✓✓✓"
            elif deviation_pct <= 10:
                control_grade = "GOOD ✓✓"
            elif deviation_pct <= 20:
                control_grade = "FAIR ✓"
            else:
                control_grade = "POOR ✗"

            # Count exact/close/off responses
            exact_count = len([t for t in token_counts if abs(t - max_output_tokens) <= 2])
            close_count = len([t for t in token_counts if 2 < abs(t - max_output_tokens) <= 5])
            off_count = len([t for t in token_counts if abs(t - max_output_tokens) > 5])

            print(f"  📏 RANGE: {min_tokens}-{max_tokens} tokens")
            print(f"  🎯 CONTROL QUALITY: {control_grade} ({deviation_pct:.1f}% deviation)")
            print(f"  📈 DISTRIBUTION: {exact_count} exact (±2), {close_count} close (±5), {off_count} off (>5)")

    print(f"")
    print(f"Other Performance Metrics:")

    # Calculate per-thread efficiency
    if ok:
        thread_stats = {}
        for result in ok:
            thread_id = result.get('thread_id', 'unknown')
            if thread_id not in thread_stats:
                thread_stats[thread_id] = []
            thread_stats[thread_id].append(result)

        if thread_stats:
            thread_counts = [len(requests) for requests in thread_stats.values()]
            print(f"  • Requests per Thread: min={min(thread_counts)}, max={max(thread_counts)}, avg={sum(thread_counts)/len(thread_counts):.1f}")

    print(f"  • Response Log: {log_file_path}")
    print("="*60)

    # Print the main RPS as a simple float number as requested
    print(f"\nActual RPS: {actual_rps:.2f}")

    return results_list, {'actual_duration': actual_duration, 'actual_rps': actual_rps, 'total_rps': total_rps}


def parse_thread_counts(thread_str):
    """Parse thread count argument - can be single number or comma-separated list."""
    try:
        if ',' in thread_str:
            # Comma-separated list
            thread_counts = [int(x.strip()) for x in thread_str.split(',')]
        else:
            # Single number
            thread_counts = [int(thread_str)]

        # Validate all thread counts are positive
        for count in thread_counts:
            if count <= 0:
                raise ValueError(f"Thread count must be positive: {count}")

        return thread_counts
    except ValueError as e:
        raise ValueError(f"Invalid thread count format '{thread_str}': {e}")


def print_final_summary(all_results):
    """Print a final summary comparing all thread count results."""
    print(f"\n" + "="*80)
    print("FINAL SUMMARY - THREAD SCALING ANALYSIS")
    print("="*80)

    print(f"{'Threads':<8} {'RPS':<8} {'Success%':<9} {'Avg Lat':<9} {'Med Lat':<9} {'Total Req':<10}")
    print("-" * 80)

    for result in all_results:
        threads = result['threads']
        rps = result['actual_rps']
        success_rate = result['success_rate']
        avg_lat = result['avg_latency']
        med_lat = result['median_latency']
        total_req = result['total_requests']

        print(f"{threads:<8} {rps:<8.2f} {success_rate:<9.1f} {avg_lat:<9.3f} {med_lat:<9.3f} {total_req:<10}")

    if len(all_results) > 1:
        print("\n" + "-" * 80)
        print("Scaling Analysis:")

        best_rps = max(all_results, key=lambda x: x['actual_rps'])
        print(f"  • Best RPS: {best_rps['actual_rps']:.2f} at {best_rps['threads']} threads")

        lowest_latency = min(all_results, key=lambda x: x['avg_latency'])
        print(f"  • Lowest Avg Latency: {lowest_latency['avg_latency']:.3f}s at {lowest_latency['threads']} threads")

        # Calculate efficiency (RPS per thread)
        for result in all_results:
            result['rps_per_thread'] = result['actual_rps'] / result['threads']

        best_efficiency = max(all_results, key=lambda x: x['rps_per_thread'])
        print(f"  • Best Efficiency: {best_efficiency['rps_per_thread']:.3f} RPS/thread at {best_efficiency['threads']} threads")

        # Show scaling pattern
        if len(all_results) >= 3:
            print("\n  Thread Scaling Pattern:")
            for i, result in enumerate(all_results):
                if i == 0:
                    print(f"    {result['threads']} threads: {result['actual_rps']:.2f} RPS (baseline)")
                else:
                    prev_rps = all_results[i-1]['actual_rps']
                    scaling_factor = result['actual_rps'] / prev_rps if prev_rps > 0 else 0
                    print(f"    {result['threads']} threads: {result['actual_rps']:.2f} RPS ({scaling_factor:.2f}x previous)")

    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Continuous multithreaded load test - measures maximum sustainable throughput")
    parser.add_argument("--host", default="http://localhost:30727", help="Base host URL")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION, help="Test duration in seconds")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="Max frames per video")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max total tokens")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="Max output/completion tokens")
    parser.add_argument("--threads", type=str, default=str(DEFAULT_THREAD_COUNT), help=f"Number of threads for concurrent requests. Can be single number (e.g., 64) or comma-separated list (e.g., 1,2,4,8,16,32,64) to run multiple tests (default: {DEFAULT_THREAD_COUNT})")
    args = parser.parse_args()

    # Parse thread counts
    try:
        thread_counts = parse_thread_counts(args.threads)
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"Starting continuous load test series")
    print(f"Configuration: {args.max_frames} frames, {args.max_tokens} max tokens, {args.max_output_tokens} max output tokens")
    print(f"Duration per test: {args.duration}s")

    if len(thread_counts) == 1:
        print(f"Thread count: {thread_counts[0]}")
    else:
        print(f"Thread counts: {', '.join(map(str, thread_counts))}")
        print(f"Total tests to run: {len(thread_counts)}")

    print(f"Each thread continuously sends new requests upon receiving responses")
    print()

    all_results = []

    try:
        for i, thread_count in enumerate(thread_counts):
            if len(thread_counts) > 1:
                print(f"\n{'='*60}")
                print(f"TEST {i+1}/{len(thread_counts)}: {thread_count} THREADS")
                print(f"{'='*60}")

            results, analysis = run_continuous_load_test(
                args.host, args.duration,
                args.max_frames, args.max_tokens, args.max_output_tokens, thread_count
            )

            # Store results for final summary
            if results:
                ok = [r for r in results if r.get("ok")]
                errs = [r for r in results if not r.get("ok")]

                # Use actual measured duration from the function
                actual_duration = analysis.get('actual_duration', args.duration)

                all_results.append({
                    'threads': thread_count,
                    'actual_rps': analysis.get('actual_rps', len(ok) / actual_duration if actual_duration > 0 else 0),
                    'total_rps': analysis.get('total_rps', len(results) / actual_duration if actual_duration > 0 else 0),
                    'success_rate': (len(ok)/len(results)*100) if results else 0,
                    'avg_latency': sum(r.get("latency", 0) for r in ok) / len(ok) if ok else 0,
                    'median_latency': sorted([r.get("latency", 0) for r in ok])[len(ok)//2] if ok else 0,
                    'total_requests': len(results),
                    'successful_requests': len(ok),
                    'failed_requests': len(errs),
                })

            print(f"\nTest {i+1} completed with {len(results)} total requests")

            # Add small delay between tests
            if i < len(thread_counts) - 1:
                print(f"\nWaiting 3 seconds before next test...")
                time.sleep(3)

        # Print final summary if multiple tests were run
        if len(all_results) > 1:
            print_final_summary(all_results)

        print(f"\nAll tests completed successfully!")

    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        if all_results:
            print("\nPartial results:")
            print_final_summary(all_results)
    except Exception as e:
        print(f"\nTest failed: {e}")


if __name__ == "__main__":
    main()



'''
# Single thread count with controlled output tokens:
python3 ramp_stress_poisson_multiplethreads.py --duration 120 --max-frames 10 --max-tokens 4096 --max-output-tokens 60 --threads 64

# Multiple thread counts for scaling analysis with brief responses:
python3 ramp_stress_poisson_multiplethreads.py --duration 60 --max-frames 10 --max-tokens 4096 --max-output-tokens 30 --threads 1,2,4,8,16,32,64

# Note: All responses will be logged to timestamped JSONL files
# Images are resized to 448x448 for optimal performance
# Prompts automatically adjust based on --max-output-tokens value
'''