import os
import time
import random
import re
import json
import shutil
import traceback
import subprocess
from tqdm import tqdm
from moviepy.editor import VideoFileClip

def sanitize_filename(filename):
    filename = filename.strip()
    sanitized = filename.replace(' ', '_')
    sanitized = re.sub(r'''[\/:*?"<>|,()'\[\]]''', '', sanitized)
    sanitized = re.sub(r'_{2,}', '_', sanitized)
    return sanitized

def download_with_ytdlp(url, output_dir, filename, timeout=180):
    filename = sanitize_filename(filename)
    output_path = os.path.join(output_dir, filename + '.mp4')
    ytdlp_cmd = [
        'python', '-m', 'yt_dlp',
        '-f', 'best[ext=mp4]/best',
        '--cookies', 'C:/Users/shaff/Downloads/cookies.txt',
        '--quiet',
        '--no-warnings',
        '--output', output_path,
        url
    ]

    try:
        proc = subprocess.Popen(ytdlp_cmd)
        start_time = time.time()
        while proc.poll() is None:
            time.sleep(1)
            if time.time() - start_time > timeout:
                proc.kill()
                print(f"Timeout reached: {filename}")
                return "timeout"
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"Downloaded: {filename}")
            return output_path
        else:
            print(f"Warning: yt-dlp produced empty file: {filename}")
            return None
    except Exception as e:
        print(f"Unexpected error during yt-dlp download: {filename}")
        traceback.print_exc()
        return None

def make_dataset(path='MAS', videos_json_path='jsons/videos.json', annotations_json_path='jsons/annotations.json'):
    os.makedirs(os.path.join(path, 'temp'), exist_ok=True)
    os.makedirs(os.path.join(path, 'clips'), exist_ok=True)

    retry_json_path = os.path.join(path, 'retrycutpastes.txt')
    retry_dict = {}

    if os.path.exists(annotations_json_path):
        shutil.copy(annotations_json_path, os.path.join(path, 'annotations.json'))
    else:
        print(f"Warning: annotations file {annotations_json_path} not found")

    try:
        with open(videos_json_path) as f:
            videos = json.load(f)
    except Exception as e:
        print(f"Error reading videos JSON: {e}")
        traceback.print_exc()
        return

    successful_downloads = 0
    failed_downloads = 0

    for video_id in tqdm(videos, desc="Processing Videos"):
        try:
            link = videos[video_id][0]
            clips = videos[video_id][1]

            video_path = download_with_ytdlp(link, os.path.join(path, 'temp'), video_id)

            if video_path == "timeout" or not video_path:
                retry_dict[video_id] = [link, clips]
                failed_downloads += 1
                continue

            try:
                video_obj = VideoFileClip(video_path)
            except Exception as e:
                print(f"Error loading video with moviepy: {video_id}")
                traceback.print_exc()
                retry_dict[video_id] = [link, clips]
                failed_downloads += 1
                continue

            for start, duration, title in clips:
                try:
                    clip_title = sanitize_filename(title)
                    clip_path = os.path.join(path, 'clips', f"{clip_title}.mp4")
                    clip = video_obj.subclip(start, start + duration)
                    clip.write_videofile(
                        clip_path,
                        codec='libx264',
                        audio_codec='aac',
                        verbose=False,
                        logger=None
                    )
                    print(f"Created clip: {title}")
                except Exception as e:
                    print(f"Error extracting clip {title}")
                    traceback.print_exc()

            video_obj.close()
            os.remove(video_path)
            successful_downloads += 1

        except Exception as e:
            print(f"Unexpected error processing video {video_id}")
            traceback.print_exc()
            retry_dict[video_id] = [link, clips]
            failed_downloads += 1

    # Save retry JSON
    with open(retry_json_path, 'w') as f:
        json.dump(retry_dict, f, indent=4)

    print("\nDataset creation complete.")
    print(f"Successful downloads: {successful_downloads}")
    print(f"Failed downloads: {failed_downloads}")
    print(f"Retry JSON saved to: {retry_json_path}")

if __name__ == '__main__':
    make_dataset()
