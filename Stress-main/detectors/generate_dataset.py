import os
import json
import csv
import concurrent.futures
from pathlib import Path
from face import FaceNode
from audio import AudioNode
from prosodic import ProsodicNode
from fidget import FidgetNode


VIDEO_DIR = "MAS/clips"
ANNOTATIONS_PATH = "MAS/annotations.json"
OUTPUT_CSV = "MAS/dataset.csv"

# Label function: no stress = 0, everything else = 1
def get_binary_label(labels):
    return 0 if labels == ["no stress"] else 1

# Each node instance
face_node = FaceNode()
audio_node = AudioNode()
prosodic_node = ProsodicNode()
fidget_node = FidgetNode()

# Wrapper to run one node's process safely
def run_node(node, video_path):
    try:
        return node.process(video_path)
    except Exception as e:
        print(f"Error in {node.__class__.__name__} for {video_path}: {e}")
        return {}

# Process one video file using all nodes in parallel
def process_video(video_file, label):
    video_path = os.path.join(VIDEO_DIR, video_file)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            "face": executor.submit(run_node, face_node, video_path),
            "audio": executor.submit(run_node, audio_node, video_path),
            "prosodic": executor.submit(run_node, prosodic_node, video_path),
            "fidget": executor.submit(run_node, fidget_node, video_path),
        }
        results = {name: future.result() for name, future in futures.items()}

    combined_features = {}
    for name, features in results.items():
        for k, v in features.items():
            combined_features[f"{k}_{name}"] = v

    combined_features["label"] = label
    combined_features["video_file"] = video_file
    return combined_features

def main():
    # Load annotations
    with open(ANNOTATIONS_PATH, "r") as f:
        annotations = json.load(f)
        
    dataset = []
    for annot_key, label_list in annotations.items():
        video_file = annot_key + ".mp4"  # match actual filename (e.g., "XYZ.mp4.mp4")
        label = get_binary_label(label_list)
        print(f"Processing {video_file}...")

        features = process_video(video_file, label)
        dataset.append(features)


    # Write to CSV
    if dataset:
        fieldnames = list(dataset[0].keys())
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dataset)

    print(f"\n Dataset saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
