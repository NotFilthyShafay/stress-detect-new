"""
Helper script to download MoveNet models from TensorFlow Hub and save them locally.
Run this script once before using FidgetNode to ensure models are available.
"""

import os
import sys
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
import shutil
import argparse

def download_movenet_models(output_dir):
    """Download MoveNet models and save them to the specified directory."""
    # Create output directories
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear existing TF Hub cache to force new download
    temp_cache_dir = output_dir / "tf_hub_cache"
    if temp_cache_dir.exists():
        print(f"Removing existing cache at {temp_cache_dir}")
        shutil.rmtree(temp_cache_dir)
    os.makedirs(temp_cache_dir, exist_ok=True)
    
    # Set TF Hub cache directory
    original_cache_dir = os.environ.get("TFHUB_CACHE_DIR")
    os.environ["TFHUB_CACHE_DIR"] = str(temp_cache_dir)
    
    try:
        print("Downloading MoveNet models to TensorFlow Hub cache...")
        print("This may take a few minutes...")
        
        # Lightning model
        lightning_model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        print("Lightning model downloaded to TF Hub cache")
        
        # Thunder model
        thunder_model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        print("Thunder model downloaded to TF Hub cache")
        
        # Find where TF Hub actually stored the models
        cache_files = list(temp_cache_dir.glob("*"))
        print(f"Downloaded model files: {[f.name for f in cache_files]}")
        
        # Create two directories for the models
        lightning_dir = output_dir / "movenet_lightning"
        thunder_dir = output_dir / "movenet_thunder"
        os.makedirs(lightning_dir, exist_ok=True)
        os.makedirs(thunder_dir, exist_ok=True)
        
        # Since we can't directly save the models using the object's save method,
        # we'll just keep the TF Hub cache and point to it in the modified code
        
        print("\nModels downloaded successfully to TF Hub cache!")
        print(f"Cache directory: {temp_cache_dir}")
        print("\nTo use these models, update your fidget.py with:")
        print("\nos.environ['TFHUB_CACHE_DIR'] = r'" + str(temp_cache_dir) + "'")
        print("\nBefore the hub.load() calls.")
        
        return str(temp_cache_dir)
        
    except Exception as e:
        print(f"Error downloading models: {e}")
        raise

def create_fidget_patch_file(cache_dir):
    """Create a patch file to modify fidget.py to use the local cache"""
    patch_content = f"""
# Add this to the top of your file, after the imports
import os
os.environ['TFHUB_CACHE_DIR'] = r'{cache_dir}'
"""
    patch_file = Path("fidget_patch.py")
    with open(patch_file, "w") as f:
        f.write(patch_content)
    
    print(f"\nCreated patch file at {patch_file.absolute()}")
    print("Copy the content of this file to the top of your fidget.py file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MoveNet models for local use")
    parser.add_argument("--output_dir", type=str, default="./models", 
                        help="Directory to save the downloaded models")
    args = parser.parse_args()
    
    cache_dir = download_movenet_models(args.output_dir)
    create_fidget_patch_file(cache_dir)