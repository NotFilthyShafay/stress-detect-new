import os
import argparse
import json
import time
import logging
from pathlib import Path
import glob
from model_inference import predict_stress

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("batch_stress_prediction")

def main():
    parser = argparse.ArgumentParser(description="Run stress prediction on multiple video files")
    parser.add_argument(
        "--clips_folder", 
        type=str, 
        default="MAS/clips/",
        help="Path to the folder containing video clips"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="results/mlp_very_deep_20250426_192327/best_model.pth", 
        help="Path to the saved model (.pth file)"
    )
    parser.add_argument(
        "--feature_order", 
        type=str, 
        default="processed_data/feature_order.json",
        help="Path to JSON file with feature order"
    )
    parser.add_argument(
        "--count", 
        type=int, 
        default=20,
        help="Number of most recent videos to process"
    )
    parser.add_argument(
        "--output_folder", 
        type=str, 
        default="results/predictions",
        help="Folder to save prediction results"
    )
    args = parser.parse_args()
    
    # Find all video files in the clips folder
    clips_path = os.path.abspath(args.clips_folder)
    if not os.path.exists(clips_path):
        logger.error(f"Clips folder not found: {clips_path}")
        return
    
    # Get all video files and sort by modification time (newest first)
    video_files = []
    for ext in ['mp4', 'avi', 'mov', 'mkv']:
        video_files.extend(glob.glob(os.path.join(clips_path, f"*.{ext}")))
    
    if not video_files:
        logger.error(f"No video files found in {clips_path}")
        return
    
    # Sort by modification time (newest first)
    video_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Take only the requested number of videos
    video_files = video_files[:args.count]
    logger.info(f"Processing {len(video_files)} videos from {clips_path}")
    
    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Process each video
    results = []
    total_start_time = time.time()
    
    for i, video_path in enumerate(video_files):
        video_name = os.path.basename(video_path)
        logger.info(f"[{i+1}/{len(video_files)}] Processing {video_name}")
        
        try:
            # Process the video
            start_time = time.time()
            result = predict_stress(
                video_path=video_path,
                model_path=args.model_path,
                feature_order_path=args.feature_order,
                silent=False
            )
            end_time = time.time()
            
            # Add video name and processing time to result
            result['video_name'] = video_name
            result['processing_time'] = end_time - start_time
            results.append(result)
            
            # Save individual result
            output_file = os.path.join(args.output_folder, f"{video_name}.json")
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"Prediction: {result['prediction']} (Confidence: {result['stress_confidence']:.4f})")
            logger.info(f"Completed in {result['processing_time']:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing {video_name}: {str(e)}")
    
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # Summarize results
    stress_count = sum(1 for r in results if r['prediction'] == 'stress')
    no_stress_count = len(results) - stress_count
    
    summary = {
        'total_videos': len(results),
        'stress_videos': stress_count,
        'no_stress_videos': no_stress_count,
        'stress_percentage': (stress_count / len(results)) * 100 if results else 0,
        'total_processing_time': total_time,
        'average_processing_time': total_time / len(results) if results else 0,
        'processed_videos': [r['video_name'] for r in results]
    }
    
    # Save summary
    summary_file = os.path.join(args.output_folder, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nProcessed {len(results)} videos in {total_time:.2f} seconds")
    logger.info(f"Stress detected: {stress_count} videos ({summary['stress_percentage']:.1f}%)")
    logger.info(f"Results saved to {args.output_folder}")

if __name__ == "__main__":
    main()