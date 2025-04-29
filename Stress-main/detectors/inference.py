import os
import argparse
import concurrent.futures
import warnings
import logging
import sys
from typing import Dict, Any
from face import FaceNode
from audio import AudioNode
from prosodic import ProsodicNode
from fidget import FidgetNode

logger = logging.getLogger(__name__)

def process_node(node_type, node, video_path, silent):
    """Process a single node and return its features."""
    try:
        result = node.process(video_path)
        # Prefix features to match training
        return {f"{k}_{node_type}": v for k, v in result.items()}
    except Exception as e:
        logging.error(f"Error in {node_type} node: {str(e)}")
        return {}

def infer(video_path, silent=False):
    """Extract features from video using all nodes in parallel."""
    # Initialize nodes
    face_node = FaceNode()
    audio_node = AudioNode()
    prosodic_node = ProsodicNode()
    fidget_node = FidgetNode()
    
    all_features = {}
    
    # Define node processing tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_face = executor.submit(lambda: process_node("face", face_node, video_path, silent))
        future_audio = executor.submit(lambda: process_node("audio", audio_node, video_path, silent))
        future_prosodic = executor.submit(lambda: process_node("prosodic", prosodic_node, video_path, silent))
        future_fidget = executor.submit(lambda: process_node("fidget", fidget_node, video_path, silent))
        
        # Collect results
        for future, name in [(future_face, "face"), (future_audio, "audio"), 
                             (future_prosodic, "prosodic"), (future_fidget, "fidget")]:
            try:
                features = future.result()
                all_features.update(features)
                logging.info(f"Completed {name} node")
            except Exception as e:
                logging.error(f"Error in {name} node: {str(e)}")
    
    return all_features

def main():
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Suppress logging
    logging.basicConfig(level=logging.ERROR)
    
    # Optionally redirect stdout/stderr if needed
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    parser = argparse.ArgumentParser(description="Run inference on a video file")
    parser.add_argument("video_path", help="Path to the video file for inference")
    parser.add_argument("--output", help="Optional path to save results as JSON", default=None)
    parser.add_argument("--quiet", action="store_true", help="Suppress all output except final results")
    args = parser.parse_args()
    
    # If quiet mode is enabled, redirect stdout/stderr
    if args.quiet:
        class NullWriter:
            def write(self, x): pass
            def flush(self): pass
        
        sys.stdout = NullWriter()
        sys.stderr = NullWriter()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return
    
    # Run inference
    results = infer(args.video_path, silent=args.quiet)
    
    # Print summary of results (unless in quiet mode)
    if not args.quiet:
        print("\nInference Results:")
        print(f"Total features extracted: {len(results)}")
        print(results)
    # Save results if output path specified
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")
    
    # Restore stdout/stderr if they were redirected
    if args.quiet:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print(f"Processing complete for: {args.video_path}")
        
    return results

if __name__ == "__main__":
    main()