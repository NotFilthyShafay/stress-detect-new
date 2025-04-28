import os
import argparse
import concurrent.futures
import warnings
import logging
import sys
from face import FaceNode
from audio import AudioNode
from prosodic import ProsodicNode
from fidget import FidgetNode

def run_node(node, video_path, silent=False):
    """Run a single node's processing safely with error handling."""
    try:
        return node.process(video_path)
    except Exception as e:
        if not silent:
            print(f"Error in {node.__class__.__name__} for {video_path}: {e}")
        return {}

def infer(video_path, silent=False):
    """Process a video using all nodes in parallel and return combined features."""
    # Instantiate all nodes
    face_node = FaceNode()
    audio_node = AudioNode()
    prosodic_node = ProsodicNode()
    fidget_node = FidgetNode()
    
    if not silent:
        print(f"Processing video: {video_path}")
    
    # Run all nodes in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            "face": executor.submit(run_node, face_node, video_path, silent),
            "audio": executor.submit(run_node, audio_node, video_path, silent),
            "prosodic": executor.submit(run_node, prosodic_node, video_path, silent),
            "fidget": executor.submit(run_node, fidget_node, video_path, silent),
        }
        results = {name: future.result() for name, future in futures.items()}

    # Combine all features with appropriate prefixes
    combined_features = {}
    for name, features in results.items():
        for k, v in features.items():
            combined_features[f"{k}_{name}"] = v
            
    return combined_features

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