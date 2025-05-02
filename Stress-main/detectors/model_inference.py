import os
import argparse
import torch
import numpy as np
import json
import logging
from collections import OrderedDict
import time
import subprocess
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("stress_prediction")

# Import from our modules
from inference import infer
from face import FaceNode
from audio import AudioNode
from prosodic import ProsodicNode
from fidget import FidgetNode

# Neural Network model architecture (must match the saved model)
class MultimodalMLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size=256, num_classes=2, dropout_rate=0.3, depth='very_deep'):
        super(MultimodalMLP, self).__init__()
        
        layers = []
        
        # Input layer
        layers.append(torch.nn.Linear(input_size, hidden_size))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout_rate))
        
        if depth == 'very_deep':
            # Add five more hidden layers with decreasing sizes
            hidden_sizes = [hidden_size, hidden_size // 2, hidden_size // 2, hidden_size // 4, hidden_size // 4, hidden_size // 8]
            
            for i in range(1, len(hidden_sizes)):
                layers.append(torch.nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Dropout(dropout_rate))
            
            # Output layer
            layers.append(torch.nn.Linear(hidden_sizes[-1], num_classes))
        elif depth == 'deep':
            # Add three more hidden layers with decreasing sizes
            hidden_sizes = [hidden_size, hidden_size // 2, hidden_size // 4]
            
            for i in range(1, len(hidden_sizes)):
                layers.append(torch.nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Dropout(dropout_rate))
            
            # Output layer
            layers.append(torch.nn.Linear(hidden_sizes[-1], num_classes))
        else:
            # Simple architecture (as in original code)
            layers.append(torch.nn.Linear(hidden_size, hidden_size // 2))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(dropout_rate))
            layers.append(torch.nn.Linear(hidden_size // 2, num_classes))
        
        self.model = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def load_model(model_path, input_size=None, hidden_size=256, depth='very_deep'):
    """Load the trained model from the given path."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = MultimodalMLP(input_size, hidden_size, depth=depth)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Load state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # If loaded from newer PyTorch version, might need to remove 'module.' prefix
        if list(state_dict.keys())[0].startswith('module.'):
            logger.info("Detected 'module.' prefix in state dict keys, removing...")
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        logger.info(f"Model successfully loaded from {model_path}")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
    
    return model, device


def load_feature_order(feature_order_path):
    """Load feature order from JSON file with proper error handling."""
    feature_order = None
    
    if not feature_order_path:
        logger.warning("No feature order path provided, will use alphabetical ordering")
        return None
    
    if not os.path.exists(feature_order_path):
        logger.warning(f"Feature order file not found: {feature_order_path}")
        return None
    
    try:
        with open(feature_order_path, 'r') as f:
            feature_order = json.load(f)
        
        # Validate feature order is a list
        if not isinstance(feature_order, list):
            logger.error(f"Feature order file should contain a list, found {type(feature_order)}")
            return None
        
        if len(feature_order) == 0:
            logger.warning("Feature order list is empty")
            return None
        
        logger.info(f"Loaded feature order with {len(feature_order)} features")
        return feature_order
    
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in feature order file: {feature_order_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading feature order: {str(e)}")
        return None


def prepare_features(features, feature_order=None):
    """Prepare features for model inference with improved error handling."""
    # If feature_order is provided, use it to ensure correct order
    if feature_order:
        logger.info(f"Using provided feature order with {len(feature_order)} features")
        ordered_features = []
        missing_features = []
        
        for feat in feature_order:
            if feat in features:
                ordered_features.append(float(features[feat]))
            else:
                missing_features.append(feat)
                ordered_features.append(0.0)  # Default value for missing features
        
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features from input: {missing_features[:5]}...")
        
        logger.info(f"Prepared {len(ordered_features)} features using provided feature order")
        return ordered_features
    
    # Otherwise, sort features by key to ensure consistent ordering
    logger.info("No feature order provided, using alphabetical ordering")
    # Remove non-numeric keys like 'video_file' if present
    numeric_features = {}
    for k, v in features.items():
        if k != 'video_file' and k != 'label':
            try:
                numeric_features[k] = float(v)
            except (ValueError, TypeError):
                logger.warning(f"Skipping non-numeric feature: {k}={v}")
    
    # Sort by key to maintain consistent order
    ordered_dict = OrderedDict(sorted(numeric_features.items()))
    feature_values = list(ordered_dict.values())
    logger.info(f"Prepared {len(feature_values)} features using alphabetical ordering")
    logger.debug(f"Feature names: {list(ordered_dict.keys())}")
    
    return feature_values


def convert_video_ffmpeg(input_path):
    """Converts video to a standard MP4 format using ffmpeg."""
    output_path = None
    try:
        # Create a temporary output file path
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_f:
            output_path = temp_f.name

        # Basic ffmpeg command: convert video, copy video codec, convert audio to aac
        command = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'copy', '-c:a', 'aac', '-ar', '48000',
            output_path
        ]

        logger.info(f"Running ffmpeg command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            logger.error(f"ffmpeg conversion failed for {input_path}.")
            logger.error(f"ffmpeg stderr: {result.stderr}")
            # Clean up failed output file
            if output_path and os.path.exists(output_path):
                os.unlink(output_path)
            return None  # Indicate failure

        logger.info(f"Successfully converted {input_path} to {output_path}")
        return output_path  # Return path to the new MP4 file

    except Exception as e:
        logger.error(f"Error during ffmpeg conversion: {e}")
        if output_path and os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except:
                pass
        return None


def predict_stress(
    video_path, 
    model_path, 
    feature_order_path=None, 
    hidden_size=256, 
    depth='very_deep', 
    silent=False
):
    """
    Process a video and predict stress using the saved model.
    
    Args:
        video_path: Path to the video file
        model_path: Path to the saved model file (.pth)
        feature_order_path: Optional path to JSON file with feature order
        hidden_size: Hidden layer size for the model
        depth: Model architecture depth ('very_deep', 'deep', or 'shallow')
        silent: Whether to suppress output
        
    Returns:
        Dict with prediction results
    """
    if silent:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)
    
    # Check if video file exists
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Attempt to convert video using ffmpeg for compatibility
    logger.info("Attempting to convert video using ffmpeg for compatibility...")
    converted_video_path = convert_video_ffmpeg(video_path)
    processing_path = converted_video_path if converted_video_path else video_path  # Use converted if successful

    if processing_path != video_path:
        logger.info(f"Using converted video path for processing: {processing_path}")
    else:
        logger.warning(f"ffmpeg conversion failed or skipped, using original video path: {video_path}")

    # Load feature order
    feature_order = load_feature_order(feature_order_path)
    
    # Process video with nodes
    face_node = FaceNode()
    audio_node = AudioNode()
    prosodic_node = ProsodicNode()
    fidget_node = FidgetNode()

    node_outputs = {
        "face": face_node.process(processing_path),
        "audio": audio_node.process(processing_path),
        "prosodic": prosodic_node.process(processing_path),
        "fidget": fidget_node.process(processing_path)
    }

    # FLATTEN the nested structure
    features = {}
    for node_features in node_outputs.values():
        features.update(node_features)
    
    # Save the flattened features too
    with open("debug_flattened_features.json", "w") as f:
        json.dump(features, f, indent=2)

    # Now 'features' contains all node outputs in a flat structure
    # with keys like "angry_face", "sadness_audio_audio", etc.
    logger.info(f"Extracted {len(features)} raw features")
    
    # Prepare features for model
    feature_values = prepare_features(features, feature_order)
    
    # Load and apply the same scaler used during training
    import joblib
    scaler_path = os.path.join(os.path.dirname(feature_order_path), 'scaler.pkl')
    if os.path.exists(scaler_path):
        logger.info(f"Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
        # Reshape for scaler (expects 2D array)
        feature_values_scaled = scaler.transform([feature_values])[0]
        logger.info("Features scaled using saved scaler")
        
        # Optional debugging to see the difference
        if not silent:
            logger.info("Sample before scaling: " + 
                       str([f"{x:.4f}" for x in feature_values[:3]]))
            logger.info("Sample after scaling: " + 
                       str([f"{x:.4f}" for x in feature_values_scaled[:3]]))
        
        # Use scaled features
        feature_values = feature_values_scaled
    else:
        logger.warning(f"Scaler not found at {scaler_path}. Using unscaled features!")
    
    # Determine input size from features
    input_size = len(feature_values)
    if input_size == 0:
        logger.error("No valid features extracted from video")
        raise ValueError("No valid features extracted from video")
    
    # Load model
    logger.info(f"Loading model from {model_path}...")
    model, device = load_model(model_path, input_size, hidden_size, depth)
    
    # Convert features to tensor
    try:
        input_tensor = torch.FloatTensor([feature_values]).to(device)
    except Exception as e:
        logger.error(f"Error converting features to tensor: {str(e)}")
        raise
    
    # Get prediction
    try:
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise
    
    # Get results
    pred_class = prediction.item()
    stress_confidence = probabilities[0][1].item()  # Probability of stress class
    no_stress_confidence = probabilities[0][0].item()  # Probability of no stress class
    
    # Create result dictionary
    result = {
        'prediction': 'stress' if pred_class == 1 else 'no stress',
        'prediction_label': pred_class,
        'stress_confidence': stress_confidence,
        'no_stress_confidence': no_stress_confidence,
        'features_extracted': len(feature_values)
    }
    
    logger.info("\nPrediction Results:")
    logger.info(f"Prediction: {result['prediction']} (Label: {result['prediction_label']})")
    logger.info(f"Confidence: Stress={stress_confidence:.4f}, No Stress={no_stress_confidence:.4f}")
    logger.info(f"Features extracted: {len(feature_values)}")

    # Cleanup converted file
    if converted_video_path and os.path.exists(converted_video_path):
        try:
            os.unlink(converted_video_path)
            logger.info(f"Cleaned up temporary converted video: {converted_video_path}")
        except Exception as e:
            logger.warning(f"Could not delete temporary converted video: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Predict stress from video using a multimodal MLP model.")
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument("--model_path", required=True, help="Path to the trained model file (.pth).")
    parser.add_argument("--feature_order", help="Path to the JSON file specifying feature order.")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size of the MLP model.")
    parser.add_argument("--depth", default='very_deep', help="Depth configuration of the MLP model.")
    parser.add_argument("--quiet", action='store_true', help="Suppress detailed JSON output to stdout (only print final result).")

    args = parser.parse_args()

    start_time = time.time()

    try:
        prediction_result = predict_stress(
            args.video_path,
            args.model_path,
            args.feature_order,
            args.hidden_size,
            args.depth,
            silent=args.quiet
        )
        inference_time = time.time() - start_time

        print(f"RESULT_JSON:{json.dumps(prediction_result)}")

    except Exception as e:
        error_result = {
            "error": f"Model inference failed: {str(e)}",
            "prediction": "unknown",
            "stress_confidence": 0.5,
            "no_stress_confidence": 0.5
        }
        print(f"RESULT_JSON:{json.dumps(error_result)}")


if __name__ == "__main__":
    main()