import sys
import subprocess
import json
import os

def run_model():
    """
    Simple wrapper that runs the inference model and ensures output is properly captured
    """
    # Get all command line arguments except the script name
    model_args = sys.argv[1:]
    
    # Construct the command to run the original model script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_script = os.path.join(script_dir, "model_inference.py")
    
    # Run the model as a subprocess
    result = subprocess.run([sys.executable, model_script] + model_args, 
                            capture_output=True, text=True)
    
    # Check if the process ran successfully
    if result.returncode != 0:
        error_data = {
            "error": f"Model inference failed: {result.stderr}",
            "prediction": "unknown",
            "stress_confidence": 0.5,
            "no_stress_confidence": 0.5
        }
        print(f"RESULT_JSON: {json.dumps(error_data)}")
        sys.exit(1)
    
    # Look for JSON output in the model's stdout
    for line in result.stdout.split('\n'):
        if line.strip().startswith('{') and '"prediction"' in line:
            try:
                data = json.loads(line.strip())
                if "prediction" in data:
                    # Found valid JSON, print it with the marker
                    print(f"RESULT_JSON: {json.dumps(data)}")
                    sys.exit(0)
            except:
                pass
    
    # If we reach here, no valid JSON was found
    default_data = {
        "prediction": "unknown",
        "stress_confidence": 0.5,
        "no_stress_confidence": 0.5
    }
    print(f"RESULT_JSON: {json.dumps(default_data)}")

if __name__ == "__main__":
    run_model()