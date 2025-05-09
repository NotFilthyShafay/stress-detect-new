### Stress Detection System

This repository contains a multimodal stress detection system that analyzes **facial expressions**, **speech prosody**, **speech sentiment**, and **body fidgeting** to detect stress in individuals from audio-visual data.

---

### Setup and Execution

Follow these steps to set up and run the stress detection system:

#### 1. Install Requirements
Install all required dependencies:
```bash
pip install -r requirements.txt
```

#### 2. Download the Dataset
Download the MAS (Multimodal Affective Signals) dataset:
```bash
python Stress-main/dataset/create_dataset.py
```
This will download video clips from YouTube as specified in the dataset configuration.

#### 3. Generate Dataset CSV Files
Process the downloaded videos using the multimodal detectors to create the features dataset:
```bash
python Stress-main/detectors/generate_dataset.py
```
This script:
- Processes each video using four detector nodes (face, audio, prosodic, and fidget).
- Extracts features for each modality.
- Combines the features and saves them to `MAS/dataset.csv`.

#### 4. Test Individual Nodes
To test how each individual detector node works, run:
```bash
python Stress-main/detectors/inference.py --video_path "path/to/video.mp4"
```
You can also run each node independently:
```bash
# Face emotion detection
python Stress-main/detectors/face.py "path/to/video.mp4"

# Audio sentiment analysis
python Stress-main/detectors/audio.py "path/to/video.mp4"

# Prosodic analysis
python Stress-main/detectors/prosodic.py "path/to/video.mp4"

# Fidget detection
python Stress-main/detectors/fidget.py "path/to/video.mp4"
```

#### 5. Train the Multimodal Fusion Model
Train the fusion model with your preferred configuration:
```bash
python Stress-main/detectors/train.py --model_type mlp --model_depth deep --batch_size 32 --epochs 100 --learning_rate 0.006 --hidden_size 256
```
Key configuration options:
- `--model_type`: Choose from `'mlp'`, `'rf'` (Random Forest), or `'both'`.
- `--model_depth`: Choose from `'shallow'`, `'medium'`, `'deep'`, `'very_deep'`.
- `--use_batch_norm`: Add this flag to use batch normalization.

Results will be saved to the `results` directory with a timestamp.

#### 6. Run Model Inference
After training, run inference on new data:
```bash
python Stress-main/detectors/model_inference.py --model_path "results/mlp_deep_XXXXXXXX_XXXXXX/mlp_model_full.pth" --video_path "path/to/video.mp4"
```
Replace `XXXXXXXX_XXXXXX` with the timestamp of your trained model.

---

### System Architecture

This stress detection system leverages four key affective modalities extracted from audio-visual data:

1. **Face Node**: Analyzes facial expressions for emotions like anger, disgust, fear, sadness, neutral, happiness, and surprise.
2. **Audio Node**: Performs sentiment analysis on speech, extracting emotions like sadness, anger, fear, joy, love, and surprise.
3. **Prosodic Node**: Analyzes speech characteristics like pitch, volume, and rhythm to detect stress indicators.
4. **Fidget Node**: Detects upper body fidgeting behaviors that may indicate stress.

---

### Fusion Methods

The repository provides three different fusion approaches:

1. **Voting-based Fusion**: Simple majority voting among modalities.
    ```bash
    python Dataset-main/detectors/voting_fusion.py --face --fidget --audio --prosodic path/to/video.mp4
    ```
2. **Deep Neural Network Fusion**: MLP-based fusion of multimodal features.
3. **Random Forest Fusion**: Tree-based fusion of multimodal features.

---

### Dataset

The MAS (Multimodal Affective Signals) dataset consists of 353 video clips sourced from YouTube channels featuring monologues of people discussing personal experiences. The clips are annotated with stress indicators in the following categories:
- **Facial stress**
- **Sentiment stress**
- **Vocal stress**
- **Fidgeting**

---

### Acknowledgments

This implementation is based on the architecture and dataset from:
```
@inproceedings{ghose2024integrating,
  title={Integrating Multimodal Affective Signals for Stress Detection from Audio-Visual Data},
  author={Ghose, Debasmita and Gitelson, Oz and Scassellati, Brian},
  booktitle={Proceedings of the 26th International Conference on Multimodal Interaction},
  pages={22--32},
  year={2024}
}
```

The original authors (Ghose, Gitelson, and Scassellati from Yale University) created the architecture and provided the dataset annotations and links. This implementation extends their work with modifications to improve performance.

For more information about the original project, see the paper or project page.
