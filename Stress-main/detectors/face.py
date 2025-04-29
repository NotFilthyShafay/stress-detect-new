"""
This file defines the `FaceNode` class, which is used for facial emotion recognition.
"""


import os
from fer import FER
from fer.utils import draw_annotations
import cv2
import tensorflow as tf
from typing import List, Tuple, Dict, Optional
import argparse
from termcolor import colored, cprint
import time
from fidget import FidgetNode
import numpy as np

# Configure GPU usage
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Configure GPU to use memory growth to avoid consuming all VRAM at once
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"GPU detected and configured: {physical_devices}")
    except Exception as e:
        print(f"Error configuring GPU: {e}")
else:
    print("No GPU found, using CPU")


class FaceNode:
    """
    A class for detecting faces and recognizing emotions in images using the FER (Facial Emotion Recognition) library.
    This node stores the emotions detected over a sequence of frames and can compute average emotion scores.
    """

    def __init__(self, memory_length: int = 10):
        """
        Initializes the FaceNode with a specified memory length for storing detected faces and their emotions.

        Args:
            memory_length (int): Number of frames of face data to remember.
        """
        # GPU detection and configuration for PyTorch
        import torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"FaceNode using device: {self.device}")
        
        # Initialize the detector
        self.detector = FER(mtcnn=True)
        # Store the memory length
        self.memory_length = memory_length
        self.faces_memory = []

    def recognize_face(self, frame: cv2.Mat) -> List[Dict[str, Dict[str, float]]]:
        """
        Runs FER on an image to identify faces and determine the emotions they are expressing.

        Args:
            frame (cv2.Mat): Image to analyze for faces.

        Returns:
            List[Dict[str, Dict[str, float]]]: A list of dictionaries containing detected faces and their emotion scores.
                                               Each dictionary represents a face with its bounding box and emotions.
                                               The 'emotions' key holds another dictionary with emotion labels and scores.
        """
        if frame is None or frame.size == 0:
            return []

        # Detect faces and their emotions in the frame
        faces = self.detector.detect_emotions(frame)
        
        # Append detected faces and emotions to memory
        self.faces_memory.append(faces)
        if len(self.faces_memory) > self.memory_length:
            self.faces_memory.pop(0)
        
        return faces

    def recognize_faces_batch(self, frames: List[cv2.Mat]) -> None:
        """
        Process multiple frames at once for better GPU utilization.
        
        Args:
            frames (List[cv2.Mat]): List of frames to process
        """
        for frame in frames:
            if frame is None or frame.size == 0:
                self.faces_memory.append([])
                continue
            
            # Detect faces and emotions in each frame
            faces = self.detector.detect_emotions(frame)
            self.faces_memory.append(faces)
            if len(self.faces_memory) > self.memory_length:
                self.faces_memory.pop(0)

    def get_avg_emotions(self) -> Dict[str, float]:
        """
        Computes the average emotion scores over the stored frames in the node's memory.

        Returns:
            Dict[str, float]: A dictionary containing each emotion and their average scores over the node's memory.
                              The dictionary keys are emotion labels (e.g., "happy", "sad"), and the values are averaged scores.
        """
        emo_dic: Dict[str, float] = {}
        mean_denominator = 0

        for frame in self.faces_memory:
            for face in frame:
                mean_denominator += 1
                for emotion, score in face['emotions'].items():
                    if emotion not in emo_dic:
                        emo_dic[emotion] = 0
                    emo_dic[emotion] += score

        # Calculate average scores
        for emotion in emo_dic:
            if mean_denominator > 0:
                emo_dic[emotion] /= mean_denominator
        
        return emo_dic

    def clear_memory(self) -> None:
        """
        Clears the node's memory of detected faces and emotions.
        """
        self.faces_memory = []

    def analyze(self, video_path: str, frame_sample_rate: int = 5) -> Dict[str, float]:
        """
        Analyzes a sequence of frames for facial emotions and calculates the percentage of frames with no detected faces.
        Uses batch processing for better GPU utilization.

        Args:
            video_path (str): Path to the video file.
            frame_sample_rate (int): Process every nth frame (default: 5 = process every 5th frame)

        Returns:
            Dict[str, float]: A dictionary of average emotion scores and the percentage of frames without detected faces.
        """
        self.clear_memory()
        start_time = time.time()
        
        video = cv2.VideoCapture(video_path)

        # Check if the video was opened successfully
        if not video.isOpened():
            print("Error: Could not open video.")
            return {}

        # Read frames at specified sample rate
        frames = []
        frame_count = 0
        while True:
            ret, frame = video.read()
            if not ret:
                break
            if frame_count % frame_sample_rate == 0:  # Only process every nth frame
                frames.append(frame)
            frame_count += 1

        # Release the video capture object
        video.release()

        if not frames:
            return {}
            
        # Process frames in batches for better GPU utilization
        batch_size = 16  # Adjust based on your GPU memory
        total_frames = len(frames)
        empty_frames = 0
        
        print(f"Processing {total_frames} frames in batches of {batch_size}...")
        
        for i in range(0, total_frames, batch_size):
            batch = frames[i:min(i+batch_size, total_frames)]
            
            try:
                self.recognize_faces_batch(batch)
                # Count empty frames in this batch
                for j in range(len(batch)):
                    idx = min(i+j, len(self.faces_memory)-1)
                    if not self.faces_memory[idx]:
                        empty_frames += 1
            except Exception as e:
                print(f"Error processing batch starting at frame {i}: {e}")
                empty_frames += len(batch)  # Count all as empty if processing failed
        
        avg_emotions = self.get_avg_emotions()
        off_screen_percent = empty_frames / total_frames if total_frames else 0
        
        end_time = time.time()
        print(f"Analysis completed in {end_time - start_time:.2f} seconds")
        
        # Return as dictionary instead of tuple
        result = {
            "angry_face": avg_emotions.get("angry", 0),
            "disgust_face": avg_emotions.get("disgust", 0),
            "fear_face": avg_emotions.get("fear", 0),
            "happy_face": avg_emotions.get("happy", 0),
            "sad_face": avg_emotions.get("sad", 0),
            "surprise_face": avg_emotions.get("surprise", 0),
            "neutral_face": avg_emotions.get("neutral", 0),
            "face_offscreen_ratio_face": off_screen_percent
        }
        return result

    def process(self, video_path: str) -> Dict[str, float]:
        """
        Wrapper to conform to the expected interface. Runs analysis and formats results.

        Args:
            video_path (str): Path to the video file.

        Returns:
            Dict[str, float]: Flattened dictionary of features.
        """
        return self.analyze(video_path)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Face Node")
    parser.add_argument('path', type=str, help='Path to the video file.')
    args=parser.parse_args()

    cprint('Initializing Face Node...', 'green', attrs=['bold'])
    node=FaceNode()

    cprint('Analyzing Data...', 'green', attrs=['bold'])
    print(node.analyze(args.path))