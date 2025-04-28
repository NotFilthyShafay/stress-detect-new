from speechbrain.inference.interfaces import foreign_class
import speech_recognition as sr
import torch
import json
import moviepy.editor as mp
from typing import Dict, List, Tuple
from moviepy.editor import VideoFileClip
import os
import argparse
from termcolor import colored, cprint
import time

_CLASSIFIER_CACHE = {}

def get_cached_classifier(source, pymodule_file, classname, run_opts=None):
    """Helper function to load classifier with caching"""
    cache_key = f"{source}_{classname}"
    if cache_key not in _CLASSIFIER_CACHE:
        print(f"Loading classifier {source} (first time)")
        _CLASSIFIER_CACHE[cache_key] = foreign_class(
            source=source,
            pymodule_file=pymodule_file,
            classname=classname,
            run_opts=run_opts
        )
    return _CLASSIFIER_CACHE[cache_key]

class ProsodicNode:
    """
    A class for prosodic analysis, utilizing a pre-trained model to classify emotions based on audio data.
    This node maintains a memory of recent analyses and can compute average emotion scores over this memory.
    """

    def __init__(self, memory_length: float = float('inf'), record_length: int = 10):
        """
        Initializes the ProsodicNode with specified memory and record lengths.

        Args:
            memory_length (float): Number of analyses to remember.
            record_length (int): Duration for which audio is recorded (if applicable).
        """
        self.recognizer = sr.Recognizer()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"ProsodicNode using device: {self.device}")
        self.classifier = get_cached_classifier(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            run_opts={"device": self.device}
        )
        self.memory: List[Dict[str, float]] = []  # List to store emotion probabilities from analyses
        self.memory_length = memory_length

    def classify_emotion(self, audio_file: str) -> Dict[str, float]:
        """
        Classifies the emotion of the audio file using the pre-trained classifier.

        Args:
            audio_file (str): The path to the audio file to analyze.

        Returns:
            Dict[str, float]: A dictionary containing emotion labels as keys and their corresponding probabilities as values.
        """
        out_prob, _, _, _ = self.classifier.classify_file(audio_file)
        emo_mapping = {'neu': 0, 'hap': 2, 'sad': 3, 'ang': 1}
        ret_dic = {emo: out_prob[0][emo_mapping[emo]].item() for emo in emo_mapping}

        # Store the analysis result in memory
        self.memory.append(ret_dic)
        if len(self.memory) > self.memory_length:
            self.memory.pop(0)

        return ret_dic

    def extract_embeddings(self, audio_file: str) -> torch.Tensor:
        """
        Extracts embeddings from the audio file using the classifier.

        Args:
            audio_file (str): The path to the audio file to analyze.

        Returns:
            torch.Tensor: The extracted embeddings as a tensor.
        """
        waveform = self.classifier.load_audio(audio_file)
        batch = waveform.unsqueeze(0)  # Add a batch dimension
        rel_length = torch.tensor([1.0])  # Full length relative to the batch
        embeddings = self.classifier.encode_batch(batch, rel_length)
        return embeddings.squeeze(0)

    def clear_memory(self) -> None:
        """
        Clears the memory of past analyses.
        """
        self.memory = []

    def get_avg_emotions(self) -> Dict[str, float]:
        """
        Computes the average emotion scores over the stored analyses in memory.

        Returns:
            Dict[str, float]: A dictionary containing average scores for each emotion.
                              The keys are emotion labels (e.g., "neu", "hap") and the values are the averaged probabilities.
        """
        ret_dic: Dict[str, float] = {}
        for emotions in self.memory:
            for emotion, score in emotions.items():
                if emotion not in ret_dic:
                    ret_dic[emotion] = 0
                ret_dic[emotion] += score

        for emotion in ret_dic:
            ret_dic[emotion] /= len(self.memory)  # Average the scores

        return ret_dic

    def analyze(self, audio_path: str) -> Dict[str, float]:
        """
        Analyzes an audio file to classify emotions.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            Dict[str, float]: A dictionary of classified emotions and their probabilities.
        """
        if audio_path[-4:]!='.wav':
            print('no audio file detected. trying to extract audio from video file')
            video=VideoFileClip(audio_path)
            audio=video.audio
            audio_path=os.path.join(os.path.dirname(audio_path), 'temp.wav')
            audio.write_audiofile(audio_path)

        try:
            return self.classify_emotion(audio_path)
        except Exception as e:
            print(f"Error during analysis: {e}")
            return {}
    def process(self, video_path: str) -> Dict[str, float]:
        """
        Wrapper for the prosodic node to extract and format prosodic features.
        """
        features = self.analyze(video_path)
        return {f"{k}_prosodic": v for k, v in features.items()}


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description="Face Node")
    parser.add_argument('path', type=str, help='Path to the video file.')
    args = parser.parse_args()

    cprint('Initializing Prosodic Node...', 'green', attrs=['bold'])
    node=ProsodicNode()

    cprint('Analyzing Data...', 'green', attrs=['bold'])
    print(node.analyze(args.path))
    
    end = time.time()
    cprint(f'Analysis completed in {end - start:.2f} seconds.', 'green', attrs=['bold'])
