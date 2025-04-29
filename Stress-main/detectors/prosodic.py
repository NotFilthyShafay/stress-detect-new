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
import tempfile
from pathlib import Path
import torchaudio

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
        """
        try:
            print(f"Loading audio with torchaudio: {audio_file}")
            waveform, sample_rate = torchaudio.load(audio_file)
            if sample_rate != 16000:
                print(f"Resampling from {sample_rate} to 16000 Hz")
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            waveform = waveform.squeeze(0)  # [time]
            batch_waveform = waveform.unsqueeze(0).to(self.device)  # [1, time]
            out_prob, *_ = self.classifier.classify_batch(batch_waveform)
            emo_mapping = {'neu': 0, 'hap': 2, 'sad': 3, 'ang': 1}
            probabilities = out_prob[0] if len(out_prob.shape) == 2 else out_prob[0][0]
            ret_dic = {emo: probabilities[emo_mapping[emo]].item() for emo in emo_mapping}
            self.memory.append(ret_dic)
            if len(self.memory) > self.memory_length:
                self.memory.pop(0)
            return ret_dic
        except Exception as e:
            print(f"Error in classify_emotion: {e}")
            return {'neu': 0.25, 'hap': 0.25, 'sad': 0.25, 'ang': 0.25}

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
        if audio_path[-4:] != '.wav':
            print('No audio file detected. Trying to extract audio from video file')

            # Use the system temp directory for the temp wav file
            temp_wav = os.path.join(tempfile.gettempdir(), "temp.wav")
            print(f"Creating temporary audio file: {temp_wav!r}")

            # Extract audio
            video = VideoFileClip(audio_path)
            audio = video.audio
            audio.write_audiofile(temp_wav)

            print(f"File exists check: {os.path.exists(temp_wav)}")
            print(f"Final path being used: {temp_wav!r}")

            # Resolve path using pathlib
            temp_wav = str(Path(temp_wav).resolve())
            print(f"Pathlib resolved temp_wav: {temp_wav!r}")

            # Extra fix: If the path still starts with 'C:' and not 'C:\\', fix it
            if os.name == 'nt' and temp_wav.startswith("C:") and not temp_wav.startswith("C:\\"):
                temp_wav = "C:\\" + temp_wav[2:]
                print(f"Fixed Windows path: {temp_wav!r}")

            # Use the temporary WAV file
            result = self.classify_emotion(temp_wav)
            return result
        else:
            return self.classify_emotion(audio_path)

    def process(self, video_path: str) -> Dict[str, float]:
        """
        Wrapper for the prosodic node to extract and format prosodic features.
        """
        features = self.analyze(video_path)
        mapping = {
            "neu": "neu_prosodic_prosodic",
            "hap": "hap_prosodic_prosodic",
            "sad": "sad_prosodic_prosodic",
            "ang": "ang_prosodic_prosodic"
        }
        return {v: float(features.get(k, 0.0)) for k, v in mapping.items()}

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description="Face Node")
    parser.add_argument('path', type=str, help='Path to the video file.')
    args = parser.parse_args()
    
    cprint('Initializing Prosodic Node...', 'green', attrs=['bold'])
    node = ProsodicNode()
    
    cprint('Analyzing Data...', 'green', attrs=['bold'])
    print(node.analyze(args.path))
    
    end = time.time()
    cprint(f'Analysis completed in {end - start:.2f} seconds.', 'green', attrs=['bold'])
