'''
This file defines the `AudioNode` class, which is used for audio sentiment analysis.
'''

import speech_recognition as sr
from transformers import pipeline, DistilBertModel, DistilBertTokenizer
import torch
from typing import List, Tuple, Dict, Optional
from moviepy.editor import VideoFileClip
import os
import argparse
from termcolor import colored, cprint


class AudioNode:
    """
    A class for audio sentiment analysis, including speech-to-text (STT) and emotion detection.
    It utilizes the DistilBert model for emotion classification and Whisper for STT.
    """

    def __init__(self, memory_length: int = 6, record_length: float = float('inf')):
        """
        Initializes the AudioNode with specified memory and record length.

        Args:
            memory_length (int): Number of audio analysis records to remember.
            record_length (float): Duration of each recorded audio chunk.
        """
        self.recognizer = sr.Recognizer()
        self.sentiment_pipeline = pipeline(
            "text-classification",
             model="bhadresh-savani/distilbert-base-uncased-emotion",
             return_all_scores=True,
             device=0 if torch.cuda.is_available() else -1
        )
        self.tokenizer = DistilBertTokenizer.from_pretrained('bhadresh-savani/distilbert-base-uncased-emotion')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DistilBertModel.from_pretrained('bhadresh-savani/distilbert-base-uncased-emotion').to(device=self.device)
        
        self.phrases: List[Tuple[str, Dict[str, float]]] = []  # List to store transcribed text and emotion scores
        self.memory_length = memory_length
        self.record_length = record_length

    def sentiment_analysis(self, text: str) -> Dict[str, float]:
        """
        Performs sentiment analysis on the given text using a pre-trained model.

        Args:
            text (str): Input text to analyze.

        Returns:
            Dict[str, float]: A dictionary of emotions and their corresponding scores.
                              The keys are emotion labels (e.g., "joy", "anger") and the values are confidence scores.
        """
        sentiment = self.sentiment_pipeline(text)
        return {x['label']: x['score'] for x in sentiment[0]}

    def extract_embeddings(self, text: str) -> torch.Tensor:
        """
        Extracts embeddings for the input text using the DistilBert model.

        Args:
            text (str): The input text to extract embeddings for.

        Returns:
            torch.Tensor: The extracted embeddings as a tensor.
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state
        pooled_embeddings = embeddings.mean(dim=1)
        return pooled_embeddings

    def analyse_audio(self, audio_data: sr.AudioData) -> Tuple[str, Dict[str, float]]:
        """
        Analyzes the given audio data to perform STT and sentiment analysis.

        Args:
            audio_data (sr.AudioData): The audio data to analyze.

        Returns:
            Tuple[str, Dict[str, float]]: The transcribed text and a dictionary of emotion scores.
                                          The emotion scores dictionary contains labels as keys and confidence scores as values.
        """
        print('STT processing')
        if audio_data is not None:
            # Convert audio to text using Whisper model with GPU
            text = self.recognizer.recognize_whisper(
                audio_data, 
                model='small.en',
                language="en"  # Add language hint for speedup
            )
        else:
            text = ''
        print('Sentiment analysis')
        # Perform sentiment analysis on the transcribed text
        emotions = self.sentiment_analysis(text)

        # Store the transcribed text and emotion scores
        self.phrases.append((text, emotions))
        if len(self.phrases) > self.memory_length:
            self.phrases.pop(0)

        return text, emotions

    def get_audio(self, source: sr.AudioFile) -> Optional[sr.AudioData]:
        """
        Records audio data from a given source.

        Args:
            source (sr.AudioFile): The source to record from.

        Returns:
            Optional[sr.AudioData]: The recorded audio data.
        """
        print('Recording audio')
        if source is None:
            return None
        return self.recognizer.record(source, duration=self.record_length)

    def set_record_length(self, length: float) -> None:
        """
        Sets the recording length for audio chunks.

        Args:
            length (float): The new length of audio chunks.
        """
        self.record_length = length

    def set_mem_length(self, length: int) -> None:
        """
        Sets the memory length, which determines how many past analyses to remember.
        Discards old data if the new length is shorter.

        Args:
            length (int): The new length of memory.
        """
        self.memory_length = length
        while len(self.phrases) > self.memory_length:
            self.phrases.pop(0)

    def get_avg_emotions(self, weighted: bool = False) -> Dict[str, float]:
        """
        Computes the average emotion scores over a windowed chunk of audio.

        Args:
            weighted (bool): Whether to weight each score by the length of the text.

        Returns:
            Dict[str, float]: A dictionary of average emotions and scores.
                              The dictionary keys are emotion labels (e.g., "joy", "anger"),
                              and the values are averaged confidence scores.
        """
        emo_dic: Dict[str, float] = {}
        mean_denominator = 0
        for phrase in self.phrases:
            weight = 1
            if weighted:
                weight = len(phrase[0])  # Weight based on text length if weighted=True
            mean_denominator += weight

            for emotion in phrase[1]:
                if emotion not in emo_dic:
                    emo_dic[emotion] = 0
                emo_dic[emotion] += phrase[1][emotion] * weight

        for emotion in emo_dic:
            emo_dic[emotion] /= mean_denominator

        return emo_dic

    def clear_memory(self) -> None:
        """
        Clears the node's memory of past analyses.
        """
        self.phrases = []
    
    def analyze(self, audio_path: str) -> Tuple[Dict[str, float], float, str]:
        """
        Performs analysis on an audio file, including STT and sentiment analysis.

        Args:
            audio_path (str): Path to the audio file.

        Returns:
            Tuple[Dict[str, float], float, str]: 
                - A dictionary of average emotion scores (keys are emotion labels, values are scores),
                - Speech rate (words per second),
                - Transcribed text.
        """
        self.clear_memory()

        if audio_path[-4:]!='.wav':
            temp_wav = os.path.join(os.path.dirname(audio_path), 'temp.wav')
            # Check if already extracted (might have been done by prosodic node)
            if not os.path.exists(temp_wav):
                print('Extracting audio from video file')
                video=VideoFileClip(audio_path)
                audio=video.audio
                audio.write_audiofile(temp_wav, verbose=False, logger=None)  # Reduce logging
            audio_path = temp_wav

        with sr.AudioFile(audio_path) as source:
            # Only record what's needed
            duration = min(source.DURATION, self.record_length)
            audio = self.recognizer.record(source, duration=duration)
            text, _ = self.analyse_audio(audio)
            num_words = len(text.split())
            speech_rate = num_words / duration  # Calculate speech rate

        return self.get_avg_emotions(), speech_rate, text

    def process(self, video_path: str) -> Dict[str, float]:
        """
        Wrapper for the audio node to extract and format audio features.
        
        Returns:
            A dictionary of features with keys suffixed by '_audio'.
        """
        avg_emotions, speech_rate, _ = self.analyze(video_path)
        expected = [
            "sadness_audio_audio", "joy_audio_audio", "love_audio_audio", "anger_audio_audio",
            "fear_audio_audio", "surprise_audio_audio", "speech_rate_audio_audio"
        ]
        # Map your emotion keys to the expected ones if needed
        emotion_map = {
            "sadness": "sadness_audio_audio",
            "joy": "joy_audio_audio",
            "love": "love_audio_audio",
            "anger": "anger_audio_audio",
            "fear": "fear_audio_audio",
            "surprise": "surprise_audio_audio"
        }
        features = {emotion_map[k]: avg_emotions.get(k, 0.0) for k in emotion_map}
        features["speech_rate_audio_audio"] = speech_rate
        # Ensure all expected keys are present
        return {k: float(features.get(k, 0.0)) for k in expected}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Face Node")
    parser.add_argument('path', type=str, help='Path to the video file.')
    args = parser.parse_args()

    cprint('Initializing Audio Node...', 'green', attrs=['bold'])
    node = AudioNode()

    cprint('Analyzing Data...', 'green', attrs=['bold'])
    print(node.analyze(args.path))
