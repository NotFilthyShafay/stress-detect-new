"""
This file defines the `FusionNode` class, which combines the outputs of the various unimodal nodes to determine whether to engage or not.
"""
import json

import moviepy.editor as mp
import cv2
import speech_recognition as sr
from fidget import MatrixRecorder
from audio import AudioNode
from face import FaceNode
from fidget import FidgetNode
from prosodic import ProsodicNode
from termcolor import colored, cprint
import argparse

import warnings
warnings.filterwarnings("ignore")


class VotingFusionNode:

    def __init__(self, audio_node=None, face_node=None, fidget_node=None, prosodic_node=None, fidget_thresh=0.5):
        # initialize thresholds and relevant emotion lists
        self.audio_engage_emotes=['sadness','anger', 'fear']
        self.audio_disengage_emotes=['joy', 'love', 'surprise']

        self.face_engage_emotes=['angry', 'disgust', 'fear', 'sad']
        self.face_disengage_emotes=['happy', 'surprise']

        self.prosodic_engage_emotes=['sad', 'ang']
        self.prosodic_disengage_emotes=['hap']

        self.audio_high_prob_threshold = .5
        self.face_high_prob_threshold = 4/7
        self.prosodic_high_prob_threshold = .5

        self.audio_med_prob_threshold = .4
        self.face_med_prob_threshold = .4
        self.prosodic_med_prob_threshold = .4

        self.fidget_threshold = fidget_thresh

        self.audio_high_prob_disengage_threshold = .5
        self.face_high_prob_disengage_threshold = 2/7
        self.prosodic_high_prob_disengage_threshold = 1/4


        # store unimodal nodes
        self.audio_node=audio_node
        self.face_node=face_node
        self.fidget_node=fidget_node
        self.prosodic_node=prosodic_node

    def combination_fusion(self, audio_dic, face_dic, fidget_percent, prosodic_dic):
        '''
        fuses together audio, fidget, and facial data using simple boolean logic.
        uses all emotions in the face and audio dicts
        Args:
            audio_dic: dic of audio emotions
            face_dic: dic of facial emotions
            fidget_percent: float between 0 and 1 which represents the % of the time the person in the video was fidgeting

        Returns: bool representing whether to engage

        '''

        # fill placeholder if dic is empty
        if len(audio_dic) == 0:
            audio_dic['placeholder'] = 0
        if len(face_dic) == 0:
            face_dic['placeholder'] = 0
        if len(prosodic_dic) == 0:
            prosodic_dic['placeholder']=0

        # calculate votes
        audio_engage_sum=sum([audio_dic[e] if e in audio_dic else 0 for e in self.audio_engage_emotes])
        face_engage_sum=sum([face_dic[e] if e in face_dic else 0 for e in self.face_engage_emotes])
        prosodic_engage_sum = sum([prosodic_dic[e] if e in prosodic_dic else 0 for e in self.prosodic_engage_emotes])

        audio_high_prob_engage =audio_engage_sum >self.audio_high_prob_threshold
        face_high_prob_engage = face_engage_sum>self.face_high_prob_threshold
        prosodic_high_prob_engage = prosodic_engage_sum>self.prosodic_high_prob_threshold

        audio_disengage_sum = sum([audio_dic[e] if e in audio_dic else 0 for e in self.audio_disengage_emotes])
        face_disengage_sum = sum([face_dic[e] if e in face_dic else 0 for e in self.face_disengage_emotes])
        prosodic_disengage_sum = sum([prosodic_dic[e] if e in prosodic_dic else 0 for e in self.prosodic_disengage_emotes])

        audio_high_prob_disengage = audio_disengage_sum >self.audio_high_prob_disengage_threshold
        face_high_prob_disengage = face_disengage_sum>self.face_high_prob_disengage_threshold
        prosodic_high_prob_disengage = prosodic_disengage_sum>self.prosodic_high_prob_disengage_threshold

        fidget_engage = fidget_percent > self.fidget_threshold

        # do voting
        engage_votes=0

        for vote in [audio_high_prob_engage, face_high_prob_engage, prosodic_high_prob_engage, fidget_engage]:
            if vote: engage_votes+=1

        for vote in [audio_high_prob_disengage,face_high_prob_disengage, prosodic_high_prob_disengage]:
            if vote: engage_votes -=1

        return engage_votes>0

    def make_prediction(self, path, clear_node_mem=True):
        face_dic, audio_dic, fidget_percent, prosodic_dic, avg_fidget_matrix={}, {}, 0, {}, None

        stt_text=''
        if clear_node_mem:
            if self.audio_node: self.audio_node.clear_memory()
            if self.face_node: self.face_node.clear_memory()
            if self.fidget_node: self.fidget_node.clear_memory()
            if self.prosodic_node: self.prosodic_node.clear_memory()

        if self.audio_node or self.prosodic_node:
            clip = mp.VideoFileClip(path)
            audio_file = 'converted.wav'

            try:
                clip.audio.write_audiofile(audio_file, verbose=False, logger=None)
                if self.audio_node:
                    with sr.AudioFile(audio_file) as source:
                        # print('analyzing audio')
                        audio = self.audio_node.get_audio(source)
                        stt_text = self.audio_node.analyse_audio(audio)[0]
                    audio_dic = self.audio_node.get_avg_emotions()

                if self.prosodic_node:
                    self.prosodic_node.classify_emotion(audio_file)
                    prosodic_dic=self.prosodic_node.get_avg_emotions()

            except AttributeError:
                pass

        if self.face_node or self.fidget_node:
            cap = cv2.VideoCapture(path)
            ret, prev_frame = cap.read()

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            curr_frame = 0

            recorder=MatrixRecorder()
            if self.fidget_node:
                self.fidget_node.set_recorder(recorder)

            while True:
                print('video progress: ' + str(curr_frame / frame_count), end='')

                ret, frame = cap.read()
                if not ret:
                    break
                clean_frame = frame.copy()
                person = self.fidget_node.detect_fidget(frame, prev_frame)[0]

                if self.face_node:
                    face_min, face_max = person.get_face_rect()
                    if face_min is not None and face_max is not None and 0<= face_min[1] < face_max[1] and 0 <= face_min[0] < face_max[0]:
                        # print('running face detection')
                        faces,frame, emotion_embedings = self.face_node.recognize_face(frame[face_min[1]:face_max[1], face_min[0]:face_max[0]])
                        # print(emotion_embedings.shape)
                prev_frame = clean_frame
                curr_frame += 1

                print('\r', end='')

                # cv2.imshow('facial detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break

            if self.face_node:
                face_dic = self.face_node.get_avg_emotions()
            if self.fidget_node:
                fidget_percent = self.fidget_node.get_fidget_percentage()
                avg_fidget_matrix=recorder.get_memory_avg()
                print("Average Fidget Matrix: ", avg_fidget_matrix)

        prediction = self.combination_fusion(audio_dic, face_dic, fidget_percent, prosodic_dic)
        print('')

        return prediction, {'audio':audio_dic, 'face':face_dic, 'fidget':fidget_percent, 'prosodic':prosodic_dic}

    def simulate_prediction(self, outputs, audio=False, face=False, fidget=False, prosodic=False):
        audio_dic, face_dic, fidget_percent, prosodic_dic = {}, {}, -1, {}
        if audio:
            audio_dic = outputs['audio_pred']

        if face:
            face_dic = outputs['face_pred']

        if fidget:
            fidget_percent = outputs['fidget_pred']

        if prosodic:
            prosodic_dic = outputs['prosodic_pred']

        prediction=self.combination_fusion(audio_dic, face_dic, fidget_percent, prosodic_dic)

        return prediction

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Process some flags and paths.")

    # Boolean flags
    parser.add_argument('--face', action='store_true', help='Enable face processing')
    parser.add_argument('--fidget', action='store_true', help='Enable fidget processing')
    parser.add_argument('--audio', action='store_true', help='Enable audio processing')
    parser.add_argument('--prosodic', action='store_true', help='Enable prosodic processing')

    # Path flag
    parser.add_argument('--path', type=str, required=True, help='Path to the data')

    # Optional save flag with default None
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save the results (optional)')

    args = parser.parse_args()

    cprint('Initializing Nodes...', 'green', attrs=['bold'])

    face, fidget, audio, prosodic=None, None, None, None
    if args.face:
        face=FaceNode()
    if args.fidget:
        fidget=FidgetNode()
    if args.audio:
        audio=AudioNode()
    if args.prosodic:
        prosodic=ProsodicNode()

    fusion_node=VotingFusionNode(audio, face, fidget, prosodic)

    cprint('Analyzing Data...', 'green', attrs=['bold'])
    prediction, unimodal_data= fusion_node.make_prediction(args.path)
    print(prediction)
    if args.save:
        with open(args.save, 'w') as f:
            json.dump([prediction, unimodal_data], f)