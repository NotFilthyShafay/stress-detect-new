"""
This file defines the `Person` class, which performs the necessary calculations to detect fidgeting.
Note: the use of this class is done automatically by the FidgetNode class, so most users should not need to access it directly.
"""

from utils.movement_utils import *
import math
from utils.quadrilateral import Quadrilateral
import numpy as np
from typing import List, Tuple, Optional, Dict

class Person:
    """
    A class representing a person in the context of fidget detection. It tracks the skeleton, limb movements,
    and potential fidgeting behavior based on optical flow and collision detection.
    """

    def __init__(self, body_optical_thresh: float = 0.7, hand_optical_thresh: float = 1.4, face_optical_thresh: float = 0.3) -> None:
        """
        Initializes a Person object with specified thresholds for detecting movements and fidgets.

        Args:
            body_optical_thresh (float): Threshold for body optical flow to consider movement as fidgeting.
            hand_optical_thresh (float): Threshold for hand optical flow to consider movement as fidgeting.
            face_optical_thresh (float): Threshold for face optical flow to consider movement as fidgeting.
        """
        self.skeleton: Optional[np.ndarray] = None  # Array of keypoints representing the person's skeleton
        self.limbs: Dict[str, Quadrilateral] = {}  # Dictionary mapping body part names to their bounding quadrilaterals
        self.left_hand: Optional[Quadrilateral] = None  # Quadrilateral representing the left hand
        self.right_hand: Optional[Quadrilateral] = None  # Quadrilateral representing the right hand
        self.face: Optional[Quadrilateral] = None  # Quadrilateral representing the face
        self.left_collisions: Optional[List[str]] = None  # List of body parts colliding with the left hand
        self.right_collisions: Optional[List[str]] = None  # List of body parts colliding with the right hand
        self.left_fidgets: Optional[List[str]] = None  # List of body parts involved in left-hand fidgeting
        self.right_fidgets: Optional[List[str]] = None  # List of body parts involved in right-hand fidgeting
        self.confidence_threshhold: float = 0.2  # Confidence threshold for considering keypoints
        self.body_optical_thresh = body_optical_thresh
        self.hand_optical_thresh = hand_optical_thresh
        self.face_optical_thresh = face_optical_thresh

    @staticmethod
    def joint_to_quad(point1: Tuple[float, float, float], point2: Tuple[float, float, float], width: int = -1) -> Optional[Quadrilateral]:
        """
        Creates a quadrilateral around two points representing a limb or body part.

        Args:
            point1 (Tuple[float, float, float]): The first point of the quad.
            point2 (Tuple[float, float, float]): The second point of the quad.
            width (int): The width of the quad. If set to -1, width will be auto-calculated as 1/3 of the length.

        Returns:
            Optional[Quadrilateral]: A quadrilateral containing point1 and point2, or None if confidence is 0.
        """
        if point1 is None or point2 is None or point1[2] == 0 or point2[2] == 0:
            return None

        x1, y1 = point1[1], point1[0]
        x2, y2 = point2[1], point2[0]

        if width == -1:
            limb_length = math.dist(point1, point2)
            width_fraction = 1 / 3
            max_width = 30
            width = min(max_width, limb_length * width_fraction)

        with np.errstate(divide='ignore', invalid='ignore'):
            sin_theta = (x2 - x1) / np.sqrt((x2 - x1) ** 2 + (y1 - y2) ** 2)
            cos_theta = (y1 - y2) / np.sqrt((x2 - x1) ** 2 + (y1 - y2) ** 2)
            sin_theta = np.nan_to_num(sin_theta)
            cos_theta = np.nan_to_num(cos_theta)

        xa, ya = int(x1 + width * cos_theta), int(y1 + width * sin_theta)
        xb, yb = int(x2 + width * cos_theta), int(y2 + width * sin_theta)
        xc, yc = int(x2 - width * cos_theta), int(y2 - width * sin_theta)
        xd, yd = int(x1 - width * cos_theta), int(y1 - width * sin_theta)

        return Quadrilateral((xa, ya), (xb, yb), (xc, yc), (xd, yd))

    @staticmethod
    def generate_hands(elbow: Tuple[float, float, float], wrist: Tuple[float, float, float]) -> Optional[Quadrilateral]:
        """
        Creates a quadrilateral for a hand based on the elbow and wrist locations.

        Args:
            elbow (Tuple[float, float, float]): The point at the elbow.
            wrist (Tuple[float, float, float]): The point at the wrist.

        Returns:
            Optional[Quadrilateral]: A quadrilateral approximating the bounds of a hand.
        """
        if elbow is None or wrist is None:
            return None
        forearm_fraction = 1 / 2
        width_to_length = 1
        forearm_length = math.dist(elbow[:2], wrist[:2])
        hand_end_midpoint = (wrist[0] + (wrist[0] - elbow[0]) * forearm_fraction,
                             wrist[1] + (wrist[1] - elbow[1]) * forearm_fraction,
                             min(wrist[2], elbow[2]))
        hand_start_midpoint = (wrist[0], wrist[1], wrist[2])

        return Person.joint_to_quad(hand_start_midpoint, hand_end_midpoint,
                                    width=forearm_length * forearm_fraction * width_to_length / 2)

    def get_joints(self, left: bool = True) -> Dict[str, Optional[Tuple[float, float, float]]]:
        """
        Extracts pose data from the skeleton in the MoveNet output format.

        Args:
            left (bool): If True, extracts data for the left side of the body.

        Returns:
            Dict[str, Optional[Tuple[float, float, float]]]: A dictionary mapping body part names to points.
        """
        joint = {}
        offset = 1 if not left else 0
        if self.skeleton is None or len(self.skeleton) == 0:
            joint = {part: None for part in ['shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle']}
        else:
            joint = {
                'shoulder': self.skeleton[5 + offset],
                'elbow': self.skeleton[7 + offset],
                'wrist': self.skeleton[9 + offset],
                'hip': self.skeleton[11 + offset],
                'knee': self.skeleton[13 + offset],
                'ankle': self.skeleton[15 + offset]
            }
        return joint

    def generate_face(self, x_tol=0.1, y_tol=0.2) -> Optional[Quadrilateral]:
        """
        Generates a quadrilateral approximating the person's face.

        Returns:
            Optional[Quadrilateral]: A quadrilateral representing the face, or None if not enough points are available.
        """
        if self.skeleton is None or len(self.skeleton) == 0:
            return None
        joints = [self.skeleton[x] for x in [0, 1, 2, 3, 4] if self.skeleton[x][2] != 0]
        if len(joints) < 2:
            return None

        joints = np.array(joints)
        confidence = min(joints[:, 2])

        x_tol = x_tol * (max(joints[:, 0]) - min(joints[:, 0]))
        y_tol = y_tol * (max(joints[:, 1]) - min(joints[:, 1]))

        max_point = (max(joints[:, 0]) + x_tol, max(joints[:, 1]) + y_tol, confidence)
        min_point = (min(joints[:, 0]) - x_tol, min(joints[:, 1]) - y_tol, confidence)
        length = math.dist(max_point, min_point)
        return Person.joint_to_quad(min_point, max_point, width=int(length / 2))

    def filter_skeleton(self, skeleton: np.ndarray) -> None:
        """
        Filters out low-confidence points from the raw pose estimation data.

        Args:
            skeleton (np.ndarray): Raw data from the pose estimator.

        Returns:
            None
        """
        self.skeleton = []
        for joint in skeleton:
            if joint[2] < self.confidence_threshhold:
                self.skeleton.append([0, 0, 0])
            else:
                self.skeleton.append(joint)
        self.skeleton = np.array(self.skeleton)

    def update_skeleton(self, skeleton: np.ndarray, face_x_tol=0.1, face_y_tol=0.2) -> None:
        """
        Updates the person's skeleton with new pose data, creating a dictionary of body part quadrilaterals.

        Args:
            skeleton (np.ndarray): Raw data from the pose estimator.

        Returns:
            None
        """
        self.filter_skeleton(skeleton)
        left = self.get_joints(True)
        right = self.get_joints(False)
        self.face = self.generate_face(face_x_tol, face_y_tol)

        self.left_hand = Person.generate_hands(left['elbow'], left['wrist'])
        self.right_hand = Person.generate_hands(right['elbow'], right['wrist'])

        self.limbs = {
            'luarm': Person.joint_to_quad(left['shoulder'], left['elbow']),
            'lfarm': Person.joint_to_quad(left['elbow'], left['wrist']),
            'ruarm': Person.joint_to_quad(right['shoulder'], right['elbow']),
            'rfarm': Person.joint_to_quad(right['elbow'], right['wrist']),
            'face': self.face,
            'luleg': Person.joint_to_quad(left['hip'], left['knee']),
            'ruleg': Person.joint_to_quad(right['hip'], right['knee']),
            'llleg': Person.joint_to_quad(left['knee'], left['ankle']),
            'rlleg': Person.joint_to_quad(right['knee'], right['ankle']),
            'lhand': self.left_hand,
            'rhand': self.right_hand
        }

    def check_collisions(self) -> Tuple[List[str], List[str]]:
        """
        Checks which body parts the hands are colliding with.

        Returns:
            Tuple[List[str], List[str]]: Lists of body part names colliding with the left and right hands, respectively.
        """
        left_checks = ['ruarm', 'rfarm', 'face', 'luleg', 'ruleg', 'rlleg', 'llleg', 'rhand']
        right_checks = ['luarm', 'lfarm', 'face', 'luleg', 'ruleg', 'rlleg', 'llleg', 'lhand']

        self.left_collisions = [x for x in left_checks if self.left_hand is not None and
                                self.left_hand.quadrilateral_intersection(self.limbs.get(x, None))]
        self.right_collisions = [x for x in right_checks if self.right_hand is not None and
                                 self.right_hand.quadrilateral_intersection(self.limbs.get(x, None))]

        return self.get_collisions()

    def get_collisions(self) -> Tuple[List[str], List[str]]:
        """
        Returns a list of all collisions happening on a person.

        Returns:
            Tuple[List[str], List[str]]: Lists of body parts involved in collisions with the left and right hands, respectively.
        """
        return self.left_collisions or [], self.right_collisions or []

    def get_optical_fidget_threshold(self, limb: str) -> float:
        """
        Returns the optical flow threshold above which a collision should be considered fidgeting.

        Args:
            limb (str): The body part being checked.

        Returns:
            float: The optical flow threshold for the specified limb.
        """
        if 'hand' in limb:
            return self.hand_optical_thresh
        elif 'face' in limb:
            return self.face_optical_thresh

        return self.body_optical_thresh

    def check_fidgets(self, clean_frame: np.ndarray, prev_frame: np.ndarray) -> Tuple[List[str], List[str]]:
        """
        Uses optical flow to check which collisions in the current frame should be considered fidgets.

        Args:
            clean_frame (np.ndarray): The current clean frame.
            prev_frame (np.ndarray): The previous clean frame.

        Returns:
            Tuple[List[str], List[str]]: Lists of body part names involved in left and right-hand fidgeting, respectively.
        """
        if self.left_collisions is None or self.right_collisions is None:
            self.check_collisions()

        self.left_fidgets = [x for x in self.left_collisions if
                             movement_occuring(calc_optical_flow(prev_frame, clean_frame, self.limbs[x])[0],
                                               self.get_optical_fidget_threshold(x))]
        self.right_fidgets = [x for x in self.right_collisions if
                              movement_occuring(calc_optical_flow(prev_frame, clean_frame, self.limbs[x])[0],
                                                self.get_optical_fidget_threshold(x))]

        return self.get_fidgets()

    def get_fidgets(self) -> Tuple[List[str], List[str]]:
        """
        Returns a list of all fidgets happening on a person.

        Returns:
            Tuple[List[str], List[str]]: Lists of body parts involved in left and right-hand fidgeting, respectively.
        """
        return self.left_fidgets or [], self.right_fidgets or []

    def get_face_rect(self) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """
        Returns the coordinates defining a rectangle around a person's face.

        Returns:
            Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]: The top-left and bottom-right corners of the face rectangle.
        """
        if self.face is None:
            return None, None
        min_point, max_point = self.face.get_min_max_points()
        tolerance = 0

        return (max(0, min_point[0] - tolerance), max(0, min_point[1] - tolerance)), (max_point[0] + tolerance, max_point[1] + tolerance)
