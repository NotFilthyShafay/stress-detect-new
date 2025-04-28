"""
Various miscellaneous functions that help with optical flow calculations.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from utils.quadrilateral import Quadrilateral

def calc_optical_flow(prev: np.ndarray, curr: np.ndarray, quad: Quadrilateral) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the optical flow over the area of an image within a specified quadrilateral.

    Args:
        prev (np.ndarray): The previous frame.
        curr (np.ndarray): The current frame.
        quad (Quadrilateral): The quadrilateral defining the area to calculate flow over.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The magnitude and angle arrays of the calculated optical flow.
    """
    min_point, max_point = quad.get_min_max_points()

    if max_point[0] - min_point[0] < 1 or max_point[1] - min_point[1] < 1:
        return np.array([[0]]), np.array([[0]])

    try:
        prev_cropped = prev[min_point[1]:max_point[1], min_point[0]:max_point[0]]
        curr_cropped = curr[min_point[1]:max_point[1], min_point[0]:max_point[0]]

        prev_gray = cv2.cvtColor(prev_cropped, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_cropped, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        return magnitude, angle
    except Exception as e:
        print(f"Error calculating optical flow: {e}")
        return np.array([[0]]), np.array([[0]])

def optical_flow_rgb(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """
    Creates a visual representation of optical flow in RGB format.

    Args:
        magnitude (np.ndarray): The magnitude array of the optical flow.
        angle (np.ndarray): The angle array of the optical flow.

    Returns:
        np.ndarray: An array representing the RGB visualization of the optical flow.
    """
    mask_shape = (angle.shape[0], angle.shape[1], 3)
    mask = np.full(mask_shape, np.array([0, 255, 0], dtype='uint8'))
    
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    return rgb

def movement_occuring(mag: np.ndarray, threshold: float = 1.0) -> bool:
    """
    Determines whether the magnitude of optical flow in an area is sufficient to consider movement occurring.

    Args:
        mag (np.ndarray): The magnitude array of optical flow.
        threshold (float): The threshold above which to consider movement.

    Returns:
        bool: True if movement is occurring, False otherwise.
    """
    return np.mean(mag) > threshold
