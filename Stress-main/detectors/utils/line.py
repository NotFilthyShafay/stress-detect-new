"""
This file handles some of the basic geometry calculations for dealing with lines.
"""

import numpy as np
from typing import Tuple, Optional

class Line:
    """
    A class representing a line segment defined by two points. Provides methods for calculating
    slope, extending the segment, and finding intersections with other line segments.
    """

    def __init__(self, point1: Tuple[float, float], point2: Tuple[float, float]):
        """
        Initializes a Line object with two endpoints.

        Args:
            point1 (Tuple[float, float]): The first endpoint of the line segment.
            point2 (Tuple[float, float]): The second endpoint of the line segment.
        """
        self.x1, self.y1, self.x2, self.y2 = point1[0], point1[1], point2[0], point2[1]

    def slope(self) -> float:
        """
        Calculates the slope of the line segment.

        Returns:
            float: The slope of the line. Returns a large number if the line is vertical to avoid division by zero.
        """
        if (self.x2 - self.x1) == 0:
            return 1e15  # Considered as a vertical line with an undefined slope
        else:
            m = (self.y2 - self.y1) / (self.x2 - self.x1)
            return m

    def x_bound(self, other: 'Line') -> Tuple[float, float]:
        """
        Determines the x-axis overlap between this line and another line.

        Args:
            other (Line): The other line to compare with.

        Returns:
            Tuple[float, float]: The lower and upper x-bounds where the two lines overlap.
        """
        a, b = self, other
        x_lower = max(min(a.x1, a.x2), min(b.x1, b.x2))
        x_upper = min(max(a.x1, a.x2), max(b.x1, b.x2))
        return x_lower, x_upper

    def y_bound(self, other: 'Line') -> Tuple[float, float]:
        """
        Determines the y-axis overlap between this line and another line.

        Args:
            other (Line): The other line to compare with.

        Returns:
            Tuple[float, float]: The lower and upper y-bounds where the two lines overlap.
        """
        a, b = self, other
        y_lower = max(min(a.y1, a.y2), min(b.y1, b.y2))
        y_upper = min(max(a.y1, a.y2), max(b.y1, b.y2))
        return y_lower, y_upper

    def extend_segment(self, length: float) -> 'Line':
        """
        Extends the line segment by a given length in both directions.

        Args:
            length (float): The amount to extend the line segment.

        Returns:
            Line: A new line object representing the extended line segment.
        """
        cos_theta = (self.x2 - self.x1) / np.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)
        sin_theta = (self.y2 - self.y1) / np.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)
        x1_ = self.x1 - length * cos_theta
        y1_ = self.y1 - length * sin_theta
        x2_ = self.x2 + length * cos_theta
        y2_ = self.y2 + length * sin_theta
        extended_line = Line((x1_, y1_), (x2_, y2_))

        return extended_line

    def line_intersection(self, other: 'Line') -> Tuple[Optional[float], Optional[float]]:
        """
        Determines the intersection point between this line and another line.

        Args:
            other (Line): The other line to check for intersection.

        Returns:
            Tuple[Optional[float], Optional[float]]: The x and y coordinates of the intersection point, or (None, None) if there is no intersection.
        """
        a, b = self, other

        m_a = a.slope()
        m_b = b.slope()
        x_lower, x_upper = Line.x_bound(a, b)
        y_lower, y_upper = Line.y_bound(a, b)

        if m_a == m_b:
            return None, None  # Lines are parallel and do not intersect

        x_intersection = ((b.y1 - m_b * b.x1) - (a.y1 - m_a * a.x1)) / (m_a - m_b)
        y_intersection = m_a * x_intersection + a.y1 - m_a * a.x1

        tolerance_vertical_intersection = 0.00001
        if x_lower - tolerance_vertical_intersection <= x_intersection <= x_upper + tolerance_vertical_intersection and y_lower - tolerance_vertical_intersection <= y_intersection <= y_upper + tolerance_vertical_intersection:
            return x_intersection, y_intersection

        return None, None

    __and__ = line_intersection
