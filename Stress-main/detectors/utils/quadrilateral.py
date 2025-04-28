"""
This file handles some of the basic geometry calculations for dealing with quadrilaterals.
"""

import cv2
from utils.line import Line
from typing import Tuple, Optional

class Quadrilateral:
    """
    A class representing a quadrilateral, defined by four points. Provides methods for 
    drawing the quadrilateral and checking for intersections with other quadrilaterals or points.
    """

    def __init__(self, pointa: Tuple[int, int], pointb: Tuple[int, int], 
                 pointc: Tuple[int, int], pointd: Tuple[int, int]):
        """
        Initializes a Quadrilateral with four points. The points must be in clockwise or counter-clockwise order.

        Args:
            pointa (Tuple[int, int]): The first point of the quadrilateral.
            pointb (Tuple[int, int]): The second point of the quadrilateral.
            pointc (Tuple[int, int]): The third point of the quadrilateral.
            pointd (Tuple[int, int]): The fourth point of the quadrilateral.
        """
        self.pointa = pointa
        self.pointb = pointb
        self.pointc = pointc
        self.pointd = pointd

    def paint_quadrilateral(self, frame: cv2.Mat, c: Tuple[int, int, int] = (240, 16, 255)) -> cv2.Mat:
        """
        Draws the quadrilateral on an image.

        Args:
            frame (cv2.Mat): The image on which to draw the quadrilateral.
            c (Tuple[int, int, int]): The color to use for drawing the quadrilateral.

        Returns:
            cv2.Mat: The image with the quadrilateral drawn on it.
        """
        frame = cv2.line(frame, self.pointa, self.pointb, color=c, thickness=2)
        frame = cv2.line(frame, self.pointb, self.pointc, color=c, thickness=2)
        frame = cv2.line(frame, self.pointc, self.pointd, color=c, thickness=2)
        frame = cv2.line(frame, self.pointd, self.pointa, color=c, thickness=2)

        return frame

    def quadrilateral_intersection(self, other: Optional['Quadrilateral'], tolerance: int = 0) -> bool:
        """
        Checks if this quadrilateral intersects with another quadrilateral.

        Args:
            other (Optional[Quadrilateral]): The other quadrilateral to check for intersection.
            tolerance (int): The tolerance to extend the sides of the other quadrilateral.

        Returns:
            bool: True if there is an intersection, False otherwise.
        """
        if other is None:
            return False
        alpha, beta = self, other

        alpha_list = [Line(alpha.pointa, alpha.pointb), Line(alpha.pointb, alpha.pointc),
                      Line(alpha.pointc, alpha.pointd), Line(alpha.pointd, alpha.pointa)]

        beta_list = [Line.extend_segment(Line(beta.pointa, beta.pointb), length=tolerance),
                     Line.extend_segment(Line(beta.pointb, beta.pointc), length=tolerance),
                     Line.extend_segment(Line(beta.pointc, beta.pointd), length=tolerance),
                     Line.extend_segment(Line(beta.pointd, beta.pointa), length=tolerance)]

        intersection_bool = beta.quadrilateral_with_point(self.pointa)

        for alpha_side in alpha_list:
            for beta_side in beta_list:
                intersection = Line.line_intersection(alpha_side, beta_side)
                if intersection[0] is not None:
                    intersection_bool = True

        return intersection_bool

    def quadrilateral_with_point(self, point: Tuple[int, int]) -> bool:
        """
        Checks if a point is inside this quadrilateral.

        Args:
            point (Tuple[int, int]): The point to check.

        Returns:
            bool: True if the point is inside the quadrilateral, False otherwise.
        """
        quadrilateral = self

        alpha1 = Line(quadrilateral.pointa, quadrilateral.pointb)
        alpha2 = Line(quadrilateral.pointb, quadrilateral.pointc)
        alpha3 = Line(quadrilateral.pointc, quadrilateral.pointd)
        alpha4 = Line(quadrilateral.pointd, quadrilateral.pointa)

        line1 = Line(quadrilateral.pointa, point)
        line2 = Line(quadrilateral.pointb, point)
        line3 = Line(quadrilateral.pointc, point)
        line4 = Line(quadrilateral.pointd, point)

        pairs = [[line1, alpha2], [line1, alpha3], [line2, alpha3], [line2, alpha4],
                 [line3, alpha4], [line3, alpha1], [line4, alpha1], [line4, alpha2]]

        inside_bool = True
        for pair in pairs:
            intersection = Line.line_intersection(pair[0], pair[1])
            if intersection[0] is not None:
                inside_bool = False

        return inside_bool

    def get_min_max_points(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Calculates the minimum and maximum coordinates of the quadrilateral.

        Returns:
            Tuple[Tuple[int, int], Tuple[int, int]]: The minimum and maximum points of the quadrilateral.
        """
        max_point = (max(self.pointa[0], self.pointb[0], self.pointc[0], self.pointd[0]),
                     max(self.pointa[1], self.pointb[1], self.pointc[1], self.pointd[1]))
        min_point = (min(self.pointa[0], self.pointb[0], self.pointc[0], self.pointd[0]),
                     min(self.pointa[1], self.pointb[1], self.pointc[1], self.pointd[1]))

        return min_point, max_point
