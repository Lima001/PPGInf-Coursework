"""
Helper functions for geometry and math computations.
"""

import math

def get_orientation(A, B, C):
    """
    Determines the orientation of an ordered triplet (A, B, C).
    Returns True if clockwise, False if counter-clockwise/collinear.
    """
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def line_intersect(A, B, C, D):
    """
    Checks if line segment AB intersects with line segment CD.
    Returns True if they intersect, False otherwise.
    """
    return (get_orientation(A, C, D) != get_orientation(B, C, D) and 
            get_orientation(A, B, C) != get_orientation(A, B, D))

def point_line_segment_distance(px, py, P1, P2):
    """
    Computes the shortest distance from point (px, py) to the line segment P1-P2.
    """
    x1, y1 = P1
    x2, y2 = P2
    dx, dy = x2 - x1, y2 - y1
    
    if dx == 0 and dy == 0: 
        return math.hypot(px - x1, py - y1)
    
    t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
    t = max(0, min(1, t))
    nx, ny = x1 + t * dx, y1 + t * dy
    
    return math.hypot(px - nx, py - ny)

def get_line_center(P1, P2):
    """Returns center point (x, y) of a line segment."""
    return ((P1[0] + P2[0]) / 2, (P1[1] + P2[1]) / 2)