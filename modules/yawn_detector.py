import numpy as np

# More stable mouth landmark set
UPPER_LIP = [13, 312, 311]
LOWER_LIP = [14, 87, 178]
LEFT_CORNER = 61
RIGHT_CORNER = 291

MAR_THRESHOLD = 0.58

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def mouth_aspect_ratio(landmarks, w, h):
    def pt(i):
        return np.array([landmarks[i].x * w, landmarks[i].y * h])

    vertical = (
        distance(pt(13), pt(14)) +
        distance(pt(312), pt(87)) +
        distance(pt(311), pt(178))
    ) / 3

    horizontal = distance(pt(LEFT_CORNER), pt(RIGHT_CORNER))

    return vertical / horizontal