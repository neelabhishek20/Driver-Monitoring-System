import cv2
import numpy as np

def estimate_head_pose(face_landmarks, frame_w, frame_h):
    face_2d = []
    face_3d = []

    landmark_ids = [33, 263, 1, 61, 291, 199]

    for idx in landmark_ids:
        lm = face_landmarks.landmark[idx]

        x = lm.x * frame_w
        y = lm.y * frame_h

        face_2d.append([x, y])
        face_3d.append([x, y, lm.z * 3000])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    focal_length = frame_w

    cam_matrix = np.array([
        [focal_length, 0, frame_w / 2],
        [0, focal_length, frame_h / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(
        face_3d,
        face_2d,
        cam_matrix,
        dist_matrix,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, *_ = cv2.RQDecomp3x3(rmat)

    pitch = angles[0]
    yaw = angles[1]

    # Camera calibration offsets
    PITCH_OFFSET = 8.5
    YAW_OFFSET = 0

    pitch -= PITCH_OFFSET
    yaw -= YAW_OFFSET

    nose = face_landmarks.landmark[1]

    nose_x = int(nose.x * frame_w)
    nose_y = int(nose.y * frame_h)

    end_x = int(nose_x + yaw * 5)
    end_y = int(nose_y - pitch * 5)

    # Direction thresholds
    if yaw < -12:
        direction = "LEFT"
    elif yaw > 12:
        direction = "RIGHT"
    elif pitch < -12:
        direction = "DOWN"
    elif pitch > 12:
        direction = "UP"
    else:
        direction = "FORWARD"

    return direction, (nose_x, nose_y), (end_x, end_y), yaw, pitch