import cv2
import mediapipe as mp
import numpy as np

from modules.yawn_detector import mouth_aspect_ratio, MAR_THRESHOLD
print("RUNNING:", __file__)
print("VERSION 3")
print("VERSION 3 - EAR + MAR")

# ---------- Helper Functions ----------

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(landmarks, eye_points, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_points]

    A = distance(pts[1], pts[5])
    B = distance(pts[2], pts[4])
    C = distance(pts[0], pts[3])

    return (A + B) / (2 * C)

# ---------- MediaPipe ----------

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.22

closed_frames = 0
yawn_frames = 0

# ---------- Webcam ----------

cap = cv2.VideoCapture(0)

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        face = results.multi_face_landmarks[0]

        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                thickness=1,
                circle_radius=1
            )
        )

        # ---------- Eye Detection ----------

        left_ear = eye_aspect_ratio(face.landmark, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(face.landmark, RIGHT_EYE, w, h)

        ear = (left_ear + right_ear) / 2

        if ear < EAR_THRESHOLD:
            closed_frames += 1
            eye_status = "Eyes: CLOSED"
            eye_color = (0, 0, 255)
        else:
            closed_frames = 0
            eye_status = "Eyes: OPEN"
            eye_color = (0, 255, 0)

        cv2.putText(frame, eye_status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, eye_color, 2)

        cv2.putText(frame, f"EAR: {ear:.2f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 0), 2)

        if closed_frames > 40:
            cv2.putText(frame,
                        "DROWSINESS ALERT!",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3)

        # ---------- Yawn Detection ----------

        mar = mouth_aspect_ratio(face.landmark, w, h)
        print(f"MAR={mar:.2f}")

        if mar > MAR_THRESHOLD:
            yawn_frames += 1
            mouth_status = "Mouth: YAWNING"
            mouth_color = (0, 165, 255)
        else:
            yawn_frames = 0
            mouth_status = "Mouth: Normal"
            mouth_color = (0, 255, 0)

        cv2.putText(frame,
                    mouth_status,
                    (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    mouth_color,
                    2)

        cv2.putText(frame,
                    f"MAR: {mar:.2f}",
                    (20, 205),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2)

        if yawn_frames > 20:
            cv2.putText(frame,
                        "YAWN ALERT!",
                        (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 165, 255),
                        3)

    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()