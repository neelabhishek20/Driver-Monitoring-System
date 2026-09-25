import cv2
import mediapipe as mp
import numpy as np

# ---------------- Helper Functions ----------------

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(landmarks, eye_points, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_points]

    A = distance(pts[1], pts[5])
    B = distance(pts[2], pts[4])
    C = distance(pts[0], pts[3])

    return (A + B) / (2.0 * C)

# ---------------- MediaPipe Setup ----------------

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark IDs
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.22
closed_frames = 0

# ---------------- Webcam ----------------

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

        # Draw Face Mesh
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

        # Calculate EAR
        left_ear = eye_aspect_ratio(face.landmark, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(face.landmark, RIGHT_EYE, w, h)
        ear = (left_ear + right_ear) / 2

        # Blink Detection
        if ear < EAR_THRESHOLD:
            closed_frames += 1
            status = "Eyes: CLOSED"
            color = (0, 0, 255)
        else:
            closed_frames = 0
            status = "Eyes: OPEN"
            color = (0, 255, 0)

        cv2.putText(frame, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(frame, f"EAR: {ear:.2f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if closed_frames > 40:
            cv2.putText(frame, "DROWSINESS ALERT!", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 3)

    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()