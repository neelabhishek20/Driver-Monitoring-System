import cv2
import mediapipe as mp
import numpy as np
import winsound

from modules.yawn_detector import mouth_aspect_ratio, MAR_THRESHOLD
from modules.head_pose import estimate_head_pose

# ---------------- Helper Functions ----------------

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(landmarks, eye_points, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_points]

    A = distance(pts[1], pts[5])
    B = distance(pts[2], pts[4])
    C = distance(pts[0], pts[3])

    return (A + B) / (2 * C)

# ---------------- MediaPipe ----------------

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33,160,158,133,153,144]
RIGHT_EYE = [362,385,387,263,373,380]

EAR_THRESHOLD = 0.22

closed_frames = 0
yawn_frames = 0

# ---------------- Head Pose ----------------

prev_end_point = None

# ---------------- Calibration ----------------

baseline_yaw = 0
baseline_pitch = 0
calibration_frames = 0
CALIBRATION_LIMIT = 60      # ~3 seconds

# ---------------- Distraction ----------------

distraction_frames = 0
FPS_ESTIMATE = 20
ALERT_FRAMES = 40

# ---------------- Alarm ----------------

alarm_playing = False

# ---------------- Webcam ----------------

cap = cv2.VideoCapture(0)

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    # Mirror the webcam
    frame = cv2.flip(frame,1)

    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:

        face = results.multi_face_landmarks[0]

        # ---------------- Face Mesh ----------------

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

        # ---------------- Eye Blink ----------------

        left_ear = eye_aspect_ratio(face.landmark, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(face.landmark, RIGHT_EYE, w, h)

        ear = (left_ear + right_ear) / 2

        if ear < EAR_THRESHOLD:
            closed_frames += 1
            eye_status = "Eyes: CLOSED"
            eye_color = (0,0,255)
        else:
            closed_frames = 0
            eye_status = "Eyes: OPEN"
            eye_color = (0,255,0)

        cv2.putText(frame, eye_status, (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,eye_color,2)

        cv2.putText(frame, f"EAR: {ear:.2f}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)

        if closed_frames > 40:
            cv2.putText(frame,"DROWSINESS ALERT!",
                        (20,120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        3)

        # ---------------- Yawn ----------------

        mar = mouth_aspect_ratio(face.landmark, w, h)

        if mar > MAR_THRESHOLD:
            yawn_frames += 1
            mouth_status = "Mouth: YAWNING"
            mouth_color = (0,165,255)
        else:
            yawn_frames = 0
            mouth_status = "Mouth: Normal"
            mouth_color = (0,255,0)

        cv2.putText(frame,mouth_status,
                    (20,170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    mouth_color,
                    2)

        cv2.putText(frame,f"MAR: {mar:.2f}",
                    (20,205),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255,255,0),
                    2)

        if yawn_frames > 20:
            cv2.putText(frame,
                        "YAWN ALERT!",
                        (20,240),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,165,255),
                        3)

        # ---------------- Head Pose ----------------

        _, start_point, end_point, yaw, pitch = estimate_head_pose(face, w, h)

        if prev_end_point is None:
            prev_end_point = end_point
        else:
            prev_end_point = (
                int(prev_end_point[0]*0.85 + end_point[0]*0.15),
                int(prev_end_point[1]*0.85 + end_point[1]*0.15)
            )

        cv2.line(frame,start_point,prev_end_point,(255,0,0),3)

        # ---------------- Auto Calibration ----------------

        if calibration_frames < CALIBRATION_LIMIT:

            baseline_yaw += yaw
            baseline_pitch += pitch
            calibration_frames += 1

            cv2.putText(frame,
                        f"Calibrating... {calibration_frames}/{CALIBRATION_LIMIT}",
                        (20,290),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,255,255),
                        2)

        else:

            avg_yaw = baseline_yaw / CALIBRATION_LIMIT
            avg_pitch = baseline_pitch / CALIBRATION_LIMIT

            rel_yaw = yaw - avg_yaw
            rel_pitch = pitch - avg_pitch

            # Mirror correction
            if rel_yaw < -8:
                direction = "RIGHT"
            elif rel_yaw > 8:
                direction = "LEFT"
            elif rel_pitch < -12:
                direction = "DOWN"
            elif rel_pitch > 12:
                direction = "UP"
            else:
                direction = "FORWARD"

            cv2.putText(frame,
                        f"Head: {direction}",
                        (20,290),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255,0,0),
                        2)

            cv2.putText(frame,
                        f"Yaw:{rel_yaw:.1f} Pitch:{rel_pitch:.1f}",
                        (20,320),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255,255,255),
                        2)

            # ---------------- Distraction Timer ----------------

            distracted = abs(rel_yaw) > 8 or rel_pitch < -12

            if distracted:
                distraction_frames += 1
            else:
                distraction_frames = max(0, distraction_frames - 3)

            timer = distraction_frames / FPS_ESTIMATE

            cv2.putText(frame,
                        f"Timer: {timer:.1f}s",
                        (20,350),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255,255,255),
                        2)

            # ---------------- Alerts ----------------

            if distraction_frames >= ALERT_FRAMES:

                if rel_pitch < -12:
                    alert = "PHONE DISTRACTION!"
                else:
                    alert = "DISTRACTION ALERT!"

                cv2.putText(frame,
                            alert,
                            (20,390),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,0,255),
                            3)

                # Start continuous alarm only once
                if not alarm_playing:
                    winsound.PlaySound(
                        "SystemExclamation",
                        winsound.SND_ALIAS |
                        winsound.SND_ASYNC |
                        winsound.SND_LOOP
                    )
                    alarm_playing = True

            else:

                # Stop alarm immediately
                if alarm_playing:
                    winsound.PlaySound(None, winsound.SND_PURGE)
                    alarm_playing = False

    cv2.imshow("Driver Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Stop sound before exiting
winsound.PlaySound(None, winsound.SND_PURGE)

cap.release()
cv2.destroyAllWindows()