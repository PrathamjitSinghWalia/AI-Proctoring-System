import cv2
from datetime import datetime
from ultralytics import YOLO
import mediapipe as mp

# Import directly from our existing files
# No need to rewrite any functions!
from gaze_tracker import (
    get_eye_ratio,
    get_iris_position,
    get_head_pose,
    get_gaze_direction,
    LEFT_EYE, RIGHT_EYE,
    LEFT_IRIS, RIGHT_IRIS,
    LEFT_EYE_LEFT_CORNER, LEFT_EYE_RIGHT_CORNER,
    RIGHT_EYE_LEFT_CORNER, RIGHT_EYE_RIGHT_CORNER
)
from logger import SessionLogger

# ============================
# CONSTANTS
# ============================
SUSPICIOUS_OBJECTS = {
    67: "PHONE",
    63: "LAPTOP",
    73: "BOOK"
}

# ============================
# COOLDOWN TRACKER
# ============================
last_logged = {}
COOLDOWN_SECONDS = 2

def can_log(event_type):
    now = datetime.now()
    if event_type not in last_logged:
        last_logged[event_type] = now
        return True
    seconds_since_last = (now - last_logged[event_type]).total_seconds()
    if seconds_since_last >= COOLDOWN_SECONDS:
        last_logged[event_type] = now
        return True
    return False

# ============================
# INIT MODELS
# ============================
print("Loading models, please wait...")
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
yolo_model = YOLO('yolov8s.pt')
print("All models loaded!")

# ============================
# MAIN PROGRAM
# ============================
student_name = input("Enter student name: ")
logger = SessionLogger(student_name)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

detection_counter = {obj: 0 for obj in SUSPICIOUS_OBJECTS.values()}
CONSECUTIVE_FRAMES_NEEDED = 5

print(f"\nProctoring session started for: {student_name}")
print("Press 'q' to end session\n")

with mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ============================
        # FACE + GAZE DETECTION
        # ============================
        face_results = face_mesh.process(rgb_frame)

        gaze = "No face detected"
        avg_ear = 0
        face_count = 0

        if face_results.multi_face_landmarks:
            face_count = len(face_results.multi_face_landmarks)

            if face_count > 1:
                if can_log("MULTIPLE_FACES"):
                    logger.log_event("MULTIPLE_FACES", f"{face_count} faces detected")

            landmarks = face_results.multi_face_landmarks[0].landmark
            gaze = get_gaze_direction(landmarks, fw, fh)

            left_ear = get_eye_ratio(landmarks, LEFT_EYE, fw, fh)
            right_ear = get_eye_ratio(landmarks, RIGHT_EYE, fw, fh)
            avg_ear = (left_ear + right_ear) / 2

            if avg_ear < 0.2:
                gaze = "Eyes CLOSED"
                
            elif gaze != "Looking FORWARD":
                if can_log("LOOKING_AWAY"):
                    logger.log_event("LOOKING_AWAY", gaze)

            mp_drawing.draw_landmarks(
                frame,
                face_results.multi_face_landmarks[0],
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=1, circle_radius=1)
            )

            for iris_points in [LEFT_IRIS, RIGHT_IRIS]:
                iris_x = int(sum(landmarks[p].x for p in iris_points) / len(iris_points) * fw)
                iris_y = int(sum(landmarks[p].y for p in iris_points) / len(iris_points) * fh)
                cv2.circle(frame, (iris_x, iris_y), 3, (0, 255, 255), -1)

        else:
            if can_log("NO_FACE"):
                logger.log_event("NO_FACE", "Face not visible")

        # ============================
        # OBJECT DETECTION
        # ============================
        yolo_results = yolo_model(frame, verbose=False)
        detected_this_frame = set()

        for box in yolo_results[0].boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if class_id in SUSPICIOUS_OBJECTS and confidence > 0.4:
                object_name = SUSPICIOUS_OBJECTS[class_id]
                detected_this_frame.add(object_name)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"{object_name} {confidence:.0%}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for obj in SUSPICIOUS_OBJECTS.values():
            if obj in detected_this_frame:
                detection_counter[obj] += 1
            else:
                detection_counter[obj] = 0

            if detection_counter[obj] == CONSECUTIVE_FRAMES_NEEDED:
                event_type = f"{obj}_DETECTED"
                if can_log(event_type):
                    logger.log_event(event_type, f"{obj} confirmed")

        # ============================
        # DISPLAY INFO ON SCREEN
        # ============================
        score = logger.get_score()

        gaze_color = (0, 255, 0) if gaze == "Looking FORWARD" else (0, 0, 255)
        cv2.putText(frame, gaze, (10, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, gaze_color, 2)

        face_color = (0, 255, 0) if face_count == 1 else (0, 0, 255)
        cv2.putText(frame, f"Faces: {face_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, face_color, 2)

        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if score < 10:
            score_color = (0, 255, 0)
        elif score < 25:
            score_color = (0, 165, 255)
        else:
            score_color = (0, 0, 255)

        cv2.putText(frame, f"Suspicion Score: {score}", (10, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, score_color, 2)

        if face_count > 1:
            cv2.putText(frame, "WARNING: Multiple faces!", (10, 175),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("AI Proctoring System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
logger.end_session()

# Auto generate report after session ends
print("\nGenerating session report...")
from report_generator import generate_pdf_report
pdf_path = generate_pdf_report(logger.csv_path, student_name)
print(f"Report saved: {pdf_path}")