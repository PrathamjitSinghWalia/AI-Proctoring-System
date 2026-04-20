import cv2
from ultralytics import YOLO

model = YOLO('yolov8s.pt')

SUSPICIOUS_OBJECTS = {
    67: "PHONE",
    63: "LAPTOP",
    73: "BOOK"
}

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Object detector running! Press 'q' to quit.")

# Consecutive detection counter - outside the loop so it persists
# This is the key to eliminating false positives
detection_counter = {}
CONSECUTIVE_FRAMES_NEEDED = 5  # must detect in 5 frames in a row to trigger warning

for obj in SUSPICIOUS_OBJECTS.values():
    detection_counter[obj] = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    results = model(frame, verbose=False)

    # Track what was detected this frame
    detected_this_frame = set()

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        # Keep confidence at 0.4 for good accuracy
        if class_id in SUSPICIOUS_OBJECTS and confidence > 0.4:
            object_name = SUSPICIOUS_OBJECTS[class_id]
            detected_this_frame.add(object_name)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{object_name} {confidence:.0%}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Update consecutive counters
    # If detected this frame - increment counter
    # If not detected this frame - reset counter to zero
    for obj in SUSPICIOUS_OBJECTS.values():
        if obj in detected_this_frame:
            detection_counter[obj] += 1
        else:
            detection_counter[obj] = 0

    # Only warn if detected consistently across multiple frames
    confirmed_detections = [
        obj for obj, count in detection_counter.items()
        if count >= CONSECUTIVE_FRAMES_NEEDED
    ]

    if confirmed_detections:
        objects_str = ", ".join(confirmed_detections)
        cv2.putText(frame, f"WARNING: {objects_str} detected!", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "No suspicious objects", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("AI Proctoring - Object Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()