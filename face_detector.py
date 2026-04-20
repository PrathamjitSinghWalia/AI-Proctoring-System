import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Face detector running! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,   
        minNeighbors=5,    
        minSize=(30, 30)  
    )
    face_count = len(faces)
    
    for (x, y, w, h) in faces:
        color = (0, 255, 0)
        
        if face_count > 1:
            color = (0, 0, 255)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    
    status = f"Faces detected: {face_count}"
    color = (0, 255, 0) if face_count == 1 else (0, 0, 255)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    if face_count == 0:
        cv2.putText(frame, "WARNING: No face detected!", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    elif face_count > 1:
        cv2.putText(frame, "WARNING: Multiple faces!", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("AI Proctoring - Face Detector", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()