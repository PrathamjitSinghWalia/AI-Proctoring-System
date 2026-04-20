import cv2

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

print("Camera started! Press 'q' to quit.")

while True:
    # Read each frame
    ret, frame = cap.read()
    
    if not ret:
        print("Cannot access camera!")
        break
    
    # Show the frame in a window
    cv2.imshow("AI Proctoring - Camera Test", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
print("Camera closed.")