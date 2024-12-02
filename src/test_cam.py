# quick webcam test - just checking if cv2 can grab frames
import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("cant open camera")
    exit()

print("press q to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
