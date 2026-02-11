import cv2
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
ok, frame = cap.read()
cap.release()
if ok:
    cv2.imwrite("demos/webcam_frame.png", frame)
    print("Wrote demos/webcam_frame.png")
else:
    print("Failed to capture")
