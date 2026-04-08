import cv2
from ultralytics import YOLO

model = YOLO("wighets/yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = results = model(frame, conf=0.5) 

    annotated = results[0].plot()
    cv2.imshow("YOLO Detection", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

