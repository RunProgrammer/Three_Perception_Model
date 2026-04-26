from ultralytics import YOLO
import cv2
import numpy as np
import math

model = YOLO("./weights/best.pt")
cap = cv2.VideoCapture(1)

def estimate_yaw_from_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0.0

    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    angle = rect[2]

    if angle < -45:
        angle = 90 + angle

    return np.deg2rad(angle)

print("2D Detection Only")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.45)

    for r in results:
        for box, cls, conf in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.cls.cpu().numpy(),
            r.boxes.conf.cpu().numpy()
        ):

            x1, y1, x2, y2 = map(int, box)
            u = int((x1 + x2) / 2)
            v = int((y1 + y2) / 2)

            roi = frame[y1:y2, x1:x2]
            yaw = estimate_yaw_from_roi(roi)

            print("--------------")
            print(f"Object: {model.names[int(cls)]}")
            print(f"Pixel center: ({u},{v})")
            print(f"Yaw (rad): {yaw:.3f}")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(frame, (u,v), 5, (0,0,255), -1)

    cv2.imshow("2D YOLO", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
