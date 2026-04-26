from ultralytics import YOLO
import cv2
import numpy as np

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO("./weights/best.pt")

# -----------------------------
# Open Camera (ZED as normal USB camera)
# -----------------------------
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

print("🎥 Starting detection... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]

    results = model(frame, conf=0.5)

    for r in results:
        for box, cls, conf in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.cls.cpu().numpy(),
            r.boxes.conf.cpu().numpy()
        ):
            x1, y1, x2, y2 = map(int, box)

            # Pixel center
            u = int((x1 + x2) / 2)
            v = int((y1 + y2) / 2)

            # -----------------------------
            # 2D Camera Coordinates
            # -----------------------------
            # Normalized image coordinates
            x_norm = (u - W/2) / (W/2)
            y_norm = (v - H/2) / (H/2)

            print("--------------")
            print(f"Object: {model.names[int(cls)]}")
            print(f"Pixel center: ({u}, {v})")
            print(f"Normalized X: {x_norm:.3f}")
            print(f"Normalized Y: {y_norm:.3f}")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(frame, (u,v), 5, (0,0,255), -1)

    cv2.imshow("YOLO Live", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
