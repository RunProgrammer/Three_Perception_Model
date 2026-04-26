from ultralytics import YOLO
import cv2
import numpy as np

# -----------------------------
# CAMERA PARAMETERS (replace after calibration)
# -----------------------------
fx = 800.0
fy = 800.0
cx = 320.0
cy = 240.0

CAMERA_HEIGHT = 0.19  # 19 cm in meters

TARGET_CLASS = "ycube"

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO("./weights/best.pt")
names = model.names

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

print("🎥 Detecting ycube only... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)

    for r in results:
        for box, cls, conf in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.cls.cpu().numpy(),
            r.boxes.conf.cpu().numpy()
        ):

            class_name = names[int(cls)]

            # 🔥 Filter only ycube
            if class_name != TARGET_CLASS:
                continue

            x1, y1, x2, y2 = map(int, box)

            # Pixel center
            u = int((x1 + x2) / 2)
            v = int((y1 + y2) / 2)

            # Convert to camera coordinates (overhead assumption)
            Z = CAMERA_HEIGHT
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy

            print("-------------")
            print("Detected: ycube")
            print(f"Camera Coordinates (meters):")
            print(f"X: {X:.4f}")
            print(f"Y: {Y:.4f}")
            print(f"Z: {Z:.4f}")

            # Draw
            label = f"{class_name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(frame, (u, v), 5, (0,0,255), -1)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,255,0), 2)

    cv2.imshow("YCUBE Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
