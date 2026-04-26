from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("./weights/best.pt")

# Try camera index 0 or 1
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

print("🎥 Starting live detection... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Run YOLO on frame
    results = model(frame, conf=0.47)

    names = model.names

    for r in results:
        for box, cls, conf in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.cls.cpu().numpy(),
            r.boxes.conf.cpu().numpy()
        ):
            x1, y1, x2, y2 = map(int, box)
            label = f"{names[int(cls)]} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

    cv2.imshow("YOLO Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
