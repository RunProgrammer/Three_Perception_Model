from ultralytics import YOLO
import cv2

# Load BEST model
model = YOLO("./weights/best.pt")

# Run inference
results = model("image3.png", conf=0.47 )

# Read image
img = cv2.imread("image3.png")

# Class names (IMPORTANT)
names = model.names

for r in results:
    for box, cls, conf in zip(
        r.boxes.xyxy.cpu().numpy(),
        r.boxes.cls.cpu().numpy(),
        r.boxes.conf.cpu().numpy()
    ):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[int(cls)]} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

cv2.imshow("YOLO Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
