from ultralytics import YOLO
import cv2
import json

# ---------------- CONFIG ----------------
IMAGE_PATH = "image1.png"
MODEL_PATH = "./weights/best.pt"
CONF_THRES = 0.47
# ----------------------------------------

# Load model
model = YOLO(MODEL_PATH)

# Run inference
results = model(IMAGE_PATH, conf=CONF_THRES)

# Read image
img = cv2.imread(IMAGE_PATH)
H, W = img.shape[:2]

names = model.names

detections = []  # for ROS / export

for r in results:
    for box, cls, conf in zip(
        r.boxes.xyxy.cpu().numpy(),
        r.boxes.cls.cpu().numpy(),
        r.boxes.conf.cpu().numpy()
    ):
        x1, y1, x2, y2 = map(int, box)

        # Pixel values
        width = x2 - x1
        height = y2 - y1
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Normalized values (better for ROS)
        cx_norm = cx / W
        cy_norm = cy / H
        w_norm = width / W
        h_norm = height / H

        class_name = names[int(cls)]

        # Store detection
        detection = {
            "class": class_name,
            "confidence": float(conf),
            "bbox_pixels": [x1, y1, x2, y2],
            "center_pixels": [cx, cy],
            "bbox_normalized": [cx_norm, cy_norm, w_norm, h_norm]
        }

        detections.append(detection)

        # Print clean output
        print("------ DETECTION ------")
        print(f"Class       : {class_name}")
        print(f"Confidence  : {conf:.3f}")
        print(f"Pixel Box   : ({x1}, {y1}) → ({x2}, {y2})")
        print(f"Center(px)  : ({cx}, {cy})")
        print(f"Center(norm): ({cx_norm:.3f}, {cy_norm:.3f})")
        print()

        # Draw on image
        label = f"{class_name} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

# Save JSON (VERY useful for ROS integration)
with open("detections.json", "w") as f:
    json.dump(detections, f, indent=4)

print("✅ Saved detections to detections.json")

cv2.imshow("YOLO Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
