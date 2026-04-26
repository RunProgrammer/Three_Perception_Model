import cv2
import numpy as np
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)

# ---------------- CONFIG ----------------
DATASET_ROOT = "./dataset/test"
MIN_SHAPE_AREA = 1200
WB_MIN_AREA = 25000
IOU_THRESHOLD = 0.5
MISSED_LABEL = -1

CLASS_NAMES = [
    "bcircle",
    "bcube",
    "btriangle",
    "mtcircle",
    "mtcube",
    "mttriangle",
    "rcircle",
    "rcube",
    "rtriangle",
    "wb",
    "ycircle",
    "ycube",
    "ytriangle",
]


# =========================================
# SHAPE + COLOR CLASSIFIER (Your CV logic)
# =========================================
def classify_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)

    if len(approx) == 3:
        return "triangle"
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ar = w / float(h)
        return "cube" if 0.85 <= ar <= 1.15 else "rectangle"
    return "circle"


def classify_color(image, contour):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [contour], -1, 255, -1)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v, _ = cv2.mean(hsv, mask=mask)

    if s < 35:
        return None

    if h < 10 or h > 160:
        return "r"
    if 18 < h < 35:
        return "y"
    if 35 < h < 85:
        return "mt"
    return "b"


def detect_cv(image):
    predictions = []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2,
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_SHAPE_AREA:
            continue

        shape = classify_shape(cnt)
        color = classify_color(image, cnt)

        if color is None:
            continue

        label = f"{color}{shape}"

        x, y, w, h = cv2.boundingRect(cnt)
        predictions.append((label, (x, y, x + w, y + h)))

    return predictions


# =========================================
# IOU CALCULATION
# =========================================
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter
    return inter / union if union > 0 else 0


def parse_yolo_label_line(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    try:
        cls = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])
    except ValueError:
        return None

    return cls, xc, yc, bw, bh


# =========================================
# MAIN EVALUATION
# =========================================
def main():
    img_dir = os.path.join(DATASET_ROOT, "images")
    lbl_dir = os.path.join(DATASET_ROOT, "labels")

    y_true = []
    y_pred = []

    total_time = 0.0
    total_images = 0

    if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
        print(f"Dataset folders not found under: {DATASET_ROOT}")
        return

    for file in os.listdir(lbl_dir):
        if not file.endswith(".txt"):
            continue

        img_path = os.path.join(img_dir, file.replace(".txt", ".jpg"))
        label_path = os.path.join(lbl_dir, file)

        image = cv2.imread(img_path)
        if image is None:
            continue

        H, W = image.shape[:2]

        # ---- CV Detection ----
        start = time.time()
        predictions = detect_cv(image)
        total_time += time.time() - start
        total_images += 1

        # ---- Ground Truth ----
        gt_boxes = []
        with open(label_path, encoding="utf-8") as f:
            for line in f:
                parsed = parse_yolo_label_line(line)
                if parsed is None:
                    continue
                cls, xc, yc, bw, bh = parsed

                x1 = int((xc - bw / 2) * W)
                y1 = int((yc - bh / 2) * H)
                x2 = int((xc + bw / 2) * W)
                y2 = int((yc + bh / 2) * H)

                gt_boxes.append((cls, (x1, y1, x2, y2)))

        # ---- Matching ----
        for gt_cls, gt_box in gt_boxes:
            best_iou = 0.0
            best_pred = None

            for pred_label, pred_box in predictions:
                iou = compute_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred_label

            y_true.append(gt_cls)
            if best_iou > IOU_THRESHOLD and best_pred in CLASS_NAMES:
                y_pred.append(CLASS_NAMES.index(best_pred))
            else:
                y_pred.append(MISSED_LABEL)  # missed detection

    if not y_true:
        print("No labeled samples were evaluated.")
        return

    # =====================================
    # METRICS
    # =====================================
    print("\nClassical CV Evaluation")

    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy:", accuracy)

    eval_labels = [MISSED_LABEL] + list(range(len(CLASS_NAMES)))
    eval_names = ["missed"] + CLASS_NAMES

    print(
        classification_report(
            y_true,
            y_pred,
            labels=eval_labels,
            target_names=eval_names,
            zero_division=0,
        )
    )

    avg_time = total_time / total_images if total_images else 0.0
    print(f"Avg inference time per image: {avg_time:.4f} sec")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=eval_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=eval_names, yticklabels=eval_names)
    plt.title("Classical CV Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Detailed Graphs
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )

    x = np.arange(len(CLASS_NAMES))

    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.2, precision, 0.4, label="Precision")
    plt.bar(x + 0.2, recall, 0.4, label="Recall")
    plt.xticks(x, CLASS_NAMES, rotation=45)
    plt.title("Classical CV - Precision & Recall")
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.bar(CLASS_NAMES, f1)
    plt.xticks(rotation=45)
    plt.title("Classical CV - F1 Score")
    plt.show()


if __name__ == "__main__":
    main()
