import cv2
import numpy as np
import os
import joblib
import time
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.model_selection import GridSearchCV

# ---------------- CONFIG ----------------
DATASET_ROOT = "./dataset"
MODEL_PATH = "ml_model_rf.pkl"
MIN_AREA = 500

CLASS_NAMES = [
    'bcircle', 'bcube', 'btriangle',
    'mtcircle', 'mtcube', 'mttriangle',
    'rcircle', 'rcube', 'rtriangle',
    'wb',
    'ycircle', 'ycube', 'ytriangle'
]

# =========================================
# FEATURE EXTRACTION
# =========================================
def extract_features(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area < MIN_AREA:
        return None

    peri = cv2.arcLength(cnt, True)
    circularity = (4 * np.pi * area) / (peri * peri + 1e-6)

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / float(h)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-6)
    extent = area / (w * h + 1e-6)

    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_mean, s_mean, v_mean = np.mean(hsv.reshape(-1, 3), axis=0)

    return np.array([
        area, peri, circularity, aspect_ratio,
        solidity, extent,
        h_mean, s_mean, v_mean,
        *hu
    ], dtype=np.float32)


# =========================================
# DATA LOADER
# =========================================
def load_split(split):
    X, y = [], []

    img_dir = os.path.join(DATASET_ROOT, split, "images")
    lbl_dir = os.path.join(DATASET_ROOT, split, "labels")

    for file in os.listdir(lbl_dir):
        if not file.endswith(".txt"):
            continue

        img_path = os.path.join(img_dir, file.replace(".txt", ".jpg"))
        label_path = os.path.join(lbl_dir, file)

        img = cv2.imread(img_path)
        if img is None:
            continue

        H, W = img.shape[:2]

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])

                x1 = int((xc - bw/2) * W)
                y1 = int((yc - bh/2) * H)
                x2 = int((xc + bw/2) * W)
                y2 = int((yc + bh/2) * H)

                roi = img[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                feat = extract_features(roi)
                if feat is not None:
                    X.append(feat)
                    y.append(cls)

    return np.array(X), np.array(y)


# =========================================
# PLOTTING FUNCTION
# =========================================
def plot_detailed_metrics(y_true, y_pred, class_names, model_name):

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_names)), zero_division=0
    )

    x = np.arange(len(class_names))

    # Precision & Recall
    plt.figure(figsize=(12,6))
    plt.bar(x - 0.2, precision, width=0.4, label='Precision')
    plt.bar(x + 0.2, recall, width=0.4, label='Recall')
    plt.xticks(x, class_names, rotation=45)
    plt.title(f"{model_name} - Precision & Recall per Class")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # F1 Score
    plt.figure(figsize=(12,6))
    plt.bar(class_names, f1)
    plt.xticks(rotation=45)
    plt.title(f"{model_name} - F1 Score per Class")
    plt.tight_layout()
    plt.show()

    # Class Accuracy
    plt.figure(figsize=(12,6))
    plt.bar(class_names, recall)
    plt.xticks(rotation=45)
    plt.title(f"{model_name} - Class-wise Accuracy")
    plt.tight_layout()
    plt.show()


# =========================================
# MAIN
# =========================================
def main():

    print("Loading Dataset...")
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("valid")
    X_test, y_test = load_split("test")

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ---------------- TRAIN ----------------
    base_model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 20]
    }

    grid = GridSearchCV(base_model, param_grid, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)

    model = grid.best_estimator_
    joblib.dump(model, MODEL_PATH)

    print("Best Params:", grid.best_params_)

    # ---------------- TEST ----------------
    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()

    accuracy = accuracy_score(y_test, y_pred)
    print("\nTEST RESULTS")
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred, zero_division=0))

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro'
    )

    print("Macro Precision:", macro_p)
    print("Macro Recall:", macro_r)
    print("Macro F1:", macro_f1)

    print(f"Avg Inference Time: {(end-start)/len(X_test)*1000:.4f} ms")

    # ---------------- CONFUSION MATRIX ----------------
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(CLASS_NAMES)))

    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # ---------------- DETAILED GRAPHS ----------------
    plot_detailed_metrics(y_test, y_pred, CLASS_NAMES, "Random Forest")


if __name__ == "__main__":
    main()


