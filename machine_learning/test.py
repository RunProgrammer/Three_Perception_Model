import cv2
import numpy as np
import os
import joblib

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# ---------------- CONFIG ----------------
DATASET_ROOT = "./dataset"     # must contain train/valid/test
MODEL_PATH = "ml_model.pkl"
MIN_AREA = 500
# ----------------------------------------


# ---------- FEATURE EXTRACTION ----------
def extract_features(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

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

    # ---- Advanced geometric features ----
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-6)

    extent = area / (w * h + 1e-6)

    # ---- Hu Moments (shape invariant) ----
    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()

    # ---- Color Features ----
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_mean, s_mean, v_mean = np.mean(hsv.reshape(-1, 3), axis=0)

    features = [
        area,
        peri,
        circularity,
        aspect_ratio,
        solidity,
        extent,
        h_mean,
        s_mean,
        v_mean,
        *hu
    ]

    return np.array(features, dtype=np.float32)


# ---------- LOAD YOLO DATA ----------
def load_split(split):
    X, y = [], []

    img_dir = os.path.join(DATASET_ROOT, split, "images")
    lbl_dir = os.path.join(DATASET_ROOT, split, "labels")

    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        print(f"⚠️ Missing split folder: {split}")
        return np.array([]), np.array([])

    for file in os.listdir(lbl_dir):
        if not file.endswith(".txt"):
            continue

        img_path = os.path.join(img_dir, file.replace(".txt", ".jpg"))
        label_path = os.path.join(lbl_dir, file)

        if not os.path.exists(img_path):
            continue

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

                coords = list(map(float, parts[1:]))
                xc, yc, bw, bh = coords[:4]

                x1 = int((xc - bw / 2) * W)
                y1 = int((yc - bh / 2) * H)
                x2 = int((xc + bw / 2) * W)
                y2 = int((yc + bh / 2) * H)

                roi = img[max(0, y1):min(H, y2),
                          max(0, x1):min(W, x2)]

                if roi.size == 0:
                    continue

                feat = extract_features(roi)
                if feat is None:
                    continue

                X.append(feat)
                y.append(cls)

    return np.array(X), np.array(y)


# ---------- MAIN ----------
def main():
    print("📦 Building ML dataset from YOLO labels...\n")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("valid")
    X_test, y_test = load_split("test")

    print(f"Train samples: {len(X_train)}")
    print(f"Val samples:   {len(X_val)}")
    print(f"Test samples:  {len(X_test)}")

    if len(X_train) == 0:
        print("❌ No training data found.")
        return

    print("\nUnique classes in train:", np.unique(y_train))
    print("Unique classes in val:", np.unique(y_val))
    print("Unique classes in test:", np.unique(y_test))

    # ---------- MODEL ----------
    base_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced"))
    ])

    param_grid = {
        "svm__C": [1, 10, 50],
        "svm__gamma": ["scale", 0.01, 0.001]
    }

    print("\n🧠 Running Grid Search...")
    grid = GridSearchCV(base_model, param_grid, cv=3, verbose=1)
    grid.fit(X_train, y_train)

    model = grid.best_estimator_

    print("\n✅ Best Parameters:", grid.best_params_)

    # ---------- SAVE ----------
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Model saved as {MODEL_PATH}")

    # ---------- EVALUATION ----------
    print("\n📊 Validation Results:")
    y_pred_val = model.predict(X_val)
    print("Accuracy:", accuracy_score(y_val, y_pred_val))
    print(classification_report(y_val, y_pred_val))

    print("\n📊 Test Results:")
    y_pred_test = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred_test))
    print(classification_report(y_test, y_pred_test))


if __name__ == "__main__":
    main()
