import cv2
import numpy as np
import joblib

# ---------------- CONFIG ----------------
MODEL_PATH = "ml_model.pkl"
IMAGE_PATH = "image2.png"
MIN_AREA = 500
SHOW = True
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

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-6)

    extent = area / (w * h + 1e-6)

    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h_mean, s_mean, v_mean = np.mean(hsv.reshape(-1, 3), axis=0)

    features = [
        area, peri, circularity, aspect_ratio,
        solidity, extent,
        h_mean, s_mean, v_mean,
        *hu
    ]

    return np.array(features, dtype=np.float32)


# ---------- MAIN ----------
def main():
    print("🔄 Loading model...")
    model = joblib.load(MODEL_PATH)

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print("❌ Image not found")
        return

    H, W = img.shape[:2]
    output = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    print("\n📦 Detections:\n")

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        roi = img[y:y+h, x:x+w]

        feat = extract_features(roi)
        if feat is None:
            continue

        feat = feat.reshape(1, -1)

        pred_class = model.predict(feat)[0]
        conf = np.max(model.predict_proba(feat))

        cx = x + w // 2
        cy = y + h // 2

        print("------")
        print(f"Class ID     : {pred_class}")
        print(f"Confidence   : {conf:.3f}")
        print(f"Pixel Box    : ({x},{y}) → ({x+w},{y+h})")
        print(f"Center (px)  : ({cx}, {cy})")
        print(f"Center (norm): ({cx/W:.3f}, {cy/H:.3f})")

        # Draw box
        label = f"{pred_class} ({conf:.2f})"
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(output, label, (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 2)

    if SHOW:
        cv2.imshow("ML Detection", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.imwrite("ml_result.png", output)
    print("\n✅ Saved result as ml_result.png")


if __name__ == "__main__":
    main()
