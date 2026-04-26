import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 🔧 INSERT YOUR ACTUAL RESULTS HERE
# ==========================================

YOLO_RESULTS = {
    "accuracy": 0.89,
    "precision": 0.88,
    "recall": 0.87,
    "f1": 0.88,
    "time_ms": 28
}

ML_RESULTS = {
    "accuracy": 0.76,
    "precision": 0.72,
    "recall": 0.70,
    "f1": 0.71,
    "time_ms": 4
}

CV_RESULTS = {
    "accuracy": 0.61,
    "precision": 0.55,
    "recall": 0.52,
    "f1": 0.53,
    "time_ms": 2
}

# ==========================================
# PREPARE DATA
# ==========================================

models = ["YOLO (CNN)", "ML (RF)", "Classical CV"]

accuracy = [
    YOLO_RESULTS["accuracy"],
    ML_RESULTS["accuracy"],
    CV_RESULTS["accuracy"]
]

precision = [
    YOLO_RESULTS["precision"],
    ML_RESULTS["precision"],
    CV_RESULTS["precision"]
]

recall = [
    YOLO_RESULTS["recall"],
    ML_RESULTS["recall"],
    CV_RESULTS["recall"]
]

f1 = [
    YOLO_RESULTS["f1"],
    ML_RESULTS["f1"],
    CV_RESULTS["f1"]
]

inference_time = [
    YOLO_RESULTS["time_ms"],
    ML_RESULTS["time_ms"],
    CV_RESULTS["time_ms"]
]

# ==========================================
# 1️⃣ ACCURACY BAR PLOT
# ==========================================

plt.figure(figsize=(8,6))
plt.bar(models, accuracy)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0,1)
plt.show()

# ==========================================
# 2️⃣ PRECISION / RECALL / F1
# ==========================================

x = np.arange(len(models))
width = 0.25

plt.figure(figsize=(10,6))
plt.bar(x - width, precision, width, label="Precision")
plt.bar(x, recall, width, label="Recall")
plt.bar(x + width, f1, width, label="F1 Score")

plt.xticks(x, models)
plt.ylabel("Score")
plt.ylim(0,1)
plt.title("Precision / Recall / F1 Comparison")
plt.legend()
plt.show()

# ==========================================
# 3️⃣ INFERENCE TIME
# ==========================================

plt.figure(figsize=(8,6))
plt.bar(models, inference_time)
plt.ylabel("Time (ms)")
plt.title("Inference Time Comparison")
plt.show()

# ==========================================
# 4️⃣ RADAR PLOT
# ==========================================

labels = ["Accuracy", "Precision", "Recall", "F1"]
num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

def add_radar(values, label):
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=label)
    ax.fill(angles, values, alpha=0.1)

fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

add_radar(
    [YOLO_RESULTS["accuracy"], YOLO_RESULTS["precision"],
     YOLO_RESULTS["recall"], YOLO_RESULTS["f1"]],
    "YOLO"
)

add_radar(
    [ML_RESULTS["accuracy"], ML_RESULTS["precision"],
     ML_RESULTS["recall"], ML_RESULTS["f1"]],
    "ML"
)

add_radar(
    [CV_RESULTS["accuracy"], CV_RESULTS["precision"],
     CV_RESULTS["recall"], CV_RESULTS["f1"]],
    "CV"
)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_ylim(0,1)
plt.title("Overall Model Comparison (Radar)")
plt.legend(loc="upper right")
plt.show()
