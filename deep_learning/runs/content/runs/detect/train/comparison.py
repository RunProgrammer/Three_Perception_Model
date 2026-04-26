import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 🔧 REPLACE WITH YOUR REAL RESULTS
# ==========================================

YOLO = {
    "accuracy": 0.89,
    "precision": 0.88,
    "recall": 0.87,
    "f1": 0.88,
    "time_ms": 28
}

ML = {
    "accuracy": 0.76,
    "precision": 0.72,
    "recall": 0.70,
    "f1": 0.71,
    "time_ms": 4
}

CV = {
    "accuracy": 0.61,
    "precision": 0.55,
    "recall": 0.52,
    "f1": 0.53,
    "time_ms": 2
}

models = ["YOLO (CNN)", "ML", "Classical CV"]

# ==========================================
# METRIC ARRAYS
# ==========================================

accuracy = [YOLO["accuracy"], ML["accuracy"], CV["accuracy"]]
precision = [YOLO["precision"], ML["precision"], CV["precision"]]
recall = [YOLO["recall"], ML["recall"], CV["recall"]]
f1 = [YOLO["f1"], ML["f1"], CV["f1"]]
time = [YOLO["time_ms"], ML["time_ms"], CV["time_ms"]]

# ==========================================
# 1️⃣ COMBINED METRIC GRAPH
# ==========================================

x = np.arange(len(models))
width = 0.2

plt.figure(figsize=(10,6))

plt.bar(x - 1.5*width, accuracy, width, label="Accuracy")
plt.bar(x - 0.5*width, precision, width, label="Precision")
plt.bar(x + 0.5*width, recall, width, label="Recall")
plt.bar(x + 1.5*width, f1, width, label="F1 Score")

plt.xticks(x, models)
plt.ylabel("Score")
plt.ylim(0, 1)
plt.title("Comparison of Detection Performance")
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# 2️⃣ INFERENCE TIME GRAPH
# ==========================================

plt.figure(figsize=(8,5))
plt.bar(models, time)
plt.ylabel("Inference Time (ms)")
plt.title("Inference Time Comparison")
plt.tight_layout()
plt.show()

# ==========================================
# 3️⃣ RADAR PLOT (Best for Paper)
# ==========================================

labels = ["Accuracy", "Precision", "Recall", "F1"]
num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

def add_model(values, label):
    values = values + values[:1]
    ax.plot(angles, values, linewidth=2, label=label)
    ax.fill(angles, values, alpha=0.1)

fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

add_model([YOLO["accuracy"], YOLO["precision"], YOLO["recall"], YOLO["f1"]], "YOLO")
add_model([ML["accuracy"], ML["precision"], ML["recall"], ML["f1"]], "ML")
add_model([CV["accuracy"], CV["precision"], CV["recall"], CV["f1"]], "CV")

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_ylim(0, 1)
plt.title("Overall Model Comparison (Radar Plot)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
