"""Evaluasi: accuracy, macro-F1, classification report, confusion matrix (PNG)."""
from pathlib import Path
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

LABELS = ["negative", "neutral", "positive"]

def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "report": classification_report(y_true, y_pred, labels=LABELS, zero_division=0, digits=3),
    }

def plot_confusion(y_true, y_pred, out_path, title=""):
    import matplotlib
    matplotlib.use("Agg")            # mode tanpa layar, aman buat menyimpan file
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)), labels=LABELS)
    ax.set_yticks(range(len(LABELS)), labels=LABELS)
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Sebenarnya")
    ax.set_title(title or "Confusion Matrix")
    thr = cm.max() / 2 if cm.max() else 0.5
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path