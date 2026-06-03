#!/usr/bin/env python3
"""Buat flowchart pipeline (WAJIB di soal) -> flowchart.png & flowchart.pdf"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C = dict(data=("#E3F2FD", "#1565C0"), proc=("#E8F5E9", "#2E7D32"),
         model=("#FFF3E0", "#E65100"), eval=("#F3E5F5", "#6A1B9A"),
         out=("#ECEFF1", "#37474F"))
TXT = "#212121"

def box(ax, x, y, w, h, text, kind, fs=9, bold=False):
    face, edge = C[kind]
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.05",
                 lw=1.6, edgecolor=edge, facecolor=face, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=TXT,
            zorder=3, weight="bold" if bold else "normal")

def arrow(ax, p1, p2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                 lw=1.5, color="#546E7A", zorder=1))

fig, ax = plt.subplots(figsize=(11, 8.5))
ax.set_xlim(0, 12); ax.set_ylim(0, 13); ax.axis("off")
fig.patch.set_facecolor("white")

ax.text(6, 12.5, "NoLimit DS Test - Analisis Sentimen Bahasa Indonesia",
        ha="center", fontsize=14, weight="bold", color=TXT)
ax.text(6, 12.0, "Model Hugging Face + Embeddings (FAISS) | Task A: Classification",
        ha="center", fontsize=9.5, color="#607D8B")

box(ax, 6, 11.0, 5.6, 0.8, "Dataset: NusaX-senti (subset 'ind')\ntext, label -> positive / neutral / negative", "data")
box(ax, 6, 9.7, 5.6, 0.8, "Preprocessing (preprocess.py)\nrapikan spasi, buang teks terlalu pendek", "proc")
box(ax, 6, 8.5, 4.6, 0.7, "Split resmi: train (500) / test (400)", "proc")
arrow(ax, (6, 10.6), (6, 10.1)); arrow(ax, (6, 9.3), (6, 8.85))

yb = 6.6
box(ax, 2.3, yb, 3.0, 0.95, "Embedding kalimat\n(HF sentence-transformers\nIndoBERT)", "model", 8)
box(ax, 2.3, yb-1.5, 3.0, 0.75, "FAISS index (cosine)\nk-NN voting", "model", 8.5)
box(ax, 6, yb, 3.0, 0.95, "Embedding kalimat\n(vektor IndoBERT\nyang sama)", "model", 8.5)
box(ax, 6, yb-1.5, 3.0, 0.75, "Logistic Regression\n(scikit-learn)", "model", 8.5)
box(ax, 9.7, yb, 3.0, 0.95, "Transformer classifier\n(HF pipeline, model\nsentimen pretrained)", "model", 8)
box(ax, 9.7, yb-1.5, 3.0, 0.75, "Klasifikasi langsung\n(tanpa latih ulang)", "model", 8.5)

for x in (2.3, 6, 9.7):
    arrow(ax, (6, 8.15), (x, yb + 0.5))
    arrow(ax, (x, yb - 0.5), (x, yb - 1.1))

box(ax, 6, 2.9, 7.2, 0.85, "Evaluasi (evaluate.py)\nAccuracy | Macro-F1 | classification report | confusion matrix", "eval", 9)
arrow(ax, (2.3, yb-1.9), (5.0, 3.32)); arrow(ax, (6, yb-1.9), (6, 3.32)); arrow(ax, (9.7, yb-1.9), (7.0, 3.32))

box(ax, 3.2, 1.4, 4.6, 0.85, "Output\npred_*.csv | cm_*.png | metrics.json", "out", 8.5)
box(ax, 8.6, 1.4, 4.6, 0.85, "Bonus: Streamlit app (app.py)\nprediksi live + tetangga FAISS", "out", 8.5)
arrow(ax, (5.2, 2.47), (3.6, 1.85)); arrow(ax, (6.8, 2.47), (8.2, 1.85))

fig.tight_layout()
fig.savefig("flowchart.png", dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig("flowchart.pdf", bbox_inches="tight", facecolor="white")
print("OK -> flowchart.png & flowchart.pdf")