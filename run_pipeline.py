#!/usr/bin/env python3
"""Pipeline analisis sentimen Indonesia (NusaX) end-to-end.
Latih di data/nusax_train.csv, evaluasi di data/nusax_test.csv.

Contoh:
  python run_pipeline.py --method all
  python run_pipeline.py --method knn --predict "pelayanannya lambat sekali"
"""
import argparse
import json
from pathlib import Path
import pandas as pd

from preprocess import load_clean, clean_text
from embeddings import load_model
from evaluate import evaluate, plot_confusion
from models import (EmbeddingKNNClassifier, EmbeddingLogRegClassifier,
                    TransformerClassifier)

OUT = Path("outputs")
CHOICES = ["knn", "logreg", "transformer", "all"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/nusax_train.csv")
    p.add_argument("--test", default="data/nusax_test.csv")
    p.add_argument("--method", default="all", choices=CHOICES)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--predict", default=None, help="Satu teks untuk diklasifikasi.")
    return p.parse_args()

def bar(t):
    print("\n" + "=" * 60 + f"\n{t}\n" + "=" * 60)

def main():
    args = parse_args()
    OUT.mkdir(exist_ok=True)

    bar("1. Muat & bersihkan data")
    train = load_clean(args.train)
    test = load_clean(args.test)
    print(f"train={len(train)}  test={len(test)}")
    print(train["label"].value_counts().to_string())

    methods = ["knn", "logreg", "transformer"] if args.method == "all" else [args.method]

    # Model embedding dimuat sekali, dipakai k-NN & logreg.
    emb_model = None
    if any(m in ("knn", "logreg") for m in methods):
        bar("2. Muat model embedding (IndoBERT)")
        emb_model = load_model()

    fitted, results = {}, {}
    for method in methods:
        bar(f"3. Metode: {method}")
        if method == "knn":
            clf = EmbeddingKNNClassifier(emb_model, k=args.k)
            clf.fit(train["text"].tolist(), train["label"].tolist())
            preds = clf.predict(test["text"].tolist())
        elif method == "logreg":
            clf = EmbeddingLogRegClassifier(emb_model)
            clf.fit(train["text"].tolist(), train["label"].tolist())
            preds = clf.predict(test["text"].tolist())
        else:  # transformer
            clf = TransformerClassifier()
            preds = clf.predict(test["text"].tolist())

        fitted[method] = clf
        m = evaluate(test["label"].tolist(), preds)
        results[method] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
        print(f"accuracy = {m['accuracy']:.3f} | macro-F1 = {m['macro_f1']:.3f}")
        print(m["report"])

        out = test.copy(); out["prediction"] = preds
        out[["text", "label", "prediction"]].to_csv(OUT / f"pred_{method}.csv", index=False)
        plot_confusion(test["label"].tolist(), preds, OUT / f"cm_{method}.png",
                       f"Confusion - {method}")
        print(f"tersimpan: outputs/pred_{method}.csv, outputs/cm_{method}.png")

    bar("4. Ringkasan")
    print(pd.DataFrame(results).T.round(3).to_string())
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))

    if args.predict:
        bar("5. Prediksi teks baru")
        clf = fitted[methods[0]]
        text = clean_text(args.predict)
        print(f'[{clf.predict([text])[0]}]  {args.predict}')

if __name__ == "__main__":
    main()