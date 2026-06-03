"""Pembersihan teks ringan & sadar-konteks untuk data sentimen NusaX.

Datanya sudah bersih (tanpa URL/mention/hashtag/emoji) dan akan diberi ke
model transformer, jadi kita SENGAJA tidak lowercase / tidak buang tanda baca:
kapitalisasi & '!' '?' membawa sinyal emosi yang berguna.
"""
import re
import pandas as pd

_SPASI = re.compile(r"\s+")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("\t", " ")  # samakan whitespace
    text = _SPASI.sub(" ", text).strip()                # rapikan spasi berlebih
    return text

def load_clean(path: str) -> pd.DataFrame:
    """Muat CSV, bersihkan ringan, buang baris teks kosong/terlalu pendek."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).map(clean_text)
    df = df[df["text"].str.split().str.len() >= 2].reset_index(drop=True)
    return df

# Uji cepat saat file ini dijalankan langsung
if __name__ == "__main__":
    df = load_clean("data/nusax_train.csv")
    print("Setelah bersih:", len(df), "baris")
    print("Contoh:", df["text"].iloc[0])