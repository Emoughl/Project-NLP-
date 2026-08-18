"""
Danh gia ket qua tom tat (Tieu chi 7): do chinh xac + uu, nhuoc diem.

Khong dung ROUGE (theo yeu cau cua co). Do chinh xac duoc do bang chinh
ky thuat da dung trong thuat toan chinh: TF-IDF + Cosine Similarity (chi
sklearn/numpy) giua ban tom tat he thong sinh ra va ban tom tat tham
chieu do con nguoi viet (thu muc DUC_SUM). Diem cang gan 1.0 thi ban tom
tat cua he thong cang "gan" voi ban tham chieu ve mat tu vung/noi dung.

Ngoai diem so, script con tinh ty le nen (compression ratio) de danh gia
muc do co dong, va in ra nhan xet uu/nhuoc diem chung cua phuong phap
TextRank dang ap dung de dua vao bao cao.
"""

from pathlib import Path
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
TEXT_DIR = BASE_DIR / "data" / "DUC_TEXT" / "train"
REFERENCE_DIR = BASE_DIR / "data" / "DUC_SUM"


def read_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def clean_tagged_text(text):
    """Boc cau ra khoi cac the <s>...</s> (dinh dang DUC_TEXT/DUC_SUM) va noi lai."""
    sentences = re.findall(r"<s[^>]*>(.*?)</s>", text, flags=re.DOTALL)
    sentences = [re.sub(r"\s+", " ", s).strip() for s in sentences]
    return " ".join(sentences)


def accuracy_score(reference_text, system_text):
    """Do chinh xac = Cosine Similarity giua vector TF-IDF cua 2 van ban.

    Day KHONG phai la ROUGE (khong do trung khop n-gram tung chu), ma do
    muc do gan nhau ve tu vung/noi dung tong the giua ban tom tat he
    thong sinh ra va ban tom tat tham chieu.
    """
    if not reference_text.strip() or not system_text.strip():
        return 0.0
    try:
        # stop_words="english": loai tu noi (the, a, of, was, said...) de
        # tranh diem bi "bom" ao khi 2 van ban dai deu dung chung nhieu tu
        # noi du noi dung hoan toan khac nhau.
        vectorizer = TfidfVectorizer(
                stop_words="english", # 1.Stop-word Removal (đặc trưng lọc từ dừng)
                ngram_range=(1, 2), # 2. TF-IDF Unigram (đặc trưng từ đơn), 3.TF-IDF Bigram (đặc trưng cụm 2 từ)
                min_df=1
            )
        tfidf = vectorizer.fit_transform([reference_text, system_text])
        return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except ValueError:
        # 2 van ban khong co tu chung nao -> vocabulary rong
        return 0.0


def main():
    results = []

    for summary_file in sorted(OUTPUT_DIR.glob("*_summary.txt")):
        doc_name = summary_file.stem.replace("_summary", "")

        reference_file = REFERENCE_DIR / doc_name
        original_file = TEXT_DIR / doc_name

        if not reference_file.exists():
            print(f"  Bo qua {doc_name}: khong co ban tham chieu trong DUC_SUM.")
            continue

        system_summary = read_file(summary_file).strip()
        reference_summary = clean_tagged_text(read_file(reference_file))

        accuracy = accuracy_score(reference_summary, system_summary)

        compression = None
        if original_file.exists():
            original_text = clean_tagged_text(read_file(original_file))
            original_words = len(original_text.split())
            summary_words = len(system_summary.split())
            if original_words > 0:
                compression = summary_words / original_words

        results.append(
            {"doc": doc_name, "accuracy": accuracy, "compression": compression}
        )

        print("=" * 60)
        print(doc_name)
        print(f"Do chinh xac (Cosine TF-IDF so voi ban tham chieu): {accuracy:.4f}")
        if compression is not None:
            print(f"Ty le nen (so tu tom tat / so tu goc): {compression:.4f}")

    if results:
        avg_accuracy = float(np.mean([r["accuracy"] for r in results]))
        compressions = [r["compression"] for r in results if r["compression"] is not None]
        avg_compression = float(np.mean(compressions)) if compressions else None

        print("\n" + "=" * 60)
        print("TONG KET")
        print("=" * 60)
        print(f"So van ban danh gia: {len(results)}")
        print(f"Do chinh xac trung binh: {avg_accuracy:.4f}")
        if avg_compression is not None:
            print(f"Ty le nen trung binh: {avg_compression:.4f}")

        print(
            """
NHAN XET (Tieu chi 7):

Uu diem:
- Phuong phap TextRank khong can du lieu gan nhan (unsupervised), chay
  nhanh, khong phu thuoc corpus huan luyen lon.
- Viec gop them trong so dac trung (do dai cau, vi tri cau, so lieu,
  viet hoa/ten rieng) vao ma tran do thi giup uu tien cac cau mo dau va
  cau chua thong tin cu the (so lieu, ten rieng) - von thuong la cau
  quan trong trong van ban tin tuc (dataset DUC).
- Do chinh xac trung binh (Cosine TF-IDF) o tren phan anh muc do trung
  lap tu vung giua ban tom tat he thong va ban tom tat tham chieu.

Nhuoc diem:
- Day la phuong phap trich xuat (extractive) - chi chon nguyen cau co
  san chu khong viet lai (abstractive), nen doi khi cau duoc chon thieu
  mach lac khi ghep lai voi nhau.
- Do chinh xac do bang Cosine TF-IDF chi phan anh su trung lap tu vung
  be mat, khong hieu duoc ngu nghia sau (hai cau dien dat cung y nhung
  khac tu se bi cham thap).
- Ket qua phu thuoc nhieu vao nguong so cau trich chon (top_n) va trong
  so alpha khi gop dac trung vao do thi - can thu nghiem de chon gia
  tri phu hop voi tung loai van ban.
"""
        )
    else:
        print("Khong tim thay ban tom tat nao trong thu muc output/ de danh gia.")
        print("Hay chay `python main.py` truoc de sinh ban tom tat.")


if __name__ == "__main__":
    main()
