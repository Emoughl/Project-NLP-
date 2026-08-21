#Đánh giá kết quả tóm tắt: độ chính xác (Cosine TF-IDF) + nhận xét ưu/nhược điểm
#Đo bằng Cosine Similarity trên vector TF-IDF giữa bản tóm tắt hệ thống và bản tham chiếu (DUC_SUM). Điểm càng gần 1.0 = càng giống tham chiếu

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
    sentences = re.findall(r"<s[^>]*>(.*?)</s>", text, flags=re.DOTALL)
    sentences = [re.sub(r"\s+", " ", s).strip() for s in sentences]
    return " ".join(sentences)


def accuracy_score(reference_text, system_text):
    # Cosine Similarity giữa vector TF-IDF của 2 văn bản.
    # Cùng config TfidfVectorizer với vector.py (stop_words, ngram 1-2).
    if not reference_text.strip() or not system_text.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1
        )
        tfidf = vectorizer.fit_transform([reference_text, system_text])
        return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except ValueError:
        return 0.0


def main():
    results = []

    for summary_file in sorted(OUTPUT_DIR.glob("*_summary.txt")):
        doc_name = summary_file.stem.replace("_summary", "")

        reference_file = REFERENCE_DIR / doc_name
        original_file = TEXT_DIR / doc_name

        if not reference_file.exists():
            print(f"  Bỏ qua {doc_name}")
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
        print(f"Độ chính xác (Cosine TF-IDF so với bản tham chiếu): {accuracy:.4f}")
        if compression is not None:
            print(f"Tỷ lệ nén (số từ tóm tắt / số từ gốc): {compression:.4f}")

    if results:
        avg_accuracy = float(np.mean([r["accuracy"] for r in results]))
        compressions = [r["compression"] for r in results if r["compression"] is not None]
        avg_compression = float(np.mean(compressions)) if compressions else None

        print("TỔNG KẾT")
        print(f"Số văn bản đánh giá: {len(results)}")
        print(f"Độ chính xác trung bình: {avg_accuracy:.4f}")
        if avg_compression is not None:
            print(f"Tỷ lệ nén trung bình: {avg_compression:.4f}")
    else:
        print("Không tìm thấy file nào trong thư mục output để đánh giá")
        print("chạy`python main.py` để có file tóm tắt")