# Đánh giá kết quả tóm tắt TextRank (Top node T = 18)
# Metrics: Precision, Recall, F1
#
# Chỉ những văn bản có bản tóm tắt tham chiếu (DUC_SUM) không rỗng mới được
# chấm; số văn bản bị bỏ qua được in ra cuối để minh bạch.

from pathlib import Path
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
REFERENCE_DIR = BASE_DIR / "data" / "DUC_SUM"
TRAIN_DIR = BASE_DIR / "data" / "DUC_TEXT" / "train"
TEST_DIR = BASE_DIR / "data" / "DUC_TEXT" / "test"


def read_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def split_sentences(text):
    """Tách câu từ thẻ <s>...</s>."""
    sentences = re.findall(r"<s[^>]*>(.*?)</s>", text, flags=re.DOTALL)
    return [re.sub(r"\s+", " ", s).strip() for s in sentences if s.strip()]


def find_source_file(doc_name):
    """Tìm file nguồn trong train/ hoặc test/."""
    for d in [TRAIN_DIR, TEST_DIR]:
        p = d / doc_name
        if p.exists():
            return p
    return None


def count_matched(system_sents, reference_sents, threshold=0.3):
    """Đếm số câu trích được (đúng) bằng cosine similarity."""
    if not system_sents or not reference_sents:
        return 0

    all_sents = system_sents + reference_sents
    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        tfidf = vectorizer.fit_transform(all_sents)
    except ValueError:
        return 0

    n_sys = len(system_sents)
    sim = cosine_similarity(tfidf[:n_sys], tfidf[n_sys:])

    matched = 0
    used = set()
    for i in range(n_sys):
        best_j, best_score = -1, -1
        for j in range(len(reference_sents)):
            if j not in used and sim[i][j] > best_score:
                best_score = sim[i][j]
                best_j = j
        if best_j >= 0 and best_score >= threshold:
            matched += 1
            used.add(best_j)

    return matched


def main():
    all_prec, all_rec, all_f1 = [], [], []
    skipped = []

    for summary_file in sorted(OUTPUT_DIR.glob("*_summary.txt")):
        doc_name = summary_file.stem.replace("_summary", "")

        reference_file = REFERENCE_DIR / doc_name
        if not reference_file.exists() or reference_file.stat().st_size == 0:
            skipped.append((doc_name, "thiếu/rỗng bản tham chiếu DUC_SUM"))
            continue

        source_file = find_source_file(doc_name)
        if source_file is None:
            skipped.append((doc_name, "không tìm thấy văn bản nguồn"))
            continue

        source_sents = split_sentences(read_file(source_file))
        ref_sents = split_sentences(read_file(reference_file))
        sys_text = read_file(summary_file).strip()
        sys_sents = [s.strip() for s in sys_text.split("\n") if s.strip()]

        if not source_sents or not ref_sents or not sys_sents:
            skipped.append((doc_name, "không tách được câu"))
            continue

        correct = count_matched(sys_sents, ref_sents)

        # Precision = #câu trích được (đúng) / #câu trích được
        prec = correct / len(sys_sents) if sys_sents else 0

        # Recall = #câu trích được (đúng) / #câu gán nhãn (câu tóm tắt)
        rec = correct / len(ref_sents) if ref_sents else 0

        # F1 = 2 * (P * R) / (P + R)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        all_prec.append(prec)
        all_rec.append(rec)
        all_f1.append(f1)

        print(f"  {doc_name:12s}  P={prec:.3f}  R={rec:.3f}  F1={f1:.1%}  (Giữa 2 file output có {correct} đúng / tổng {len(sys_sents)})")

    total = len(all_prec) + len(skipped)

    if all_prec:
        print(f"\n  TRUNG BÌNH ({len(all_prec)}/{total} văn bản được chấm, Top T = 18):")
        print(f"    Precision = {np.mean(all_prec):.4f}  ({np.mean(all_prec)*100:.1f}%)")
        print(f"    Recall    = {np.mean(all_rec):.4f}  ({np.mean(all_rec)*100:.1f}%)")
        print(f"    F1        = {np.mean(all_f1):.4f}  ({np.mean(all_f1)*100:.1f}%)")

    if skipped:
        print(f"\n  BỎ QUA {len(skipped)}/{total} văn bản:")
        for doc_name, reason in skipped:
            print(f"    {doc_name:12s}  — {reason}")


if __name__ == "__main__":
    main()
