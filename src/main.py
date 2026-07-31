from preprocess import read_text, split_sentences, get_all_files
from vector import build_tfidf, build_similarity_matrix
from textrank import rank_sentences, generate_summary
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

files = get_all_files()

print(f"Tổng số file: {len(files)}")

for input_file in files:

    print(f"Đang xử lý: {input_file}")

    # Bước 1: Đọc dữ liệu từ tập DUC_TEXT
    text = read_text(input_file)

    # Bước 2: Tiền xử lý, tách văn bản thành các câu
    sentences = split_sentences(text)

    if len(sentences) == 0:
        continue

    # Bước 3: Biểu diễn câu bằng TF-IDF
    tfidf_matrix, vectorizer = build_tfidf(sentences)

    # Bước 4: Tính độ tương đồng giữa các câu
    similarity_matrix = build_similarity_matrix(tfidf_matrix)

    # Bước 5: Xếp hạng câu bằng thuật toán TextRank
    scores = rank_sentences(similarity_matrix)

    # Bước 6: Sinh bản tóm tắt
    summary = generate_summary(
        sentences,
        scores,
        similarity_matrix,
        top_n=18
    )

    output_file = OUTPUT_DIR / f"{Path(input_file).stem}_summary.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in summary:
            f.write(sentence + "\n")

print("\nHoàn thành!")