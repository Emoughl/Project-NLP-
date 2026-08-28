import sys
from pathlib import Path
from preprocess import get_all_files, read_text, split_sentences
from textrank import (
    calculate_pagerank_numpy,
    generate_summary,
    generate_summary_improved,
)
from vector import build_tfidf_matrix, calculate_similarity_matrix

BASE_DIR = Path(__file__).resolve().parent.parent

# Run "python main.py --improved" cho PHƯƠNG PHÁP CẢI TIẾN
USE_IMPROVED = "--improved" in sys.argv

OUTPUT_DIR = BASE_DIR / ("output_improved" if USE_IMPROVED else "output")
OUTPUT_DIR.mkdir(exist_ok=True)

files = get_all_files()

if USE_IMPROVED:
    print("TÓM TẮT VĂN BẢN — TEXTRANK CẢI TIẾN (đặc trưng vị trí + MMR)")
else:
    print("TÓM TẮT VĂN BẢN TỰ ĐỘNG BẰNG THUẬT TOÁN TEXTRANK")

for input_file in files:
    print(f"\nĐang xử lý: {input_file}")

    # B1: Đọc dữ liệu 
    # B2: Tách câu
    text = read_text(input_file)
    sentences = split_sentences(text)

    if len(sentences) == 0:
        continue

    # B3: Biểu diễn TF-IDF 
    # B4: Tính ma trận tương đồng Cosine
    try:
        tfidf_matrix, vectorizer = build_tfidf_matrix(sentences)
        similarity_matrix = calculate_similarity_matrix(tfidf_matrix)
    except ValueError:
        print(
            f"  Bỏ qua {input_file}"
        )
        continue

    # B5: Xếp hạng câu bằng PageRank 
    # B6: Trích xuất tóm tắt
    scores = calculate_pagerank_numpy(similarity_matrix)

    if USE_IMPROVED:
        summary = generate_summary_improved(
            sentences, scores, similarity_matrix, top_n=18
        )
    else:
        summary = generate_summary(sentences, scores, top_n=18)

    # Ghi kết quả
    output_file = OUTPUT_DIR / f"{Path(input_file).stem}_summary.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"Đã lưu bản tóm tắt : {output_file.name}")

print("\nHoàn Thành!")
