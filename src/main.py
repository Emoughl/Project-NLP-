from pathlib import Path
from preprocess import get_all_files, read_text, split_sentences
from textrank import apply_feature_weights, calculate_pagerank_numpy, generate_summary
from vector import build_tfidf_matrix, calculate_similarity_matrix, extract_additional_features

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

files = get_all_files()

print("=" * 60)
print("  HỆ THỐNG TÓM TẮT VĂN BẢN TỰ ĐỘNG BẰNG THUẬT TOÁN TEXTRANK  ")
print("=" * 60)

for input_file in files:
    print(f"\nĐang xử lý: {input_file}")

    # Bước 1: Đọc dữ liệu
    text = read_text(input_file)

    # Bước 2: Tiền xử lý & Tách câu
    sentences = split_sentences(text)

    if len(sentences) == 0:
        continue

    # Bước 3 & 4: Biểu diễn TF-IDF & Tính Ma trận tương đồng Cosine
    try:
        tfidf_matrix, vectorizer = build_tfidf_matrix(sentences)
        similarity_matrix = calculate_similarity_matrix(tfidf_matrix)
    except ValueError:
        print(
            f"  Bỏ qua {input_file}: không đủ từ lặp lại giữa các câu để tính TF-IDF."
        )
        continue

    # Bước 4b: Trích xuất các đặc trưng bổ sung (độ dài câu, vị trí câu,
    # chứa số liệu, chữ hoa/tên riêng) — Tiêu chí 4 (>=5 đặc trưng)
    extra_features = extract_additional_features(sentences)

    # Bước 4c: Cải tiến đồ thị bằng cách gộp trọng số đặc trưng vào
    # ma trận tương đồng Cosine — Tiêu chí 8 (cải tiến phương pháp)
    similarity_matrix = apply_feature_weights(similarity_matrix, extra_features)

    # Bước 5: Xếp hạng câu bằng TextRank (PageRank) trên đồ thị đã cải tiến
    scores = calculate_pagerank_numpy(similarity_matrix)

    # Bước 6: Trích xuất bản tóm tắt (top 18 câu)
    summary = generate_summary(sentences, scores, top_n=18)

    # Ghi kết quả ra thư mục output
    output_file = OUTPUT_DIR / f"{Path(input_file).stem}_summary.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        if isinstance(summary, str):
            f.write(summary)
        else:
            for sentence in summary:
                f.write(sentence + "\n")

    print(f"  -> Đã lưu bản tóm tắt tại: {output_file.name}")

print("\nFinished!")
