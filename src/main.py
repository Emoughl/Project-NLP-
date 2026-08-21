from pathlib import Path
from preprocess import get_all_files, read_text, split_sentences
from textrank import apply_feature_weights, calculate_pagerank_numpy, generate_summary
from vector import build_tfidf_matrix, calculate_similarity_matrix, extract_additional_features

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

files = get_all_files()

print("TÓM TẮT VĂN BẢN TỰ ĐỘNG BẰNG THUẬT TOÁN TEXTRANK")

for input_file in files:
    print(f"\nĐang xử lý: {input_file}")

    # Đọc dữ liệu & tách câu
    text = read_text(input_file)
    sentences = split_sentences(text)

    if len(sentences) == 0:
        continue

    # Biểu diễn TF-IDF & tính ma trận tương đồng Cosine
    try:
        tfidf_matrix, vectorizer = build_tfidf_matrix(sentences)
        similarity_matrix = calculate_similarity_matrix(tfidf_matrix)
    except ValueError:
        print(
            f"  Bỏ qua {input_file}"
        )
        continue

    # Trích xuất đặc trưng bổ sung & gộp trọng số vào đồ thị
    extra_features = extract_additional_features(sentences)
    similarity_matrix = apply_feature_weights(similarity_matrix, extra_features)

    # Xếp hạng câu bằng PageRank & trích xuất tóm tắt
    scores = calculate_pagerank_numpy(similarity_matrix)
    summary = generate_summary(sentences, scores, top_n=18)

    # Ghi kết quả
    output_file = OUTPUT_DIR / f"{Path(input_file).stem}_summary.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"Đã lưu bản tóm tắt tại: {output_file.name}")

print("\nFinished!")
