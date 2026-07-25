from preprocess import read_text, split_sentences
from vector import build_tfidf ,build_similarity_matrix
from textrank import rank_sentences, generate_summary
from pathlib import Path

text = read_text("AI.txt")

sentences = split_sentences(text)

tfidf_matrix, vectorizer = build_tfidf(sentences)

similarity_matrix = build_similarity_matrix(tfidf_matrix)

scores = rank_sentences(similarity_matrix, sentences)

summary = generate_summary(sentences, scores, top_n=3)


print("1.Số câu:", len(sentences))
print("2. Kích thước TF-IDF:", tfidf_matrix.shape)
print("3. Kích thước Similarity Matrix:", similarity_matrix.shape)
print("4. Điểm xếp hạng:", scores)
print("\n===== TÓM TẮT =====")

for sentence in summary:
    print("-", sentence)

#Xuất output ra file
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

output_file = OUTPUT_DIR / "ai_summary.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write("===== TÓM TẮT =====\n\n")

    for sentence in summary:
        f.write("- " + sentence + "\n")