from preprocess import read_text, split_sentences
from vector import build_tfidf

text = read_text("AI.txt")

sentences = split_sentences(text)

tfidf_matrix, vectorizer = build_tfidf(sentences)

print("Số câu:", len(sentences))
print("Kích thước TF-IDF:", tfidf_matrix.shape)