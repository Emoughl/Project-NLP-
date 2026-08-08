from preprocess import read_text, split_sentences, get_all_files
from vector import build_tfidf, build_similarity_matrix, sentence_length_scores, numeric_content_scores
from textrank import rank_sentences, generate_summary
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

files = get_all_files()

for input_file in files:

    print(f"Đang xử lý: {input_file}")

    #Step 1: Read data from the DUC_TEXT dataset
    text = read_text(input_file)

    #Step 2: Text Preprocessing and Sentence Segmentation
    sentences = split_sentences(text)

    if len(sentences) == 0:
        continue

    #Step 3: Represent sentences using TF-IDF
    tfidf_matrix, vectorizer = build_tfidf(sentences)

    #Step 4: Calculate similarity between sentences
    similarity_matrix, raw_similarity_matrix = build_similarity_matrix(
        tfidf_matrix, return_raw=True
    )

    #Step 5: Rank sentences using TextRank algorithm
    scores = rank_sentences(similarity_matrix)

    #Step 5b: Combine PageRank score with Feature 4 and Feature 5
    len_scores = sentence_length_scores(sentences)
    num_scores = numeric_content_scores(sentences)

    combined_scores = {
        idx: (
            0.8 * sc
            + 0.1 * len_scores[idx]
            + 0.1 * num_scores[idx]
        )
        for idx, sc in scores.items()
    }

    #Step 6: Generate summary
    summary = generate_summary(
        sentences,
        combined_scores,
        raw_similarity_matrix,
        top_n=18
    )

    output_file = OUTPUT_DIR / f"{Path(input_file).stem}_summary.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in summary:
            f.write(sentence + "\n")

print("\nFinished!")