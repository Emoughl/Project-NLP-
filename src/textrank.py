import numpy as np


def build_stochastic_matrix(sim_matrix):
    #Chuẩn hóa ma trận tương đồng thành ma trận chuyển trạng thái (Stochastic Matrix).
    #Mỗi hàng chuẩn hóa tổng bằng 1. Câu cô lập (dangling node) được phân phối đều 1/n để tránh thất thoát tổng điểm PageRank.

    n = sim_matrix.shape[0]
    row_sums = sim_matrix.sum(axis=1)
    dangling = row_sums == 0

    stochastic_matrix = np.zeros_like(sim_matrix, dtype=float)

    # Câu có cạnh: chuẩn hóa tổng hàng = 1
    if np.any(~dangling):
        stochastic_matrix[~dangling] = (
            sim_matrix[~dangling] / row_sums[~dangling][:, np.newaxis]
        )

    # Câu cô lập: phân phối đều
    if np.any(dangling):
        stochastic_matrix[dangling] = 1.0 / n

    return stochastic_matrix.T


def calculate_pagerank_numpy(
    sim_matrix, damping_factor=0.85, max_iter=100, tol=1e-6
):
    #Tính điểm PageRank bằng Power Iteration
    n = sim_matrix.shape[0]
    if n == 0:
        return {}

    M = build_stochastic_matrix(sim_matrix)
    scores = np.full(n, 1.0 / n)
    damping_vector = np.full(n, (1.0 - damping_factor) / n)

    # Power Iteration: R_new = d * M * R_old + (1-d)/N
    for _ in range(max_iter):
        prev_scores = scores.copy()
        scores = damping_factor * np.dot(M, prev_scores) + damping_vector

        if np.linalg.norm(scores - prev_scores, ord=1) < tol:
            break

    return {i: score for i, score in enumerate(scores)}


def apply_feature_weights(sim_matrix, extra_features, alpha=0.7):
    # Gộp trọng số đặc trưng bổ sung vào ma trận Cosine Similarity
    # CT: enhanced_sim = alpha * cosine_sim + (1 - alpha) * feature_matrix
    num_sentences = len(extra_features)
    if num_sentences == 0:
        return sim_matrix

    feature_scores = np.mean(extra_features, axis=1)

    # Ma trận trọng số đặc trưng giữa các cặp câu
    feature_matrix = np.zeros_like(sim_matrix)
    for i in range(num_sentences):
        for j in range(num_sentences):
            if i != j:
                feature_matrix[i][j] = (
                    feature_scores[i] + feature_scores[j]
                ) / 2.0

    enhanced_sim_matrix = alpha * sim_matrix + (1 - alpha) * feature_matrix
    np.fill_diagonal(enhanced_sim_matrix, 0)
    return enhanced_sim_matrix


def generate_summary(sentences, scores, top_n=3):
    #Chọn top_n câu điểm PageRank cao nhất, sắp xếp theo thứ tự gốc
    if not sentences or not scores:
        return ""

    top_n = min(top_n, len(sentences))

    ranked_sentences = sorted(
        scores.items(), key=lambda x: x[1], reverse=True
    )
    top_indices = sorted(idx for idx, _ in ranked_sentences[:top_n])

    return "\n".join(sentences[i] for i in top_indices)


# --- KIỂM TRA TRỰC TIẾP ---
if __name__ == "__main__":
    from preprocess import get_all_files, read_text, split_sentences
    from vector import (
        build_tfidf_matrix,
        calculate_similarity_matrix,
        extract_additional_features,
    )

    files = get_all_files()
    sentences = split_sentences(read_text(files[0]))
    print(f"--- Kiểm tra textrank.py với {len(sentences)} câu ---")

    tfidf_matrix, _ = build_tfidf_matrix(sentences)
    sim_matrix = calculate_similarity_matrix(tfidf_matrix)
    extra_features = extract_additional_features(sentences)

    enhanced_sim = apply_feature_weights(sim_matrix, extra_features)
    scores = calculate_pagerank_numpy(enhanced_sim)
    summary = generate_summary(sentences, scores, top_n=3)

    print("\n--- BẢN TÓM TẮT (3 CÂU) ---")
    print(summary)
