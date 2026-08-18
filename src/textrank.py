import numpy as np


def build_stochastic_matrix(sim_matrix):
    """Biến ma trận tương đồng (N x N) thành Ma trận chuyển trạng thái (Stochastic Matrix).

    Mỗi hàng được chuẩn hóa để tổng bằng 1. Nếu một câu hoàn toàn cô lập
    (không có cạnh nào), nó sẽ được phân phối ĐỀU (1/n) cho mọi câu khác
    thay vì bỏ trống hàng đó — đúng cách xử lý "dangling node" chuẩn của
    PageRank, tránh làm thất thoát tổng điểm PageRank qua mỗi vòng lặp.
    """
    n = sim_matrix.shape[0]

    # Tính tổng theo hàng: mỗi hàng i biểu diễn trọng số các cạnh đi ra từ câu i
    row_sums = sim_matrix.sum(axis=1)
    dangling = row_sums == 0

    stochastic_matrix = np.zeros_like(sim_matrix, dtype=float)

    # Các câu có cạnh: chuẩn hóa để tổng hàng bằng 1
    if np.any(~dangling):
        stochastic_matrix[~dangling] = (
            sim_matrix[~dangling] / row_sums[~dangling][:, np.newaxis]
        )

    # Câu cô lập (dangling node): phân phối đều 1/n cho mọi câu khác
    if np.any(dangling):
        stochastic_matrix[dangling] = 1.0 / n

    # Chuyển vị ma trận để phục vụ cho phép nhân vector PageRank
    return stochastic_matrix.T


def calculate_pagerank_numpy(
    sim_matrix, damping_factor=0.85, max_iter=100, tol=1e-6
):
    """Chạy thuật toán PageRank bằng thuật toán Lặp Lũy Thừa (Power Iteration)

    Chỉ sử dụng NumPy thuần túy, không dùng networkx.
    """
    n = sim_matrix.shape[0]
    if n == 0:
        return {}

    # 1. Tạo ma trận chuyển trạng thái M
    M = build_stochastic_matrix(sim_matrix)

    # 2. Khởi tạo vector điểm PageRank ban đầu bằng nhau cho mọi câu (1/N)
    scores = np.full(n, 1.0 / n)

    # Vector phân bố ngẫu nhiên (Damping vector)
    damping_vector = np.full(n, (1.0 - damping_factor) / n)

    # 3. Vòng lặp Lũy thừa (Power Iteration)
    for _ in range(max_iter):
        prev_scores = scores.copy()

        # Công thức PageRank: R_new = d * M * R_old + (1-d)/N
        scores = damping_factor * np.dot(M, prev_scores) + damping_vector

        # Kiểm tra điều kiện hội tụ (nếu khoảng cách thay đổi nhỏ hơn threshold tol thì dừng)
        if np.linalg.norm(scores - prev_scores, ord=1) < tol:
            break

    # Trả về kết quả dạng dictionary: {chỉ_số_câu: điểm_pagerank}
    return {i: score for i, score in enumerate(scores)}


def apply_feature_weights(sim_matrix, extra_features, alpha=0.7):
    """Cải tiến đồ thị (Tiêu chí 8): Kết hợp Cosine Similarity từ TF-IDF với Ma

    trận Đặc trưng bổ sung.
    """
    num_sentences = len(extra_features)
    if num_sentences == 0:
        return sim_matrix

    # Tính điểm đặc trưng trung bình của từng câu
    feature_scores = np.mean(extra_features, axis=1)

    # Tạo ma trận thưởng dựa trên đặc trưng giữa các cặp câu
    feature_matrix = np.zeros_like(sim_matrix)
    for i in range(num_sentences):
        for j in range(num_sentences):
            if i != j:
                feature_matrix[i][j] = (
                    feature_scores[i] + feature_scores[j]
                ) / 2.0

    # Ma trận tương đồng mới kết hợp cả TF-IDF và các đặc trưng
    enhanced_sim_matrix = alpha * sim_matrix + (1 - alpha) * feature_matrix
    np.fill_diagonal(enhanced_sim_matrix, 0)

    return enhanced_sim_matrix


def generate_summary(sentences, scores, top_n=3):
    """Chọn top_n câu có điểm PageRank cao nhất và sắp xếp lại theo đúng thứ tự

    xuất hiện gốc trong văn bản.
    """
    if not sentences or not scores:
        return ""

    top_n = min(top_n, len(sentences))

    # 1. Sắp xếp danh sách câu theo điểm PageRank giảm dần
    ranked_sentences = sorted(
        scores.items(), key=lambda x: x[1], reverse=True
    )

    # 2. Lấy ra top_n index của các câu có điểm cao nhất
    top_indices = [idx for idx, score in ranked_sentences[:top_n]]

    # 3. Sắp xếp các index lại theo thứ tự ban đầu trong bài báo (logic dòng thời gian)
    top_indices.sort()

    # 4. Ghép các câu được chọn thành đoạn văn tóm tắt
    summary = "\n".join([sentences[i] for i in top_indices])
    return summary


# --- KIỂM TRA TRỰC TIẾP FILE TEXTRANK.PY ---
if __name__ == "__main__":
    from preprocess import get_all_files, read_text, split_sentences
    from vector import (
        build_tfidf_matrix,
        calculate_similarity_matrix,
        extract_additional_features,
    )

    # 1. Chuẩn bị dữ liệu
    files = get_all_files()
    sentences = split_sentences(read_text(files[0]))
    print(
        f"--- Kiểm tra textrank.py (Chỉ dùng NumPy/sklearn) với {len(sentences)} câu ---"
    )

    # 2. Tạo ma trận tương đồng & đặc trưng
    tfidf_matrix, _ = build_tfidf_matrix(sentences)
    sim_matrix = calculate_similarity_matrix(tfidf_matrix)
    extra_features = extract_additional_features(sentences)

    # 3. Cải tiến ma trận bằng đặc trưng (Tiêu chí 8)
    enhanced_sim = apply_feature_weights(sim_matrix, extra_features)

    # 4. Tính PageRank thuần bằng NumPy (Tiêu chí 5)
    scores = calculate_pagerank_numpy(enhanced_sim)

    # 5. Tạo bản tóm tắt 3 câu (Tiêu chí 6)
    summary = generate_summary(sentences, scores, top_n=3)

    print("\n--- KẾT QUẢ BẢN TÓM TẮT TRÍCH XUẤT (3 CÂU) ---")
    print(summary)