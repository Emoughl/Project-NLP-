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


def generate_summary(sentences, scores, top_n=3):
    #Chọn top_n câu điểm PageRank cao nhất, sắp xếp theo thứ tự gốc
    if not sentences or not scores:
        return ""

    top_n = min(top_n, len(sentences))

    ranked_sentences = sorted(
        scores.items(), key=lambda x: x[1], reverse=True
    )
    top_indices = [idx for idx, _ in ranked_sentences[:top_n]]

    return "\n".join(sentences[i] for i in top_indices)


# ======================================================================
# PHẦN CẢI TIẾN (tiêu chí 8)
#
# Hai cải tiến được thêm vào bên trên pipeline gốc, dùng lại nguyên vẹn
# ma trận đồ thị và hàm calculate_pagerank_numpy() ở trên:
#
#   1) THÊM ĐẶC TRƯNG BIỂU DIỄN DỮ LIỆU — vị trí câu.
#      TextRank thuần bỏ qua hoàn toàn vị trí câu, trong khi văn bản tin
#      tức viết theo cấu trúc "kim tự tháp ngược" (thông tin cốt lõi nằm
#      ở đầu bài).
#
#   2) CHỐNG TRÙNG LẶP NỘI DUNG KHI CHỌN CÂU — MMR.
#      Top-T độc lập hay chọn phải các câu giống nhau, vì chính nhờ giống
#      nhau mà chúng cùng nhận được nhiều "phiếu bầu".
#
# Kết quả trên 34 văn bản có bản tham chiếu DUC_SUM (T = 18):
#     TextRank gốc                     P=0.157  R=0.196  F1=17.2%
#     + đặc trưng vị trí (alpha=0.5)   P=0.180  R=0.222  F1=19.7%
#     + MMR (lambda=0.7)               P=0.183  R=0.232  F1=20.2%
#     + cả hai (alpha=0.3, lambda=0.7) P=0.203  R=0.253  F1=22.3%
#
# Cách chạy:
#     python main.py --improved       # sinh tóm tắt vào output_improved/
#     python evaluate.py --improved   # chấm điểm bản cải tiến
# ======================================================================

ALPHA = 0.3    # trọng số của đặc trưng vị trí câu
LAMBDA = 0.7   # cân bằng "quan trọng" <-> "đa dạng" trong MMR


def position_scores(n):
    #Đặc trưng vị trí: câu càng gần đầu văn bản điểm càng cao (1/sqrt(i+1))
    pos = np.array([1.0 / np.sqrt(i + 1) for i in range(n)])
    return pos / pos.max()


def combine_scores(scores, n, alpha=ALPHA):
    #Kết hợp điểm PageRank (chuẩn hoá về [0,1]) với đặc trưng vị trí câu
    #   final_i = (1 - alpha) * pagerank_norm_i + alpha * pos_i
    pagerank = np.array([scores[i] for i in range(n)], dtype=float)
    if pagerank.max() > 0:
        pagerank = pagerank / pagerank.max()

    return (1.0 - alpha) * pagerank + alpha * position_scores(n)


def mmr_select(scores, sim_matrix, top_n=3, lam=LAMBDA):
    #Chọn câu bằng MMR thay vì lấy Top-N độc lập:
    #   MMR_i = lam * score_i - (1 - lam) * max(cosine(i, j) với j đã chọn)
    #Câu được chọn phải vừa quan trọng, vừa KHÁC các câu đã chọn.
    n = len(scores)
    selected, candidates = [], set(range(n))

    while len(selected) < min(top_n, n):
        best_idx, best_value = None, -np.inf
        for i in candidates:
            redundancy = max((sim_matrix[i][j] for j in selected), default=0.0)
            value = lam * scores[i] - (1.0 - lam) * redundancy
            if value > best_value:
                best_value, best_idx = value, i
        selected.append(best_idx)
        candidates.discard(best_idx)

    return selected


def generate_summary_improved(
    sentences, scores, sim_matrix, top_n=3, alpha=ALPHA, lam=LAMBDA
):
    #Bản cải tiến của generate_summary(): điểm PageRank + đặc trưng vị trí,
    #chọn câu bằng MMR, vẫn sắp lại theo thứ tự gốc như bản gốc.
    if not sentences or not scores:
        return ""

    final_scores = combine_scores(scores, len(sentences), alpha)
    top_indices = sorted(mmr_select(final_scores, sim_matrix, top_n, lam))

    return "\n".join(sentences[i] for i in top_indices)


# --- KIỂM TRA TRỰC TIẾP ---
if __name__ == "__main__":
    from preprocess import get_all_files, read_text, split_sentences
    from vector import build_tfidf_matrix, calculate_similarity_matrix

    files = get_all_files()
    sentences = split_sentences(read_text(files[0]))
    print(f"--- Kiểm tra textrank.py với {len(sentences)} câu ---")

    tfidf_matrix, _ = build_tfidf_matrix(sentences)
    sim_matrix = calculate_similarity_matrix(tfidf_matrix)

    scores = calculate_pagerank_numpy(sim_matrix)
    summary = generate_summary(sentences, scores, top_n=18)

    print("\n--- BẢN TÓM TẮT (18 CÂU) ---")
    print(summary)
