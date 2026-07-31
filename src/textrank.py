import networkx as nx

def rank_sentences(similarity_matrix):
    # Đặc trưng 5: Biểu diễn văn bản thành đồ thị
    graph = nx.from_numpy_array(similarity_matrix)

    # Xếp hạng mức độ quan trọng của các câu bằng PageRank
    scores = nx.pagerank(
        graph,
        alpha=0.85,
        weight="weight",
        max_iter=500
    )

    return scores

# Chọn các câu có điểm cao nhất để tạo bản tóm tắt
def generate_summary(sentences, scores, similarity_matrix, top_n=18):

    if not sentences:
        return []

    top_n = min(top_n, len(sentences))

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    summary = []

    for index , _ in ranked:

        sentence = sentences[index].strip()

        duplicate = False
        for selected_index, _ in summary:
            sim = similarity_matrix[index][selected_index]
            if sim > 0.9:
                duplicate = True
                break

        if duplicate:
            continue

        summary.append((index, sentence))

        if len(summary) == top_n:
            break

    return [sentence for _, sentence in summary]