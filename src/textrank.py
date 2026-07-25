import networkx as nx
from similarity import sentence_similarity

def rank_sentences(similarity_matrix, sentences):
    # Xếp hạng các câu bằng thuật toán PageRank'

    # Mục 4 :Biểu diễn văn bản thành đồ thị'
    graph = nx.from_numpy_array(similarity_matrix) 

    # Mục 5 :Xếp hạng câu trong đồ thị theo mức độ quan trọng
    scores = nx.pagerank(graph)

    # Mục 8 : Cải tiến 1 - Phạt câu quá dài
    for i in scores:
        length = len(sentences[i].split())

        if length > 60:
            scores[i] *= 0.7

        if length > 90:
            scores[i] *= 0.5

    # Mục 8 : Cải tiến 2: Cộng điểm vị trí câu (position score)
    n = len(sentences)

    for i in scores:
        position_score = 1 - i / n
        scores[i] = 0.8 * scores[i] + 0.2 * position_score

    return scores

def generate_summary(sentences, scores, top_n=3):

    if not sentences:
        return []

    top_n = min(top_n, len(sentences))

    #Mục 6 :Lấy được tóm tắt sơ bộ
    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    summary = []

    for index, score in ranked:

        sentence = sentences[index].strip()

        # Mục 8 : Cải tiến 3 : Loại bỏ hẳn câu quá dài (>80 từ) 
        if len(sentence.split()) > 80:
            continue

        # Mục 8 : Cải tiến 4 : Lọc câu liệt kê
        if sentence.count(";") >= 3:
            continue

        # Mục 8 : Cải tiến 5 : Chống trùng ý
        duplicate = False

        for _, selected in summary:

            sim = sentence_similarity(sentence, selected)

            if sim > 0.65:
                duplicate = True
                break

        if duplicate:
            continue

        summary.append((index, sentence))

        if len(summary) == top_n:
            break

    summary.sort(key=lambda x: x[0])

    return [sentence for _, sentence in summary]