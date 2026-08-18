"""
Flask API nhỏ bọc lại pipeline TextRank hiện có (preprocess/vector/textrank)
để phục vụ giao diện web (static/index.html).

Chạy:
    python api.py
    truy cập http://127.0.0.1:5000
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from flask import Flask, request, jsonify, send_from_directory

try:
    from flask_cors import CORS
    _HAS_CORS = True
except ImportError:
    _HAS_CORS = False

from vector import (
    build_tfidf_matrix,
    calculate_similarity_matrix,
    extract_additional_features,
)
from textrank import apply_feature_weights, calculate_pagerank_numpy
from web_text import split_sentences_generic, clean_reference_text
from similarity import sentence_similarity, keyword_overlap

app = Flask(__name__, static_folder="static", static_url_path="")
if _HAS_CORS:
    CORS(app)

MAX_CHARS = 50_000


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/summarize", methods=["POST"])
def summarize():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    ratio = data.get("ratio")
    top_n = data.get("top_n")
    reference_text = (data.get("reference") or "").strip()

    if not text:
        return jsonify({"error": "Vui lòng nhập văn bản cần tóm tắt."}), 400

    if len(text) > MAX_CHARS:
        return jsonify({"error": f"Văn bản quá dài (tối đa {MAX_CHARS:,} ký tự)."}), 400

    sentences = split_sentences_generic(text)

    if len(sentences) < 2:
        return jsonify({"error": "Văn bản cần có ít nhất 2 câu để có thể tóm tắt."}), 400

    if not top_n:
        ratio = float(ratio) if ratio else 0.3
        ratio = min(max(ratio, 0.05), 1.0)
        top_n = max(1, round(len(sentences) * ratio))

    top_n = min(int(top_n), len(sentences))

    try:
        tfidf_matrix, _ = build_tfidf_matrix(sentences)
        similarity_matrix = calculate_similarity_matrix(tfidf_matrix)
    except ValueError:
        # Xảy ra khi TF-IDF không tìm được từ nào lặp lại giữa các câu
        # -> văn bản quá ngắn / các câu quá khác biệt nhau.
        return jsonify({
            "error": "Không đủ từ lặp lại giữa các câu để tính TF-IDF. "
                     "Hãy thử dán một đoạn văn bản dài hơn."
        }), 400

    # Trích đặc trưng bổ sung (độ dài câu, vị trí câu, số liệu, viết hoa)
    # và gộp trọng số vào đồ thị trước khi xếp hạng — giống hệt main.py.
    extra_features = extract_additional_features(sentences)
    enhanced_matrix = apply_feature_weights(similarity_matrix, extra_features)

    scores = calculate_pagerank_numpy(enhanced_matrix)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_indices = sorted(idx for idx, _ in ranked[:top_n])

    summary_sentences = [sentences[i] for i in top_indices]
    summary_index_set = set(top_indices)

    detail = [
        {
            "index": i,
            "text": s,
            "score": round(float(scores[i]), 4),
            "in_summary": i in summary_index_set,
        }
        for i, s in enumerate(sentences)
    ]

    # Độ chính xác (Tiêu chí 7) — Cosine TF-IDF giữa bản tóm tắt hệ thống và
    # bản tóm tắt tham chiếu do người dùng dán vào, cùng cách tính với
    # src/evaluate.py (không dùng ROUGE).
    accuracy = None
    accuracy_warning = None
    accuracy_detail = None
    if reference_text:
        ref_clean = clean_reference_text(reference_text)
        if ref_clean:
            summary_text = " ".join(summary_sentences)
            try:
                accuracy = round(float(sentence_similarity(ref_clean, summary_text)), 4)
                # Chi tiết để giải thích "vì sao" ra điểm đó — từ khoá xuất
                # hiện ở cả 2 bên, dùng để minh hoạ cho phần tử/mẫu số của
                # công thức Cosine Similarity = (A . B) / (|A| * |B|).
                overlap = keyword_overlap(ref_clean, summary_text)
                accuracy_detail = {
                    "shared_words": overlap["shared_words"],
                    "shared_total": overlap["shared_total"],
                    "reference_word_count": overlap["reference_word_count"],
                    "candidate_word_count": overlap["candidate_word_count"],
                }
            except ValueError:
                accuracy_warning = "Không đủ từ chung giữa 2 văn bản để tính độ chính xác."
        else:
            accuracy_warning = "Bản tham chiếu trống sau khi làm sạch."

    return jsonify({
        "sentence_count": len(sentences),
        "summary_count": len(summary_sentences),
        "compression": round(len(summary_sentences) / len(sentences), 3),
        "summary": summary_sentences,
        "sentences": detail,
        "accuracy": accuracy,
        "accuracy_warning": accuracy_warning,
        "accuracy_detail": accuracy_detail,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
