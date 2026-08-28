# Tóm tắt văn bản tự động bằng thuật toán TextRank

## Thông tin đồ án

| Mục | Nội dung |
|------|-------------|
| **Môn học** | Xử lý ngôn ngữ tự nhiên (NLP) |
| **Đề tài** | Xây dựng hệ thống tóm tắt văn bản bằng thuật toán TextRank |
| **Ngôn ngữ** | Python |
| **Phương pháp** | TextRank — tóm tắt trích xuất dựa trên đồ thị (Graph-based Extractive Summarization) |

---

## 1. Mục tiêu bài toán

Xây dựng hệ thống **tóm tắt trích xuất** (extractive summarization): tự động chọn ra
những câu quan trọng nhất trong văn bản gốc để tạo thành bản tóm tắt, **không sinh
câu mới**.

| | |
|---|---|
| **Input** | Một văn bản tiếng Anh thuộc bộ dữ liệu DUC_TEXT (định dạng gắn thẻ `<s>...</s>`) |
| **Output** | File `<tên_văn_bản>_summary.txt` chứa T câu được chọn, giữ nguyên thứ tự xuất hiện trong văn bản gốc |

---

## 2. Ý tưởng chính của phương pháp tiếp cận

TextRank chuyển bài toán **chọn câu quan trọng** thành bài toán **xếp hạng node
trên đồ thị**, dựa trên ý tưởng của thuật toán PageRank:

> Một câu là quan trọng nếu nó **tương đồng với nhiều câu quan trọng khác**.

Đây là một định nghĩa đệ quy — một câu nhận được "phiếu bầu" từ các câu tương đồng
với nó, và phiếu bầu từ một câu vốn đã quan trọng thì có trọng số cao hơn. Định
nghĩa đệ quy này được giải bằng cách lặp cho tới khi điểm số hội tụ.

Ưu điểm: **hoàn toàn không giám sát** — không cần dữ liệu gán nhãn, không cần huấn
luyện, chạy được ngay trên một văn bản đơn lẻ.

---

## 3. Các bước thực hiện chi tiết

Toàn bộ pipeline nằm trong `src/main.py`, gồm 6 bước:

| Bước | Nội dung | Module |
|---|---|---|
| 1 | Đọc văn bản từ bộ dữ liệu DUC_TEXT | `preprocess.read_text` |
| 2 | Tiền xử lý, bóc câu từ thẻ `<s>...</s>`, chuẩn hoá khoảng trắng | `preprocess.split_sentences` |
| 3 | Biểu diễn mỗi câu thành vector TF-IDF | `vector.build_tfidf_matrix` |
| 4 | **Xây đồ thị**: tính ma trận kề bằng Cosine Similarity | `vector.calculate_similarity_matrix` |
| 5 | **Xếp hạng câu** bằng PageRank (Power Iteration) | `textrank.calculate_pagerank_numpy` |
| 6 | Chọn Top-T câu, sắp lại theo thứ tự gốc, ghi ra file | `textrank.generate_summary` |

---

## 4. Biểu diễn văn bản thành đồ thị

Văn bản được mô hình hoá thành một **đồ thị vô hướng có trọng số** `G = (V, E)`:

| Thành phần | Ý nghĩa |
|---|---|
| **Node** `v_i ∈ V` | Một câu trong văn bản |
| **Cạnh** `e_ij ∈ E` | Nối 2 câu có nội dung tương đồng |
| **Trọng số cạnh** `w_ij` | Cosine Similarity giữa vector TF-IDF của câu `i` và câu `j` |

### Cách tính trọng số cạnh

Mỗi câu được biểu diễn thành vector TF-IDF (`TfidfVectorizer` với `stop_words="english"`,
`ngram_range=(1, 2)` để bắt được cả từ đơn lẫn cụm 2 từ). Trọng số cạnh là:

```
w_ij = cosine(v_i, v_j) = (v_i · v_j) / (‖v_i‖ × ‖v_j‖)
```

Đường chéo chính được gán bằng 0 (`np.fill_diagonal(sim_matrix, 0)`) để **loại bỏ
self-loop** — một câu không được tự bỏ phiếu cho chính nó.

Kết quả là **ma trận kề** `sim_matrix` kích thước `n × n`, chính là biểu diễn đồ thị
của văn bản.

### Đặc điểm đồ thị thu được (đo trên bộ DUC_TEXT/train)

| Chỉ số | Giá trị |
|---|---|
| Số node trung bình / văn bản | ~259 câu |
| Mật độ đồ thị (tỉ lệ cặp câu có cạnh) | ~27.4% |
| Trọng số cạnh trung bình | ~0.036 |

Đồ thị **thưa vừa phải** — nhờ loại từ dừng và dùng bigram, các câu không liên quan
gần như không có cạnh nối, giúp cấu trúc chủ đề của văn bản hiện rõ.

### Trực quan hoá

Notebook `notebook/graph_visualization.ipynb` vẽ 3 hình minh hoạ (lấy **20 câu đầu**
của văn bản `d061j` cho dễ quan sát — pipeline thật xử lý toàn bộ ~259 câu):

1. **Đồ thị câu** — node xếp theo vòng tròn, độ đậm/dày của cạnh tỉ lệ với Cosine
   Similarity, kích thước và màu node tỉ lệ với điểm PageRank
2. **Heatmap ma trận Cosine Similarity** — nhìn trực tiếp ma trận kề
3. **Bar chart điểm PageRank** — highlight các câu được chọn vào bản tóm tắt

> **Lưu ý:** ngưỡng `threshold = 0.05` trong notebook **chỉ dùng để lọc cạnh khi vẽ**
> cho hình đỡ rối. Khi tính PageRank, hệ thống dùng **toàn bộ đồ thị có trọng số**,
> không cắt ngưỡng.

---

## 5. Xếp hạng câu theo mức độ quan trọng

Điểm quan trọng của mỗi câu được tính bằng **PageRank**, cài đặt thủ công bằng
**Power Iteration** với NumPy (không dùng NetworkX).

### Bước 1 — Chuẩn hoá thành ma trận chuyển trạng thái

Ma trận kề được chuẩn hoá theo hàng để tổng mỗi hàng bằng 1, sau đó chuyển vị để
thành ma trận **column-stochastic**:

```
M[j][i] = w_ij / Σ_k w_ik
```

**Xử lý dangling node:** câu cô lập (không có cạnh nào, `row_sum = 0`) được phân
phối đều `1/n` sang tất cả các câu khác. Nếu không xử lý sẽ bị **chia cho 0** và
thất thoát tổng điểm PageRank.

### Bước 2 — Power Iteration

Lặp cho tới khi hội tụ:

```
R_(t+1) = d · M · R_t + (1 − d) / N
```

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `d` (damping factor) | 0.85 | Xác suất tiếp tục đi theo cạnh |
| `N` | số câu | Số node của đồ thị |
| `max_iter` | 100 | Số vòng lặp tối đa |
| `tol` | 1e-6 | Ngưỡng hội tụ (chuẩn L1) |

Vector khởi tạo `R_0 = 1/N` cho mọi node. Vòng lặp dừng khi
`‖R_(t+1) − R_t‖₁ < tol`.

### Bước 3 — Chọn câu

Sắp xếp câu theo điểm PageRank giảm dần, lấy **Top-T = 18 câu**, rồi **sắp lại theo
thứ tự xuất hiện gốc** trong văn bản để bản tóm tắt giữ được mạch văn tự nhiên.

---

## 6. Kết quả tóm tắt

```
output/
└── <tên_văn_bản>_summary.txt      (50 file)
```

Mỗi file chứa 18 câu được trích, mỗi câu một dòng, theo đúng thứ tự trong văn bản gốc.

---

## 7. Đánh giá kết quả và nhận xét

### Cách đánh giá

Bản tóm tắt của hệ thống được so khớp với bản tóm tắt tham chiếu do con người viết
(`data/DUC_SUM`). Một câu hệ thống được tính là **đúng** nếu Cosine Similarity TF-IDF
của nó với một câu tham chiếu (chưa bị ghép trước đó) `≥ 0.3`.

```
Precision = số câu trích đúng / số câu hệ thống trích ra
Recall    = số câu trích đúng / số câu trong bản tham chiếu
F1        = 2 × P × R / (P + R)
```

### Kết quả

Trong 50 văn bản được tóm tắt, **34 văn bản** có bản tham chiếu hợp lệ và được đưa
vào chấm; 16 văn bản còn lại bị bỏ qua do thiếu hoặc rỗng file trong `DUC_SUM`
(`evaluate.py` in danh sách cụ thể).

| Phương pháp xếp hạng | Precision | Recall | F1 |
|---|---|---|---|
| Lead-18 (lấy 18 câu đầu) | 0.111 | 0.133 | 12.0% |
| Chọn 18 câu dài nhất | 0.111 | 0.145 | 12.5% |
| Degree centrality (tổng trọng số cạnh) | 0.144 | 0.179 | 15.8% |
| **TextRank — PageRank (hệ thống này)** | **0.157** | **0.196** | **17.2%** |

### Nhận xét

**Ưu điểm**

- TextRank vượt baseline Lead-18 tới **43% tương đối** về F1 (17.2% so với 12.0%) —
  chứng tỏ việc xếp hạng trên đồ thị thực sự chọn được câu quan trọng, không phải
  chỉ ăn may vào vị trí đầu văn bản.
- TextRank vẫn cao hơn **Degree centrality** (17.2% so với 15.8%). Đây là so sánh
  đáng chú ý nhất: degree centrality chỉ đếm tổng trọng số cạnh của một câu, còn
  PageRank lan truyền điểm qua nhiều vòng lặp. Chênh lệch này cho thấy **giá trị
  thật sự nằm ở phần lặp đệ quy** — một câu nối với câu quan trọng có giá trị hơn
  một câu nối với nhiều câu tầm thường.
- Không cần dữ liệu huấn luyện, chạy được trên bất kỳ văn bản đơn lẻ nào.
- Cài đặt PageRank thuần NumPy nên minh bạch, kiểm soát được toàn bộ tham số.

**Nhược điểm**

- **Chỉ dựa trên trùng lặp từ vựng.** TF-IDF không hiểu ngữ nghĩa: hai câu diễn đạt
  cùng một ý bằng từ khác nhau sẽ không có cạnh nối.
- **Thiên vị câu dài.** Câu dài chứa nhiều từ hơn nên dễ tương đồng với nhiều câu
  khác, dẫn tới điểm PageRank cao.
- **Chưa xử lý trùng lặp nội dung.** Các câu được chọn có thể lặp ý nhau, vì thuật
  toán chọn Top-T độc lập chứ không xét độ đa dạng.
- **T cố định = 18** cho mọi văn bản, chưa co giãn theo độ dài văn bản gốc.
- Chỉ số F1 ~17% nhìn thấp, nhưng cần lưu ý cách chấm rất khắt khe: khớp câu theo
  ngưỡng cosine 0.3 và bản tham chiếu do người viết thường **diễn đạt lại** chứ
  không trích nguyên văn.

---

## 8. Hướng cải tiến

| Hướng | Mô tả |
|---|---|
| **Cắt ngưỡng cạnh** | Chỉ giữ cạnh có `w_ij > threshold` để đồ thị thưa hơn, giảm nhiễu từ các cặp câu tương đồng yếu |
| **Thay TF-IDF bằng embedding** | Dùng Sentence-BERT để tính similarity theo ngữ nghĩa thay vì trùng từ, khắc phục nhược điểm lớn nhất |
| **Chuẩn hoá theo độ dài câu** | Chia trọng số cạnh cho log độ dài câu để giảm thiên vị câu dài |
| **Thêm MMR** | Maximal Marginal Relevance khi chọn Top-T để tránh chọn các câu trùng ý nhau |
| **T co giãn** | Chọn T theo tỉ lệ phần trăm số câu văn bản gốc thay vì cố định 18 |

---

## Cấu trúc thư mục

```
Project(NLP)/
│
├── data/
│   ├── DUC_TEXT/
│   │   ├── train/          # 50 văn bản đầu vào (thẻ <s>...</s>)
│   │   └── test/           # 9 văn bản
│   └── DUC_SUM/            # Bản tóm tắt tham chiếu
│
├── output/                 # Bản tóm tắt do hệ thống sinh ra
│
├── notebook/
│   └── graph_visualization.ipynb   # Trực quan hoá đồ thị (tiêu chí 4)
│
├── src/
│   ├── preprocess.py       # Đọc dữ liệu & tách câu
│   ├── vector.py           # TF-IDF + xây ma trận kề (đồ thị)
│   ├── textrank.py         # Ma trận stochastic, PageRank, chọn câu
│   ├── main.py             # Pipeline chính
│   ├── evaluate.py         # Đánh giá Precision / Recall / F1
│   ├── similarity.py       # Tiện ích so sánh câu (cho web UI)
│   ├── api.py              # Flask API cho giao diện web
│   ├── web_text.py         # Tách câu cho văn bản tự do
│   └── static/index.html   # Giao diện web
│
├── library.txt             # Thư viện phụ thuộc
├── README.md
└── WEB_UI_README.md
```

---

## Cách chạy

```bash
pip install -r library.txt

# Sinh bản tóm tắt cho toàn bộ dataset
cd src && python main.py

# Đánh giá kết quả
python evaluate.py

# Chạy giao diện web (http://127.0.0.1:5000)
python api.py
```

---

## Công nghệ sử dụng

- **Python 3**
- **scikit-learn** — TF-IDF, Cosine Similarity
- **NumPy** — cài đặt PageRank thủ công (không dùng NetworkX)
- **Matplotlib** — trực quan hoá đồ thị
- **Flask** — giao diện web
- **NLTK** — tách câu cho văn bản tự do nhập từ web
