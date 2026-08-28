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
| Số node trung bình / văn bản | ~294 câu |
| Mật độ đồ thị (tỉ lệ cặp câu có cạnh) | ~27.4% |
| Trọng số cạnh trung bình (chỉ tính các cặp **có cạnh**) | ~0.033 |
| Trọng số trung bình trên **mọi cặp câu** (kể cả cặp không có cạnh) | ~0.008 |

> Số liệu đo trên toàn bộ 50 văn bản trong `DUC_TEXT/train`.

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

Đo trên 50 văn bản: thuật toán hội tụ sau trung bình **28 vòng lặp** (nhiều nhất 55),
tức luôn dừng vì đạt ngưỡng hội tụ chứ không phải vì chạm giới hạn `max_iter = 100`.

### Bước 3 — Chọn câu

Sắp xếp câu theo điểm PageRank giảm dần, lấy **Top-T = 18 câu**, rồi **sắp lại theo
thứ tự xuất hiện gốc** trong văn bản để bản tóm tắt giữ được mạch văn tự nhiên.

**Vì sao T = 18?** Các bản tóm tắt tham chiếu do người viết trong `DUC_SUM` dài trung
bình **15.1 câu** (ngắn nhất 10, dài nhất 20). Chọn T = 18 để độ dài bản tóm tắt của
hệ thống tương đương bản của người, nhờ đó phép so Precision / Recall mới công bằng
— nếu T quá nhỏ thì Recall bị ép xuống, T quá lớn thì Precision bị ép xuống.

#### Khảo sát tham số T

Con số 18 không chọn theo một văn bản đơn lẻ. Dưới đây là kết quả quét T trên **cả 34
văn bản** có bản tham chiếu, cho cả phương pháp gốc lẫn phương pháp cải tiến (§8):

| T cố định | Precision | Recall | F1 | F1 (bản cải tiến) |
|---|---|---|---|---|
| 10 | 0.171 | 0.119 | 13.9% | 20.4% |
| 15 | 0.159 | 0.165 | 16.0% | 22.0% |
| **18** | **0.157** | **0.196** | **17.2%** | **22.3%** |
| 20 | 0.149 | 0.207 | 17.1% | 22.2% |
| 25 | 0.141 | 0.248 | 17.8% | 21.6% |
| 30 | 0.128 | 0.268 | 17.2% | 20.7% |

F1 đạt đỉnh trong khoảng T = 18–25 với phương pháp gốc, và đúng tại **T = 18** với
phương pháp cải tiến — nên hệ thống chốt T = 18 cho cả hai để so sánh công bằng.

#### Vì sao KHÔNG chọn T theo tỉ lệ phần trăm số câu

Một hướng tự nhiên là cho T co giãn theo độ dài văn bản, ví dụ lấy 30% số câu. Nhưng
dữ liệu DUC cho thấy bản tóm tắt của người viết có độ dài **gần như cố định**, không
tỉ lệ với văn bản gốc:

| | Trung bình | Nhỏ nhất | Lớn nhất |
|---|---|---|---|
| Số câu văn bản gốc | 309 | 143 | 680 |
| Số câu bản tham chiếu | **15.1** | 10 | 20 |
| Tỉ lệ tham chiếu / gốc | 5.7% | 1.7% | 14.0% |

Người viết luôn tóm tắt khoảng 15 câu dù bài dài 143 hay 680 câu — tỉ lệ dao động
1.7%–14.0% chính là *hệ quả* của việc độ dài bài thay đổi, chứ không phải vì họ tóm
tắt theo tỉ lệ. Kết quả thực nghiệm cũng xác nhận điều đó:

| T = tỉ lệ số câu | T trung bình | Precision | Recall | F1 | F1 (cải tiến) |
|---|---|---|---|---|---|
| 3% | 9.3 | 0.191 | 0.106 | 13.1% | 17.9% |
| 5% | 15.5 | 0.171 | 0.163 | 16.0% | 20.3% |
| 7% | 21.6 | 0.151 | 0.206 | 16.7% | 21.4% |
| 10% | 30.9 | 0.130 | 0.249 | 16.4% | 21.4% |
| 30% | 92.7 | 0.100 | 0.563 | 16.7% | 18.2% |

Không mức tỉ lệ nào vượt được T = 18 cố định. Riêng mức 30% sinh ra bản tóm tắt trung
bình **93 câu** (văn bản dài nhất là 204 câu — dài gần bằng một phần ba bài gốc), khiến
Precision rơi xuống 0.100. Bản lai giữa hai cách (tỉ lệ nhưng chặn trên/dưới) cũng chỉ
đạt tối đa 17.1% / 21.9%, vẫn không hơn.

> Kết luận: với bộ DUC — nơi bản tóm tắt tham chiếu có độ dài cố định — **T cố định là
> lựa chọn đúng**. Nếu đổi sang bộ dữ liệu mà bản tóm tắt dài tỉ lệ với văn bản gốc thì
> mới nên chuyển sang T theo tỉ lệ. Giao diện web (`api.py`) vẫn cho người dùng chọn tỉ
> lệ nén vì văn bản tự do dán vào thường ngắn hơn nhiều so với văn bản DUC.

---

## 6. Kết quả tóm tắt

```
output/                             # TextRank gốc
└── <tên_văn_bản>_summary.txt      (50 file)

output_improved/                    # Phương pháp cải tiến (mục 8)
└── <tên_văn_bản>_summary.txt      (50 file)
```

Mỗi file chứa 18 câu được trích, mỗi câu một dòng, theo đúng thứ tự trong văn bản gốc.

**Về thư mục `data/DUC_TEXT/test/`:** TextRank là phương pháp **không giám sát**, không
có tham số nào được học từ dữ liệu, nên không cần chia train/test để tránh rò rỉ dữ
liệu — cách chia này là cấu trúc sẵn của bộ DUC. Toàn bộ báo cáo này chạy trên tập
`train/` (50 văn bản) vì đây là phần có bản tóm tắt tham chiếu trong `DUC_SUM`.

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

| Phương pháp | Precision | Recall | F1 |
|---|---|---|---|
| **TextRank — PageRank (hệ thống này)** | **0.157** | **0.196** | **17.2%** |
| **TextRank cải tiến** (vị trí + MMR, xem §8) | **0.203** | **0.253** | **22.3%** |

```bash
cd src
python evaluate.py              # chấm bản gốc      -> F1 17.2%
python evaluate.py --improved   # chấm bản cải tiến -> F1 22.3%
```

### Nhận xét

**Ưu điểm**

- Không cần dữ liệu huấn luyện, không cần gán nhãn, chạy được trên bất kỳ văn bản
  đơn lẻ nào — rất phù hợp với bộ DUC vì chỉ 34/50 văn bản có bản tóm tắt tham chiếu,
  không đủ để huấn luyện một mô hình có giám sát.
- Không phụ thuộc ngôn ngữ: phần đồ thị và PageRank giữ nguyên khi đổi sang ngôn ngữ
  khác, chỉ cần thay bước tách câu / tách từ.
- Cài đặt PageRank thuần NumPy nên minh bạch, kiểm soát được toàn bộ tham số.

**Nhược điểm**

- **Chỉ dựa trên trùng lặp từ vựng.** TF-IDF không hiểu ngữ nghĩa: hai câu diễn đạt
  cùng một ý bằng từ khác nhau sẽ không có cạnh nối.
- **Thiên vị câu dài.** Câu dài chứa nhiều từ hơn nên dễ tương đồng với nhiều câu
  khác, dẫn tới điểm PageRank cao.
- **Chưa xử lý trùng lặp nội dung.** Các câu được chọn có thể lặp ý nhau, vì thuật
  toán chọn Top-T độc lập chứ không xét độ đa dạng.
- **T cố định = 18** cho mọi văn bản. Đây là lựa chọn có căn cứ với bộ DUC (xem phần
  khảo sát tham số ở §5: T theo tỉ lệ đều cho F1 thấp hơn), nhưng sẽ phải xem lại nếu
  áp dụng lên bộ dữ liệu có bản tóm tắt dài tỉ lệ với văn bản gốc.
- Chỉ số F1 ~17% nhìn thấp, nhưng cần lưu ý cách chấm rất khắt khe: khớp câu theo
  ngưỡng cosine 0.3 và bản tham chiếu do người viết thường **diễn đạt lại** chứ
  không trích nguyên văn.

> **Về độ đo:** cách chấm ở đây là khớp **cả câu** theo ngưỡng Cosine TF-IDF ≥ 0.3
> — một biến thể tự định nghĩa, khắt khe hơn ROUGE. Độ đo chuẩn của ngành cho bài
> toán tóm tắt là ROUGE-1 / ROUGE-2 / ROUGE-L (khớp theo n-gram, rộng lượng hơn
> nên điểm sẽ cao hơn). Đây là hướng bổ sung tiếp theo cho phần đánh giá.

---

## 8. Cải tiến phương pháp

Hai cải tiến được **cài đặt ngay trong pipeline hiện có**, không tách thành module
riêng: phần `PHẦN CẢI TIẾN (tiêu chí 8)` ở cuối `src/textrank.py` dùng lại nguyên vẹn
ma trận đồ thị của `vector.py` và hàm `calculate_pagerank_numpy()` ở phía trên, chỉ
thêm bước tính điểm và bước chọn câu mới. `main.py` và `evaluate.py` nhận thêm tuỳ
chọn `--improved` để chạy nhánh cải tiến.

| Thành phần | Bản gốc | Bản cải tiến |
|---|---|---|
| Xây đồ thị | `vector.calculate_similarity_matrix` | *dùng lại* |
| Xếp hạng | `textrank.calculate_pagerank_numpy` | *dùng lại* + `combine_scores` (đặc trưng vị trí) |
| Chọn câu | `textrank.generate_summary` (Top-T) | `textrank.generate_summary_improved` (MMR) |
| Kết quả ghi ra | `output/` | `output_improved/` |

### Cải tiến 1 — Thêm đặc trưng biểu diễn dữ liệu: vị trí câu

TextRank thuần chỉ nhìn vào quan hệ tương đồng giữa các câu, **bỏ qua hoàn toàn vị
trí câu trong văn bản**. Nhưng văn bản tin tức (bộ DUC) viết theo cấu trúc *kim tự
tháp ngược*: thông tin cốt lõi nằm ở đầu bài. Điểm cuối cùng của mỗi câu vì vậy được
kết hợp tuyến tính giữa điểm PageRank đã chuẩn hoá và điểm vị trí:

```
final_i = (1 − α) · pagerank_norm_i + α · pos_i
pos_i   = (1 / √(i + 1)) chuẩn hoá về [0, 1],  i = chỉ số câu tính từ 0
α       = 0.3
```

### Cải tiến 2 — Chống trùng lặp nội dung khi chọn câu: MMR

Nhược điểm lớn nhất của việc lấy Top-T độc lập là **các câu điểm cao thường giống
nhau** — chính vì giống nhau nên chúng mới cùng nhận được nhiều "phiếu bầu". MMR
(*Maximal Marginal Relevance*) chọn câu **tuần tự**, mỗi bước trừ đi phần trùng lặp
với những câu đã chọn:

```
MMR_i = λ · final_i − (1 − λ) · max_{j đã chọn} cosine(i, j)
λ = 0.7
```

Câu được chọn phải vừa **quan trọng**, vừa **khác** các câu đã có trong bản tóm tắt.

### Kết quả

Đo trên đúng 34 văn bản có bản tham chiếu, cùng T = 18, cùng bộ độ đo:

| Cấu hình | Precision | Recall | F1 |
|---|---|---|---|
| TextRank gốc | 0.157 | 0.196 | 17.2% |
| + đặc trưng vị trí (α = 0.5) | 0.180 | 0.222 | 19.7% |
| + MMR (λ = 0.7) | 0.183 | 0.232 | 20.2% |
| **+ cả hai (λ = 0.7, α = 0.3)** | **0.203** | **0.253** | **22.3%** |

F1 tăng từ 17.2% lên **22.3%**, tức **+30% tương đối**. Hai cải tiến bổ trợ cho nhau:
đặc trưng vị trí cải thiện việc *chấm điểm*, MMR cải thiện việc *chọn câu*.

> Hai dòng giữa là thí nghiệm tách riêng từng cải tiến (ablation) để biết mỗi phần
> đóng góp bao nhiêu; code hiện tại chạy cấu hình cuối cùng — `α = 0.3`, `λ = 0.7`,
> khai báo ở đầu phần cải tiến trong `textrank.py`, đổi trực tiếp ở đó là tái lập
> được các dòng còn lại.

```bash
cd src
python main.py --improved       # sinh tóm tắt cải tiến vào output_improved/
python evaluate.py --improved   # chấm bản cải tiến  -> F1 22.3%
python evaluate.py              # chấm bản gốc để đối chiếu -> F1 17.2%
```

### Cải tiến đã thử nhưng **không** hiệu quả: cắt ngưỡng cạnh

Ý tưởng ban đầu là chỉ giữ lại cạnh có `w_ij > threshold` để đồ thị thưa hơn, giảm
nhiễu từ các cặp câu tương đồng yếu. Thực nghiệm cho kết quả **ngược lại**:

| Ngưỡng cắt cạnh | F1 |
|---|---|
| không cắt (hệ thống hiện tại) | **17.2%** |
| 0.02 | 16.1% |
| 0.05 | 15.0% |
| 0.10 | 11.1% |

Lý do: cắt ngưỡng loại bỏ quá nhiều cạnh yếu nhưng hợp lệ, làm đồ thị vỡ thành nhiều
thành phần liên thông rời rạc, PageRank không còn lan truyền được điểm giữa các cụm.
Vì vậy hệ thống **giữ nguyên đồ thị đầy đủ có trọng số**.

### Cải tiến đã thử nhưng **không** hiệu quả: cho T co giãn theo tỉ lệ

Ý tưởng: thay T cố định bằng T = tỉ lệ phần trăm số câu văn bản gốc, để bài dài thì
tóm tắt dài hơn. Thực nghiệm cho thấy mọi mức tỉ lệ đều **thua** T = 18 cố định
(3% → F1 13.1%, 5% → 16.0%, 7% → 16.7%, 10% → 16.4%, 30% → 16.7%), kể cả bản lai có
chặn trên/dưới. Nguyên nhân đã phân tích ở §5: bản tóm tắt tham chiếu của DUC dài gần
như cố định (~15 câu) bất kể văn bản gốc dài 143 hay 680 câu.

### Hướng cải tiến tiếp theo

| Hướng | Mô tả |
|---|---|
| **Thay TF-IDF bằng embedding** | Dùng Sentence-BERT để tính similarity theo ngữ nghĩa thay vì trùng từ — khắc phục nhược điểm lớn nhất của phương pháp |
| **Đánh giá bằng ROUGE** | Bổ sung ROUGE-1 / ROUGE-2 / ROUGE-L bên cạnh độ đo khớp câu hiện tại |
| **Chuẩn hoá theo độ dài câu** | Chia trọng số cạnh cho log độ dài câu để giảm thiên vị câu dài |

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
├── output/                 # Bản tóm tắt do hệ thống sinh ra (TextRank gốc)
├── output_improved/        # Bản tóm tắt của phương pháp cải tiến (§8)
│
├── notebook/
│   └── graph_visualization.ipynb   # Trực quan hoá đồ thị (tiêu chí 4)
│
├── src/
│   ├── preprocess.py       # Đọc dữ liệu & tách câu
│   ├── vector.py           # TF-IDF + xây ma trận kề (đồ thị)
│   ├── textrank.py         # Ma trận stochastic, PageRank, chọn câu
│   │                       #   + PHẦN CẢI TIẾN: đặc trưng vị trí & MMR (§8)
│   ├── main.py             # Pipeline chính (--improved để chạy bản cải tiến)
│   ├── evaluate.py         # Đánh giá Precision / Recall / F1 (--improved)
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

# Phương pháp cải tiến (mục 8)
python main.py --improved       # sinh tóm tắt vào output_improved/
python evaluate.py --improved   # chấm điểm bản cải tiến

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
