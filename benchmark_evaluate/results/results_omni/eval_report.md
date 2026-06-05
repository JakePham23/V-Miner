# Benchmark Evaluation Report

**Mode:** `traditional`

## Traditional Metrics Summary
Metrics evaluated:
- **Similarity (Normalized Edit Distance)**: Higher is better.
- **CER / WER**: Character/Word Error Rate. Lower is better.
- **BLEU / METEOR**: Text semantic matching. Higher is better.
- **TEDS (Structure / Content)**: Table layout accuracy. Higher is better.
- **GT Tbls / Pred Tbls**: Số lượng bảng gốc (Ground Truth) và số lượng bảng nhận diện được.

| Document                                                                              |    Sim |    CER |    WER |   BLEU |   METEOR |   TEDS-Str |   TEDS-Con |   GT Tbls |   Pred Tbls |
|:--------------------------------------------------------------------------------------|-------:|-------:|-------:|-------:|---------:|-----------:|-----------:|----------:|------------:|
| docstructbench_dianzishu_zhongwenzaixian-o.O-61569294.pdf_128                         | 0.8221 | 0.1779 | 0.7143 | 0.1779 |   0.39   |     1      |     1      |         0 |           0 |
| notes_f7f010b78016aeebd76e56d9283eb67f_49                                             | 0.9217 | 0.0783 | 0.4762 | 0.4697 |   0.7416 |     0.3962 |     0.2824 |         1 |           1 |
| notes_1ba14cb325bc448f7201b20502ecf2b5_15                                             | 0.7572 | 0.2759 | 0.963  | 0.1932 |   0.4044 |     0.8148 |     0.8056 |         1 |           1 |
| docstructbench_llm-raw-scihub-o.O-j.physletb.2004.06.101.pdf_3                        | 0.8208 | 0.1906 | 1.0511 | 0.6537 |   0.5979 |     1      |     1      |         0 |           0 |
| newspaper_1cddf9d22ca549f3a86cf1512a3110cc_1                                          | 0.8751 | 0.127  | 0.8168 | 0.2368 |   0.38   |     1      |     1      |         0 |           0 |
| yanbaopptmerge_0c79d327060dbf9f1582d03c235dadb039533a19091d2c0d24f2ad95d267f79b.pdf_2 | 0.9947 | 0.0053 | 0.2222 | 0.6776 |   0.83   |     0.8387 |     0.8311 |         1 |           1 |
| newspaper_5e266dfd9c498cab274e12a7b4a75755_4                                          | 0.9835 | 0.0165 | 0.0933 | 0.8416 |   0.8986 |     1      |     1      |         0 |           0 |
| docstructbench_llm-raw-scihub-o.O-j.chroma.2005.05.085.pdf_4                          | 0.9491 | 0.0519 | 0.1511 | 0.7756 |   0.8951 |     0.9222 |     0.6926 |         1 |           1 |
| jiaocaineedrop_jiaocai_needrop_en_1898                                                | 0.9852 | 0.0149 | 0.0571 | 0.8792 |   0.9774 |     0.8932 |     0.8932 |         1 |           1 |
| docstructbench_dianzishu_zhongwenzaixian-o.O-61520814.pdf_185                         | 0.7595 | 0.2658 | 0.9219 | 0.055  |   0.422  |     0.6792 |     0.6708 |         1 |           1 |
| eastmoney_62b4149b1612ce28d20f26cd5c5b2e18f80b26fca6e4452e090376a2fe72eae3.pdf_0      | 0.6245 | 0.3755 | 0.7786 | 0.175  |   0.3601 |     0.3017 |     0.2309 |         2 |           4 |
| docstructbench_dianzishu_zhongwenzaixian-o.O-61522235.pdf_170                         | 0.7213 | 0.2991 | 0.8148 | 0.188  |   0.5907 |     0.8654 |     0.8541 |         1 |           1 |
| jiaocaineedrop_jiaocai_needrop_en_3361                                                | 0.6686 | 0.3314 | 0.6535 | 0.1867 |   0.3496 |     1      |     1      |         0 |           0 |
| yanbaopptmerge_yanbaoPPT_145                                                          | 0.9277 | 0.0745 | 0.8    | 0.613  |   0.8962 |     1      |     1      |         0 |           0 |
| docstructbench_dianzishu_zhongwenzaixian-o.O-60599898.pdf_30                          | 0.9872 | 0.0128 | 0.8261 | 0.9178 |   0.9427 |     1      |     1      |         0 |           0 |
| jiaocaineedrop_Chapter9.pdf_46                                                        | 0.8132 | 0.1868 | 0.3524 | 0.8057 |   0.8538 |     0.85   |     0.85   |         1 |           1 |
| jiaocaineedrop_Evans_PDE_Solution_Chapter_6_Second-Order_Elliptic_Equations.pdf_5     | 0.6928 | 0.3223 | 1.3714 | 0.4753 |   0.535  |     1      |     1      |         0 |           0 |
| yanbaopptmerge_SE05.pdf_7                                                             | 0.9296 | 0.0704 | 0.1538 | 0.6482 |   0.8721 |     1      |     1      |         0 |           0 |

## Phân tích và Nhận xét (Cập nhật Mới nhất)

Dựa vào bảng kết quả đánh giá trên tập dữ liệu OmniDocBench, chúng ta có thể rút ra các nhận xét chi tiết về hiệu suất của MinerU trên từng loại tài liệu:

### 1. Tài liệu văn bản thuần (Notes, E-books, Newspaper)
- Các file như `newspaper_5e26...`, `newspaper_1cdd...` có **Sim** rất cao (>0.87 - 0.98) và **CER** rất thấp.
- **Đánh giá:** MinerU xử lý cực kỳ xuất sắc các tài liệu dạng báo chí chia cột đơn giản hoặc văn bản đọc tuyến tính. Tuy nhiên, đối với một số tài liệu ghi chú (`notes_f7f0...`), dù văn bản nhận diện tốt (Sim 0.92) nhưng cấu trúc bảng biểu lại bị phá vỡ một phần (TEDS-Str giảm còn 0.39).

### 2. Tài liệu Khoa học (Sci-Hub) và Giáo trình (Jiaocai/Textbooks)
- Các file học thuật như `jiaocaineedrop_jiaocai_needrop_en_1898`, `docstructbench_llm-raw-scihub-o.O-j.chroma...` đạt **Sim** cực tốt (0.94 - 0.98).
- **Bảng biểu:** Cấu trúc bảng trong các báo cáo khoa học được giữ nguyên vẹn với độ chuẩn xác tuyệt vời (TEDS-Str > 0.89).
- **Đánh giá:** MinerU rất đáng tin cậy để bóc tách các bài báo khoa học chuẩn mực có chứa bảng biểu học thuật.

### 3. Vấn đề với Báo cáo Tài chính phức tạp (EastMoney)
- File `eastmoney_62b414...` tiếp tục là một thách thức lớn.
- **CER** lên mức 37.55%, **Sim** chỉ còn 0.62.
- **Bảng biểu:** Trong Ground Truth chỉ có **2** bảng, nhưng MinerU lại chia cắt sai thành **4** bảng (Pred Tbls = 4). Điểm TEDS sụt giảm trầm trọng (TEDS-Str = 0.30, TEDS-Con = 0.23).
- **Đánh giá:** MinerU gặp rất nhiều rủi ro khi parse các báo cáo tài chính đặc thù của Trung Quốc, lưới dữ liệu quá dày đặc khiến thuật toán layout nhận diện phân mảnh sai các khối bảng.

### 4. Vấn đề với Công thức Toán học (PDE)
- File `jiaocaineedrop_Evans_PDE...` có **Sim** chỉ 0.69, **CER** cao (32%).
- **Đánh giá:** Lỗi chủ yếu do MinerU không thể chuyển đổi trơn tru các công thức toán học vi tích phân (Math/Equation) sang chuỗi chuẩn, sinh ra các ký tự rác làm đội tỷ lệ lỗi (CER) lên rất cao.
