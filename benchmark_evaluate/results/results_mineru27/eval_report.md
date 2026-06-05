# Benchmark Evaluation Report

**Mode:** `traditional`

## Traditional Metrics Summary
Metrics evaluated:
- **Similarity (Normalized Edit Distance)**: Higher is better.
- **CER / WER**: Character/Word Error Rate. Lower is better.
- **BLEU / METEOR**: Text semantic matching. Higher is better.
- **TEDS (Structure / Content)**: Table layout accuracy. Higher is better.
- **GT Tbls / Pred Tbls**: Số lượng bảng gốc (Ground Truth) và số lượng bảng nhận diện được.

| Document                  |   Sim |   CER |    WER |   BLEU |   METEOR |   TEDS-Str |   TEDS-Con |   GT Tbls |   Pred Tbls |
|:--------------------------|------:|------:|-------:|-------:|---------:|-----------:|-----------:|----------:|------------:|
| CTDT_Khoa_CNTT_K2024_HTTT | 0.711 | 0.289 | 0.7853 | 0.1257 |   0.2682 |     0.7307 |     0.6541 |        20 |          18 |

/Users/jakepham/Downloads/V-minerU-new-setting/vmineru6af67f8/V-Miner/benchmark_evaluate/evalute_finally
1. giờ tạo file doc báo cáo trình bày kịch bản là các file pdf ở trong này liệt kê ra "/Users/jakepham/Downloads/V-minerU-new-setting/vmineru6af67f8/V-Miner/input_benchmark" bao nhiêu pages đó  
2. lí do dùng là lightrag framework KG có sẵn mineru nhưng khi sử dụng cho các tài liệu tiếng việt kết quả không ra cấu trúc tiếng việt "/Users/jakepham/Downloads/V-minerU-new-setting/vmineru6af67f8/V-Miner/benchmark_evaluate/results_mineru27/eval_report.md" và chừa chỗ cho tôi capture ảnh  dùng model 1.5 thường mineru gì ấy 
3. sau đó đã cải tiến như nào cho dùng ppdoclayout parse ra rồi gửi text, table như nào skeletonn rồi cho local và api ..... (luồng thuật toán)
4. xong đưa ra kết quả "/Users/jakepham/Downloads/V-minerU-new-setting/vmineru6af67f8/V-Miner/benchmark_evaluate/results" lightonocr và qwen3vl4b 
nhớ có nhận xét => trước đó phải giải thích các metrics và hàm sử dụng.... rồi thông số đó là sao (copy bảng vào) 
5. quãng thời gian sau => mineru cập nhập bản 3.2 (thời gian) với model minerurpo 1.5 gì ấy tốt hơn cho tiếng việt .... => kết quả result_mineru "/Users/jakepham/Downloads/V-minerU-new-setting/vmineru6af67f8/V-Miner/benchmark_evaluate/results" 