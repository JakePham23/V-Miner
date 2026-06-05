# Benchmark Evaluation Report

**Mode:** `traditional`

## Traditional Metrics Summary
Metrics evaluated:
- **Similarity (Normalized Edit Distance)**: Higher is better.
- **CER / WER**: Character/Word Error Rate. Lower is better.
- **BLEU / METEOR**: Text semantic matching. Higher is better.
- **TEDS (Structure / Content)**: Table layout accuracy. Higher is better.
- **GT Tbls / Pred Tbls**: Số lượng bảng gốc (Ground Truth) và số lượng bảng nhận diện được.

| Document                        |    Sim |    CER |    WER |   BLEU |   METEOR |   TEDS-Str |   TEDS-Con |   GT Tbls |   Pred Tbls |
|:--------------------------------|-------:|-------:|-------:|-------:|---------:|-----------:|-----------:|----------:|------------:|
| CTDT_Khoa_2024_19Sep2024_CNTN   | 0.8539 | 0.1643 | 0.1249 | 0.8538 |   0.8696 |     0.9994 |     0.9632 |        16 |          16 |
| CTDT_Khoa_2024_19Sep2024_TTNT   | 0.8075 | 0.2266 | 0.166  | 0.8117 |   0.8432 |     0.4277 |     0.3081 |        18 |          22 |
| CTDT_Khoa_2024_18Sep2024_CNTTin | 0.8151 | 0.2166 | 0.1808 | 0.8008 |   0.8475 |     0.7416 |     0.7051 |        25 |          31 |
| CTDT_Khoa_CNTT_K2024_HTTT       | 0.993  | 0.007  | 0.0284 | 0.9469 |   0.9736 |     1      |     0.9995 |        20 |          20 |
| CTDT_Khoa_2024_18Sep2024_KHMT   | 0.9939 | 0.0061 | 0.0188 | 0.9676 |   0.9853 |     1      |     0.9994 |        35 |          35 |
| CTDT_Khoa_2024_KTPM             | 0.776  | 0.2633 | 0.2137 | 0.8018 |   0.8069 |     0.3342 |     0.2291 |        17 |          22 |

