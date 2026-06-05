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
| CTDT_Khoa_2024_19Sep2024_CNTN   | 0.6096 | 0.5932 | 0.1585 | 0.8247 |   0.7958 |     0.6297 |     0.6127 |        16 |          18 |
| CTDT_Khoa_2024_19Sep2024_TTNT   | 0.5577 | 0.7039 | 0.7588 | 0.5543 |   0.7497 |     0.5972 |     0.5506 |        18 |          20 |
| CTDT_Khoa_2024_18Sep2024_CNTTin | 0.8062 | 0.2249 | 0.2247 | 0.7764 |   0.7688 |     0.5412 |     0.4942 |        25 |          27 |
| CTDT_Khoa_CNTT_K2024_HTTT       | 0.8284 | 0.1716 | 0.1274 | 0.84   |   0.8551 |     0.7082 |     0.6711 |        20 |          19 |
| CTDT_Khoa_2024_18Sep2024_KHMT   | 0.8975 | 0.1025 | 0.0963 | 0.8729 |   0.8402 |     0.7088 |     0.6526 |        35 |          35 |
| CTDT_Khoa_2024_KTPM             | 0.7985 | 0.2141 | 0.2321 | 0.7848 |   0.7831 |     0.6742 |     0.6334 |        17 |          19 |

