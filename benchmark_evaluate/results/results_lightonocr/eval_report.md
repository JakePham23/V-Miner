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
| CTDT_Khoa_2024_19Sep2024_CNTN   | 0.7735 | 0.2781 | 0.2533 | 0.7087 |   0.7962 |     0.7411 |     0.7085 |        16 |          17 |
| CTDT_Khoa_2024_19Sep2024_TTNT   | 0.8078 | 0.224  | 0.2406 | 0.7048 |   0.8009 |     0.9334 |     0.9081 |        18 |          18 |
| CTDT_Khoa_2024_18Sep2024_CNTTin | 0.7897 | 0.2523 | 0.2299 | 0.7036 |   0.7851 |     0.5414 |     0.5122 |        25 |          33 |
| CTDT_Khoa_CNTT_K2024_HTTT       | 0.4998 | 0.8118 | 0.5434 | 0.3079 |   0.7111 |     0.6454 |     0.5919 |        20 |          20 |
| CTDT_Khoa_2024_18Sep2024_KHMT   | 0.8718 | 0.1318 | 0.1363 | 0.7862 |   0.8177 |     0.7135 |     0.6818 |        35 |          38 |
| CTDT_Khoa_2024_KTPM             | 0.7917 | 0.2295 | 0.2604 | 0.7224 |   0.7554 |     0.5986 |     0.5657 |        17 |          19 |

