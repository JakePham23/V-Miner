# # import os
# # import sys
# # import json
# # import argparse
# # import re
# # import logging
# # from pathlib import Path

# # # Các thư viện bắt buộc cho bộ Pipeline
# # import markdown
# # import jiwer
# # from rapidfuzz.distance import Levenshtein
# # import nltk
# # from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
# # from nltk.translate.meteor_score import meteor_score
# # from table_recognition_metric import TEDS
# # from tabulate import tabulate

# # try:
# #     import google.generativeai as genai
# #     from openai import OpenAI
# # except ImportError:
# #     pass

# # # Đảm bảo dữ liệu NLTK được tải xuống đầy đủ
# # def setup_nltk():
# #     try:
# #         nltk.data.find('tokenizers/punkt')
# #     except LookupError:
# #         nltk.download('punkt', quiet=True)
# #     try:
# #         nltk.data.find('tokenizers/punkt_tab')
# #     except LookupError:
# #         nltk.download('punkt_tab', quiet=True)
# #     try:
# #         nltk.data.find('corpora/wordnet')
# #     except LookupError:
# #         nltk.download('wordnet', quiet=True)
# #         nltk.download('omw-1.4', quiet=True)

# # logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# # def clean_html_tags(text):
# #     """Xóa bỏ các thẻ HTML sinh ra trong quá trình convert bảng 
# #     để tránh làm nhiễu chỉ số CER/WER của phần text nền."""
# #     clean_pattern = re.compile(r'<[^>]+>')
# #     return clean_pattern.sub(' ', text)

# # def markdown_to_html_table(md_table_text):
# #     """Converts a Markdown table to an HTML table strictly."""
# #     try:
# #         md_table_text = md_table_text.strip()
# #         html = markdown.markdown(md_table_text, extensions=['tables'])
# #         m = re.search(r'<table.*?>.*?</table>', html, re.DOTALL | re.IGNORECASE)
# #         if m:
# #             return m.group(0)
        
# #         html_fallback = markdown.markdown(f"\n\n{md_table_text}\n\n", extensions=['tables'])
# #         m_fallback = re.search(r'<table.*?>.*?</table>', html_fallback, re.DOTALL | re.IGNORECASE)
# #         if m_fallback:
# #             return m_fallback.group(0)
            
# #         return ""
# #     except Exception:
# #         return ""

# # def extract_tables(markdown_content):
# #     """
# #     Trích xuất cấu trúc bảng biểu (HTML và MD format).
# #     Trả về (văn_bản_sạch, danh_sách_bảng_html)
# #     """
# #     html_tables = []
    
# #     html_pattern = re.compile(r'(<table.*?>.*?</table>)', re.DOTALL | re.IGNORECASE)
# #     def replace_html(match):
# #         html_tables.append(match.group(1))
# #         return " [TABLE_PLACEHOLDER] "
    
# #     content = html_pattern.sub(replace_html, markdown_content)
    
# #     md_table_pattern = re.compile(r'(^\|[^\n]+\|\n)((?:\|[ \-:\|]+\|\n)+)((\|[^\n]+\|\n?)+)', re.MULTILINE)    
# #     def replace_md(match):
# #         md_text = match.group(0)
# #         html_table = markdown_to_html_table(md_text)
# #         if html_table:
# #             html_tables.append(html_table)
# #             return " [TABLE_PLACEHOLDER] "
# #         return md_text
    
# #     content = md_table_pattern.sub(replace_md, content)
    
# #     return content, html_tables

# # def evaluate_text(gt_text, pred_text):
# #     """Đánh giá chất lượng văn bản sử dụng CER, WER, Edit Distance, BLEU, METEOR."""
# #     metrics = {}
    
# #     gt_clean = clean_html_tags(gt_text)
# #     pred_clean = clean_html_tags(pred_text)
    
# #     gt_clean = re.sub(r'\s+', ' ', gt_clean).strip()
# #     pred_clean = re.sub(r'\s+', ' ', pred_clean).strip()
    
# #     if not gt_clean and not pred_clean:
# #         return {"cer": 0.0, "wer": 0.0, "edit_distance_sim": 1.0, "bleu": 1.0, "meteor": 1.0}
# #     if not gt_clean or not pred_clean:
# #         return {"cer": 1.0, "wer": 1.0, "edit_distance_sim": 0.0, "bleu": 0.0, "meteor": 0.0}
    
# #     metrics['cer'] = jiwer.cer(gt_clean, pred_clean)
# #     metrics['wer'] = jiwer.wer(gt_clean, pred_clean)
    
# #     dist = Levenshtein.distance(gt_clean, pred_clean)
# #     max_len = max(len(gt_clean), len(pred_clean))
# #     metrics['edit_distance_sim'] = 1.0 - (dist / max_len) if max_len > 0 else 1.0
    
# #     gt_tokens = nltk.word_tokenize(gt_clean)
# #     pred_tokens = nltk.word_tokenize(pred_clean)
    
# #     cc = SmoothingFunction().method1
# #     metrics['bleu'] = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=cc)
    
# #     try:
# #         metrics['meteor'] = meteor_score([gt_tokens], pred_tokens)
# #     except Exception:
# #         metrics['meteor'] = 0.0
        
# #     return metrics

# # def evaluate_tables(gt_tables, pred_tables):
# #     """Evaluates tables using TEDS với cơ chế chuẩn hóa HTML Document chống sập cây DOM."""
# #     if not gt_tables and not pred_tables:
# #         return {"teds_struct": 1.0, "teds_content": 1.0, "table_match_count": 0}
# #     if not gt_tables or not pred_tables:
# #         return {"teds_struct": 0.0, "teds_content": 0.0, "table_match_count": 0}
        
# #     teds_struct_evaluator = TEDS(structure_only=True)
# #     teds_content_evaluator = TEDS(structure_only=False)
    
# #     struct_scores = []
# #     content_scores = []
    
# #     min_len = min(len(gt_tables), len(pred_tables))
# #     for i in range(min_len):
# #         try:
# #             gt_html = re.sub(r'\s+', ' ', gt_tables[i]).strip()
# #             pred_html = re.sub(r'\s+', ' ', pred_tables[i]).strip()
            
# #             if not gt_html.startswith("<html>"):
# #                 gt_html = f"<html><body>{gt_html}</body></html>"
# #             if not pred_html.startswith("<html>"):
# #                 pred_html = f"<html><body>{pred_html}</body></html>"
            
# #             struct_scores.append(teds_struct_evaluator(pred_html, gt_html))
# #             content_scores.append(teds_content_evaluator(pred_html, gt_html))
# #         except Exception as e:
# #             logging.warning(f"TEDS Exception caught at table index {i}: {e}")
# #             struct_scores.append(0.0)
# #             content_scores.append(0.0)
            
# #     max_len = max(len(gt_tables), len(pred_tables))
# #     for _ in range(max_len - min_len):
# #         struct_scores.append(0.0)
# #         content_scores.append(0.0)
        
# #     avg_struct = sum(struct_scores) / len(struct_scores) if struct_scores else 0.0
# #     avg_content = sum(content_scores) / len(content_scores) if content_scores else 0.0
    
# #     return {
# #         "teds_struct": avg_struct,
# #         "teds_content": avg_content,
# #         "table_match_count": min_len
# #     }

# # def split_by_hierarchy(gt_text, pred_text):
# #     """
# #     Bóc tách tài liệu theo phân cấp Heading và ghép cặp đồng bộ giữa GT và Test.
# #     Trả về danh sách các dict chứa tên mục, nội dung GT, nội dung Test.
# #     """
# #     # Tìm tất cả các dòng heading ở file Ground Truth làm điểm mốc
# #     heading_pattern = re.compile(r'^\s*(#{1,6})\s+([^\n]+)', re.MULTILINE)
# #     gt_matches = list(heading_pattern.finditer(gt_text))
    
# #     if not gt_matches:
# #         # Nếu tài liệu phẳng không có heading, trả về nguyên trạng như 1 phân đoạn lớn
# #         return [{"heading": "Toàn bộ tài liệu", "gt_chunk": gt_text, "pred_chunk": pred_text}]
        
# #     sections = []
# #     for idx, match in enumerate(gt_matches):
# #         start_pos = match.start()
# #         end_pos = gt_matches[idx+1].start() if idx + 1 < len(gt_matches) else len(gt_text)
        
# #         heading_title = match.group(2).strip()
# #         gt_chunk = gt_text[start_pos:end_pos].strip()
        
# #         # Định vị phân đoạn tương ứng bên file Test dựa trên tiêu đề mỏ neo
# #         pred_start = pred_text.find(heading_title)
# #         if pred_start != -1:
# #             # Tìm vị trí tiêu đề kế tiếp trong file Test để chặn điểm cuối
# #             next_pred_start = len(pred_text)
# #             if idx + 1 < len(gt_matches):
# #                 next_heading_title = gt_matches[idx+1].group(2).strip()
# #                 find_next = pred_text.find(next_heading_title, pred_start + len(heading_title))
# #                 if find_next != -1:
# #                     next_pred_start = find_next
# #             pred_chunk = pred_text[pred_start:next_pred_start].strip()
# #         else:
# #             pred_chunk = "" # Không tìm thấy tiêu đề tương ứng bên file Test (Lỗi Layout)

# #         # Chỉ đưa vào đánh giá nếu phân đoạn chứa nội dung thực tế (tránh tiêu đề trống)
# #         if len(gt_chunk) > len(heading_title) + 5:
# #             sections.append({
# #                 "heading": heading_title,
# #                 "gt_chunk": gt_chunk,
# #                 "pred_chunk": pred_chunk
# #             })
            
# #     return sections

# # def run_llm_judge(heading, gt_text, pred_text, provider, model, api_key, base_url):
# #     """Gọi LLM Judge đánh giá cho từng phân đoạn cục bộ."""
# #     prompt = f"""Bạn là một chuyên gia đánh giá chất lượng số hóa tài liệu (Document Conversion QA Expert).
# # Nhiệm vụ của bạn là so sánh văn bản kết quả (Prediction) từ công cụ nhận diện tài liệu (OCR / PDF-to-Markdown) với văn bản gốc chuẩn (Ground Truth) của mục cụ thể.

# # Mục đang đánh giá: {heading}

# # Dưới đây là văn bản gốc chuẩn (Ground Truth):
# # --- START GROUND TRUTH ---
# # {gt_text[:8000]}
# # --- END GROUND TRUTH ---

# # Dưới đây là văn bản kết quả nhận diện (Prediction):
# # --- START PREDICTION ---
# # {pred_text[:8000]}
# # --- END PREDICTION ---

# # Hãy đánh giá chất lượng của bản nhận diện so với bản gốc trên thang điểm từ 1.0 đến 10.0 cho các tiêu chí sau:
# # 1. text_extraction (Độ chính xác của văn bản): Đánh giá việc mất chữ, sai lỗi chính tả, sai dấu tiếng Việt, nhầm ký tự.
# # 2. layout_and_reading_order (Cấu trúc và thứ tự đọc): Đánh giá việc giữ định dạng (heading, list, paragraph), thứ tự đọc có bị xáo trộn không.
# # 3. tables (Độ chính xác của bảng biểu): Đánh giá cấu trúc bảng biểu, căn hàng/cột (nếu có). Nếu không có bảng biểu trong tài liệu, cho điểm là null.
# # 4. formulas (Độ chính xác của công thức toán học): Đánh giá ký tự toán học, công thức LaTeX. Nếu không có công thức, cho điểm là null.
# # 5. overall_score (Điểm tổng quan): Điểm đánh giá tổng hợp chung cho phân đoạn này.

# # Yêu cầu đầu ra PHẢI là một chuỗi JSON hợp lệ với cấu trúc sau:
# # {{
# #   "text_extraction": {{"score": 9.5, "rationale": "..."}},
# #   "layout_and_reading_order": {{"score": 9.0, "rationale": "..."}},
# #   "tables": {{"score": null, "rationale": "Không có bảng"}},
# #   "formulas": {{"score": null, "rationale": "Không có công thức"}},
# #   "overall_score": {{"score": 9.0, "rationale": "..."}}
# # }}"""

# #     try:
# #         if provider in ["openai", "ollama"]:
# #             client = OpenAI(api_key=api_key or "dummy", base_url=base_url)
# #             extra_args = {}
# #             if provider == "openai" and "localhost" not in (base_url or ""):
# #                 extra_args["response_format"] = {"type": "json_object"}
                
# #             response = client.chat.completions.create(
# #                 model=model,
# #                 messages=[{"role": "user", "content": prompt}],
# #                 temperature=0.0,
# #                 **extra_args
# #             )
# #             content = response.choices[0].message.content
# #         elif provider == "gemini":
# #             genai.configure(api_key=api_key)
# #             genai_model = genai.GenerativeModel(model)
# #             response = genai_model.generate_content(
# #                 prompt,
# #                 generation_config=genai.GenerationConfig(
# #                     response_mime_type="application/json",
# #                     temperature=0.0
# #                 )
# #             )
# #             content = response.text
# #         else:
# #             return {"error": f"Unsupported LLM provider: {provider}"}
            
# #         content_clean = content.strip()
# #         if content_clean.startswith("```json"):
# #             content_clean = content_clean.split("```json")[-1].split("```")[0].strip()
# #         elif content_clean.startswith("```"):
# #             content_clean = content_clean.split("```")[1].split("```")[0].strip()
            
# #         return json.loads(content_clean)
# #     except Exception as e:
# #         return {"error": str(e)}

# # def pair_files(gt_dir, test_dir):
# #     gt_files = list(Path(gt_dir).rglob("*.md"))
# #     test_files = list(Path(test_dir).rglob("*.md"))
    
# #     pairs = []
# #     for gt_file in gt_files:
# #         basename = gt_file.stem
# #         matched_test = None
# #         for tf in test_files:
# #             if tf.name == gt_file.name:
# #                 matched_test = tf
# #                 break
# #             if tf.stem.startswith(basename):
# #                 matched_test = tf
        
# #         if matched_test:
# #             pairs.append((gt_file, matched_test))
# #             test_files.remove(matched_test)
# #         else:
# #             logging.warning(f"No match found for GT file: {gt_file}")
            
# #     return pairs

# # def main():
# #     env_provider = os.environ.get("LLM_SERVICE", "gemini")
# #     env_api_key = os.environ.get("OPENAI_API_KEY") if env_provider == "openai" else os.environ.get("GEMINI_API_KEY")
# #     env_base_url = os.environ.get("OPENAI_API_BASE", "http://localhost:1234/v1")
# #     env_model = os.environ.get("OPENAI_MODEL_NAME", "qwen2-vl-2b-instruct" if env_provider == "openai" else "gemini-1.5-flash")

# #     parser = argparse.ArgumentParser(description="Evaluate Hierarchical Document Parser Benchmark")
# #     parser.add_argument("--gt-dir", required=True, help="Ground truth directory")
# #     parser.add_argument("--test-dir", required=True, help="Test predictions directory")
# #     parser.add_argument("--output-dir", required=True, help="Output directory for reports")
# #     parser.add_argument("--mode", choices=["traditional", "llm", "all"], default="traditional", help="Evaluation mode")
# #     parser.add_argument("--llm-provider", default=env_provider, choices=["gemini", "openai", "ollama"], help="LLM Provider")
# #     parser.add_argument("--llm-model", default=env_model, help="LLM Model name")
# #     parser.add_argument("--llm-api-key", default=env_api_key, help="API Key")
# #     parser.add_argument("--llm-base-url", default=env_base_url, help="Base URL for OpenAI/Ollama")
# #     args = parser.parse_args()
    
# #     setup_nltk()
# #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
# #     pairs = pair_files(args.gt_dir, args.test_dir)
# #     logging.info(f"Found {len(pairs)} file pairs for evaluation.")
    
# #     results = {}
    
# #     for gt_path, pred_path in pairs:
# #         doc_name = gt_path.stem
# #         logging.info(f"Evaluating document: {doc_name}...")
        
# #         with open(gt_path, 'r', encoding='utf-8') as f:
# #             gt_text = f.read()
# #         with open(pred_path, 'r', encoding='utf-8') as f:
# #             pred_text = f.read()
            
# #         # Chia tách tài liệu ra từng phân cấp Heading nhỏ để đánh giá cục bộ
# #         sub_sections = split_by_hierarchy(gt_text, pred_text)
# #         logging.info(f"Detected {len(sub_sections)} hierarchical sub-sections for evaluation.")
        
# #         # Biến tích lũy dùng để tính trung bình tổng hợp (Aggregation)
#         total_gt_len = 0
#         agg_metrics = {
#             "sim": 0.0, "cer": 0.0, "wer": 0.0, "bleu": 0.0, "meteor": 0.0,
#             "teds_str": 0.0, "teds_con": 0.0, "table_count": 0,
#             "llm_text": 0.0, "llm_layout": 0.0, "llm_tables": 0.0, "llm_overall": 0.0,
#             "llm_table_weights": 0
#         }
        
#         # Pre-compute LLM results concurrently to save time
#         if args.mode in ["llm", "all"]:
#             import concurrent.futures
#             logging.info("Starting concurrent LLM API requests...")
#             with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
#                 future_to_idx = {
#                     executor.submit(
#                         run_llm_judge, 
#                         sec["heading"], sec["gt_chunk"], sec["pred_chunk"], 
#                         args.llm_provider, args.llm_model, args.llm_api_key, args.llm_base_url
#                     ): idx for idx, sec in enumerate(sub_sections)
#                 }
#                 for future in concurrent.futures.as_completed(future_to_idx):
#                     idx = future_to_idx[future]
#                     sub_sections[idx]["llm_result"] = future.result()
#                     logging.info(f"  [LLM] Finished chunk {idx+1}/{len(sub_sections)}: '{sub_sections[idx]['heading']}'")
        
#         section_results = []
        
# #         # Biến tích lũy dùng để tính trung bình tổng hợp (Aggregation)
# #         total_gt_len = 0
# #         agg_metrics = {
# #             "sim": 0.0, "cer": 0.0, "wer": 0.0, "bleu": 0.0, "meteor": 0.0,
# #             "teds_str": 0.0, "teds_con": 0.0, "table_count": 0,
# #             "llm_text": 0.0, "llm_layout": 0.0, "llm_tables": 0.0, "llm_overall": 0.0,
# #             "llm_table_weights": 0
# #         }
        
# #         for idx, sec in enumerate(sub_sections):
# #             heading = sec["heading"]
# #             gt_chunk = sec["gt_chunk"]
# #             pred_chunk = sec["pred_chunk"]
            
# #             chunk_len = len(gt_chunk)
# #             total_gt_len += chunk_len
            
# #             chunk_res = {"heading": heading, "weight_chars": chunk_len}
            
# #             # 1. Đánh giá thuật toán truyền thống cho từng mục
# #             if args.mode in ["traditional", "all"]:
# #                 gt_clean, gt_tables = extract_tables(gt_chunk)
# #                 pred_clean, pred_tables = extract_tables(pred_chunk)
                
# #                 text_m = evaluate_text(gt_clean, pred_clean)
# #                 table_m = evaluate_tables(gt_tables, pred_tables)
                
# #                 chunk_res["traditional"] = {**text_m, **table_m}
                
# #                 # Tích lũy điểm có trọng số theo độ dài phân đoạn
# #                 agg_metrics["sim"] += text_m["edit_distance_sim"] * chunk_len
# #                 agg_metrics["cer"] += text_m["cer"] * chunk_len
# #                 agg_metrics["wer"] += text_m["wer"] * chunk_len
# #                 agg_metrics["bleu"] += text_m["bleu"] * chunk_len
# #                 agg_metrics["meteor"] += text_m["meteor"] * chunk_len
                
# #                 if table_m["table_match_count"] > 0:
# #                     agg_metrics["teds_str"] += table_m["teds_struct"] * table_m["table_match_count"]
# #                     agg_metrics["teds_con"] += table_m["teds_content"] * table_m["table_match_count"]
# #                     agg_metrics["table_count"] += table_m["table_match_count"]

# #             # 2. Đánh giá bằng LLM Judge cho từng mục
#             if args.mode in ["llm", "all"]:
#                 llm_m = sec.get("llm_result", {"error": "Not evaluated"})
#                 chunk_res["llm"] = llm_m
                
#                 if "error" not in llm_m:
#                     agg_metrics["llm_text"] += float(llm_m.get("text_extraction", {}).get("score" or 0, 0)) * chunk_len
#                     agg_metrics["llm_layout"] += float(llm_m.get("layout_and_reading_order", {}).get("score" or 0, 0)) * chunk_len
#                     agg_metrics["llm_overall"] += float(llm_m.get("overall_score", {}).get("score" or 0, 0)) * chunk_len
                    
# #                     t_score = llm_m.get("tables", {}).get("score")
# #                     if t_score is not None:
# #                         agg_metrics["llm_tables"] += float(t_score) * chunk_len
# #                         agg_metrics["llm_table_weights"] += chunk_len

# #             section_results.append(chunk_res)
            
# #         # Tính toán điểm tổng hợp (Final Aggregated Score) sau khi duyệt hết phân cấp
# #         if total_gt_len > 0:
# #             final_traditional = {
# #                 "edit_distance_sim": agg_metrics["sim"] / total_gt_len,
# #                 "cer": agg_metrics["cer"] / total_gt_len,
# #                 "wer": agg_metrics["wer"] / total_gt_len,
# #                 "bleu": agg_metrics["bleu"] / total_gt_len,
# #                 "meteor": agg_metrics["meteor"] / total_gt_len,
# #                 # FIX: Đổi từ agg_metrics["teds_struct"] thành agg_metrics["teds_str"]
# #                 "teds_struct": agg_metrics["teds_str"] / agg_metrics["table_count"] if agg_metrics["table_count"] > 0 else 1.0,
# #                 "teds_content": agg_metrics["teds_con"] / agg_metrics["table_count"] if agg_metrics["table_count"] > 0 else 1.0,
# #                 "table_match_count": agg_metrics["table_count"]
# #             }
# #             final_llm = {
# #                 "text_extraction": {"score": agg_metrics["llm_text"] / total_gt_len},
# #                 "layout_and_reading_order": {"score": agg_metrics["llm_layout"] / total_gt_len},
# #                 "tables": {"score": agg_metrics["llm_tables"] / agg_metrics["llm_table_weights"] if agg_metrics["llm_table_weights"] > 0 else "N/A"},
# #                 "overall_score": {"score": agg_metrics["llm_overall"] / total_gt_len}
# #             }
# #         else:
# #             final_traditional, final_llm = {}, {}

# #         results[doc_name] = {
# #             "aggregated": {"traditional": final_traditional, "llm": final_llm},
# #             "detailed_sections": section_results
# #         }
        
# #     # --- XUẤT FILE BÁO CÁO REPORT ---
# #     json_path = Path(args.output_dir) / "eval_hierarchical_report.json"
# #     with open(json_path, 'w', encoding='utf-8') as f:
# #         json.dump(results, f, ensure_ascii=False, indent=2)
        
# #     md_path = Path(args.output_dir) / "eval_hierarchical_report.md"
# #     with open(md_path, 'w', encoding='utf-8') as f:
# #         f.write("# Hierarchical Benchmark Evaluation Report\n\n")
# #         f.write(f"**Mode:** `{args.mode}`\n\n")
        
# #         for doc, data in results.items():
# #             f.write(f"## Document: {doc}\n\n")
            
# #             # Bảng tổng hợp kết quả (Aggregated Summary)
# #             f.write("### 📊 Final Aggregated Summary (Tổng hợp toàn tài liệu)\n\n")
# #             if args.mode in ["traditional", "all"]:
# #                 t = data["aggregated"]["traditional"]
# #                 f.write(f"- **Sim (Edit Distance)**: {t.get('edit_distance_sim', 0):.4f}\n")
# #                 f.write(f"- **CER / WER**: {t.get('cer', 0):.4f} / {t.get('wer', 0):.4f}\n")
# #                 f.write(f"- **BLEU / METEOR**: {t.get('bleu', 0):.4f} / {t.get('meteor', 0):.4f}\n")
# #                 f.write(f"- **TEDS-Str / TEDS-Con**: {t.get('teds_struct', 0):.4f} / {t.get('teds_content', 0):.4f}\n")
# #             if args.mode in ["llm", "all"]:
# #                 l = data["aggregated"]["llm"]
# #                 f.write(f"- **LLM Text Score**: {l.get('text_extraction', {}).get('score', 0):.2f}/10\n")
# #                 f.write(f"- **LLM Layout Score**: {l.get('layout_and_reading_order', {}).get('score', 0):.2f}/10\n")
# #                 f.write(f"- **LLM Tables Score**: {l.get('tables', {}).get('score', 0)}\n")
# #                 f.write(f"- **LLM Overall Score**: {l.get('overall_score', {}).get('score', 0):.2f}/10\n")
# #             f.write("\n---\n\n")
            
# #             # Bảng chi tiết từng phân cấp mục con (Detailed breakdown)
# #             f.write("### 🔍 Detailed Breakdown by Sections (Chi tiết theo từng phân cấp)\n\n")
# #             headers = ["Section Heading", "Weight (Chars)", "Sim", "CER", "WER", "TEDS-Str", "LLM Text", "LLM Layout", "LLM Overall"]
# #             table_data = []
            
# #             for sec in data["detailed_sections"]:
# #                 t_m = sec.get("traditional", {})
# #                 l_m = sec.get("llm", {})
                
# #                 # Ép kiểu dữ liệu an toàn để đưa vào bảng hiển thị
# #                 llm_txt = l_m.get("text_extraction", {}).get("score", "N/A") if "error" not in l_m else "Err"
# #                 llm_lay = l_m.get("layout_and_reading_order", {}).get("score", "N/A") if "error" not in l_m else "Err"
# #                 llm_ovr = l_m.get("overall_score", {}).get("score", "N/A") if "error" not in l_m else "Err"
                
# #                 table_data.append([
# #                     sec["heading"][:40], # Rút gọn tên heading dài để bảng không gãy dòng
# #                     sec["weight_chars"],
# #                     f"{t_m.get('edit_distance_sim', 0):.4f}" if t_m else "N/A",
# #                     f"{t_m.get('cer', 0):.4f}" if t_m else "N/A",
# #                     f"{t_m.get('wer', 0):.4f}" if t_m else "N/A",
# #                     f"{t_m.get('teds_struct', 0):.4f}" if t_m else "N/A",
# #                     f"{llm_txt:.1f}" if isinstance(llm_txt, (int, float)) else str(llm_txt),
# #                     f"{llm_lay:.1f}" if isinstance(llm_lay, (int, float)) else str(llm_lay),
# #                     f"{llm_ovr:.1f}" if isinstance(llm_ovr, (int, float)) else str(llm_ovr)
# #                 ])
                
# #             f.write(tabulate(table_data, headers=headers, tablefmt="pipe"))
# #             f.write("\n\n")

# #     logging.info(f"Evaluation complete. Hierarchical reports saved to {args.output_dir}")

# # if __name__ == "__main__":
# #     main()

import os
import sys
import json
import argparse
import re
import logging
from pathlib import Path

try:
    import markdown
except ImportError:
    pass
try:
    import jiwer
except ImportError:
    pass
try:
    from rapidfuzz.distance import Levenshtein
except ImportError:
    pass
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.translate.meteor_score import meteor_score
except ImportError:
    pass
try:
    from table_recognition_metric import TEDS
except ImportError:
    pass
try:
    from tabulate import tabulate
except ImportError:
    pass
try:
    import google.generativeai as genai
    from openai import OpenAI
except ImportError:
    pass

# Đảm bảo dữ liệu NLTK được tải xuống đầy đủ
def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def clean_html_tags(text):
    """Xóa bỏ các thẻ HTML sinh ra trong quá trình convert bảng 
    để tránh làm nhiễu chỉ số CER/WER của phần text nền."""
    clean_pattern = re.compile(r'<[^>]+>')
    return clean_pattern.sub(' ', text)

def markdown_to_html_table(md_table_text):
    """Converts a Markdown table to an HTML table strictly."""
    try:
        # Đảm bảo chuỗi truyền vào không bị trống dòng thừa thãi ở đầu/cuối
        md_table_text = md_table_text.strip()
        
        html = markdown.markdown(md_table_text, extensions=['tables'])
        # Trích xuất chính xác thẻ <table>
        m = re.search(r'<table.*?>.*?</table>', html, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(0)
        
        # Nếu thư viện markdown lỗi không ra thẻ table, dùng giải pháp thay thế thủ công (Fallback)
        # Bọc chuỗi trơn để kích hoạt parser nội bộ của markdown
        html_fallback = markdown.markdown(f"\n\n{md_table_text}\n\n", extensions=['tables'])
        m_fallback = re.search(r'<table.*?>.*?</table>', html_fallback, re.DOTALL | re.IGNORECASE)
        if m_fallback:
            return m_fallback.group(0)
            
        return "" # Trả về chuỗi rỗng nếu hoàn toàn không phải cấu trúc bảng
    except Exception:
        return ""

def extract_tables(markdown_content):
    """
    Trích xuất cấu trúc bảng biểu (HTML và MD format).
    Trả về (văn_bản_sạch, danh_sách_bảng_html)
    """
    html_tables = []
    
    # 1. Trích xuất các bảng HTML tồn tại sẵn
    html_pattern = re.compile(r'(<table.*?>.*?</table>)', re.DOTALL | re.IGNORECASE)
    def replace_html(match):
        html_tables.append(match.group(1))
        return " [TABLE_PLACEHOLDER] "
    
    content = html_pattern.sub(replace_html, markdown_content)
    
    # 2. Trích xuất bảng Markdown bằng Regex đã sửa lỗi đa cột
    md_table_pattern = re.compile(r'(^\|[^\n]+\|\n)((?:\|[ \-:\|]+\|\n)+)((\|[^\n]+\|\n?)+)', re.MULTILINE)    
    def replace_md(match):
        md_text = match.group(0)
        html_table = markdown_to_html_table(md_text)
        if html_table: # CHỈ append khi html_table khác rỗng (tức là có thẻ <table> thực sự)
            html_tables.append(html_table)
            return " [TABLE_PLACEHOLDER] "
        return md_text # Nếu không phải bảng, trả lại nguyên vẹn text gốc, không thay placeholder
    
    content = md_table_pattern.sub(replace_md, content)
    
    return content, html_tables

def evaluate_text(gt_text, pred_text):
    """Đánh giá chất lượng văn bản sử dụng CER, WER, Edit Distance, BLEU, METEOR."""
    metrics = {}
    
    # Xóa các thẻ HTML rác dính lại để đảm bảo công bằng cho CER
    gt_clean = clean_html_tags(gt_text)
    pred_clean = clean_html_tags(pred_text)
    
    # Chuẩn hóa khoảng trắng
    gt_clean = re.sub(r'\s+', ' ', gt_clean).strip()
    pred_clean = re.sub(r'\s+', ' ', pred_clean).strip()
    
    if not gt_clean and not pred_clean:
        return {"cer": 0.0, "wer": 0.0, "edit_distance_sim": 1.0, "bleu": 1.0, "meteor": 1.0}
    if not gt_clean or not pred_clean:
        return {"cer": 1.0, "wer": 1.0, "edit_distance_sim": 0.0, "bleu": 0.0, "meteor": 0.0}
    
    # Tính toán chỉ số JiWER
    metrics['cer'] = jiwer.cer(gt_clean, pred_clean)
    metrics['wer'] = jiwer.wer(gt_clean, pred_clean)
    
    # Tính Khoảng cách chỉnh sửa Normalized Edit Distance bằng RapidFuzz
    dist = Levenshtein.distance(gt_clean, pred_clean)
    max_len = max(len(gt_clean), len(pred_clean))
    metrics['edit_distance_sim'] = 1.0 - (dist / max_len) if max_len > 0 else 1.0
    
    # Phân tách Token từ NLTK
    gt_tokens = nltk.word_tokenize(gt_clean)
    pred_tokens = nltk.word_tokenize(pred_clean)
    
    # BLEU Score (Cumulative 4-gram)
    cc = SmoothingFunction().method1
    metrics['bleu'] = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=cc)
    
    # METEOR Score
    try:
        metrics['meteor'] = meteor_score([gt_tokens], pred_tokens)
    except Exception:
        metrics['meteor'] = 0.0
        
    return metrics

def evaluate_tables(gt_tables, pred_tables):
    """Evaluates tables using TEDS với cơ chế chuẩn hóa HTML Document chống sập cây DOM."""
    if not gt_tables and not pred_tables:
        return {"teds_struct": 1.0, "teds_content": 1.0, "table_match_count": 0, "gt_table_count": 0, "pred_table_count": 0}
    if not gt_tables or not pred_tables:
        return {"teds_struct": 0.0, "teds_content": 0.0, "table_match_count": 0, "gt_table_count": len(gt_tables), "pred_table_count": len(pred_tables)}
        
    teds_struct_evaluator = TEDS(structure_only=True)
    teds_content_evaluator = TEDS(structure_only=False)
    
    struct_scores = []
    content_scores = []
    
    min_len = min(len(gt_tables), len(pred_tables))
    for i in range(min_len):
        try:
            # Loại bỏ các khoảng trắng thừa thãi hoặc ký tự xuống dòng gây lệch cấu trúc chuỗi
            gt_html = re.sub(r'\s+', ' ', gt_tables[i]).strip()
            pred_html = re.sub(r'\s+', ' ', pred_tables[i]).strip()
            
            # Ép cấu trúc về định dạng HTML Document chuẩn để trình XML Parser của TEDS không bị Exception
            if not gt_html.startswith("<html>"):
                gt_html = f"<html><body>{gt_html}</body></html>"
            if not pred_html.startswith("<html>"):
                pred_html = f"<html><body>{pred_html}</body></html>"
            
            # Gọi trực tiếp đối tượng (Callable format) theo đúng chuẩn của thư viện table_recognition_metric
            struct_scores.append(teds_struct_evaluator(pred_html, gt_html))
            content_scores.append(teds_content_evaluator(pred_html, gt_html))
        except Exception as e:
            logging.warning(f"TEDS Exception caught at table index {i}: {e}")
            struct_scores.append(0.0)
            content_scores.append(0.0)
            
    max_len = max(len(gt_tables), len(pred_tables))
    for _ in range(max_len - min_len):
        struct_scores.append(0.0)
        content_scores.append(0.0)
        
    avg_struct = sum(struct_scores) / len(struct_scores) if struct_scores else 0.0
    avg_content = sum(content_scores) / len(content_scores) if content_scores else 0.0
    
    return {
        "teds_struct": avg_struct,
        "teds_content": avg_content,
        "table_match_count": min_len,
        "gt_table_count": len(gt_tables),
        "pred_table_count": len(pred_tables)
    }

def get_synchronized_chunks(gt_text, pred_text, max_chars=15000):
    """
    Cắt cả hai file tại cùng một điểm logic dựa trên Heading của Ground Truth
    để đảm bảo LLM Judge nhận được nội dung đồng bộ 100%.
    """
    if len(gt_text) <= max_chars and len(pred_text) <= max_chars:
        return gt_text, pred_text

    gt_cutoff = max_chars
    truncated_gt = gt_text[:max_chars]
    heading_matches = list(re.finditer(r'\n(#{1,6}\s+([^\n]+))\n', truncated_gt))
    
    if heading_matches:
        last_match = heading_matches[-1]
        gt_cutoff = last_match.start()
        anchor_text = last_match.group(2).strip()
        
        pred_cutoff = pred_text.find(anchor_text)
        
        if pred_cutoff == -1:
            first_word = anchor_text.split()[0] if anchor_text.split() else ""
            if first_word and len(first_word) > 2:
                pred_search_zone = pred_text[:int(max_chars * 1.2)]
                word_matches = [m.start() for m in re.finditer(re.escape(first_word), pred_search_zone)]
                if word_matches:
                    pred_cutoff = min(word_matches, key=lambda x: abs(x - max_chars))

        if pred_cutoff != -1 and pred_cutoff > max_chars * 0.3:
            return gt_text[:gt_cutoff].strip(), pred_text[:pred_cutoff].strip()

    sentence_matches = list(re.finditer(r'[\.\?\!]\s', truncated_gt))
    if sentence_matches:
        gt_cutoff = sentence_matches[-1].end()
        gt_tail = gt_text[gt_cutoff-20:gt_cutoff].strip()
        pred_cutoff = pred_text.find(gt_tail)
        if pred_cutoff != -1:
            return gt_text[:gt_cutoff].strip(), pred_text[:pred_cutoff].strip()

    return gt_text[:max_chars], pred_text[:max_chars]

def run_llm_judge(gt_text, pred_text, provider, model, api_key, base_url):
    # Sử dụng hàm cắt thông minh đồng bộ theo mỏ neo Heading
    gt_segmented, pred_segmented = get_synchronized_chunks(gt_text, pred_text, max_chars=15000)

    prompt = f"""Bạn là một chuyên gia đánh giá chất lượng số hóa tài liệu (Document Conversion QA Expert).
Nhiệm vụ của bạn là so sánh văn bản kết quả (Prediction) từ công cụ nhận diện tài liệu (OCR / PDF-to-Markdown) với văn bản gốc chuẩn (Ground Truth).

Dưới đây là văn bản gốc chuẩn (Ground Truth):
--- START GROUND TRUTH ---
{gt_segmented}
--- END GROUND TRUTH ---

Dưới đây là văn bản kết quả nhận diện (Prediction):
--- START PREDICTION ---
{pred_segmented}
--- END PREDICTION ---

Hãy đánh giá chất lượng của bản nhận diện so với bản gốc trên thang điểm từ 1.0 đến 10.0 cho các tiêu chí sau:
1. text_extraction (Độ chính xác của văn bản): Đánh giá việc mất chữ, sai lỗi chính tả, sai dấu tiếng Việt, nhầm ký tự.
2. layout_and_reading_order (Cấu trúc và thứ tự đọc): Đánh giá việc giữ định dạng (heading, list, paragraph), thứ tự đọc có bị xáo trộn không.
3. tables (Độ chính xác của bảng biểu): Đánh giá cấu trúc bảng biểu, căn hàng/cột (nếu có). Nếu không có bảng biểu trong tài liệu, cho điểm là null.
4. formulas (Độ chính xác của công thức toán học): Đánh giá ký tự toán học, công thức LaTeX. Nếu không có công thức, cho điểm là null.
5. overall_score (Điểm tổng quan): Điểm đánh giá tổng hợp chung.

Yêu cầu đầu ra PHẢI là một chuỗi JSON hợp lệ với cấu trúc sau:
{{
  "text_extraction": {{"score": 9.5, "rationale": "..."}},
  "layout_and_reading_order": {{"score": 9.0, "rationale": "..."}},
  "tables": {{"score": null, "rationale": "Không có bảng"}},
  "formulas": {{"score": null, "rationale": "Không có công thức"}},
  "overall_score": {{"score": 9.0, "rationale": "..."}}
}}"""

    try:
        if provider in ["openai", "ollama"]:
            client = OpenAI(api_key=api_key or "dummy", base_url=base_url)
            
            extra_args = {}
            if provider == "openai" and "localhost" not in (base_url or ""):
                extra_args["response_format"] = {"type": "json_object"}
                
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                **extra_args
            )
            content = response.choices[0].message.content
        elif provider == "gemini":
            genai.configure(api_key=api_key)
            genai_model = genai.GenerativeModel(model)
            response = genai_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                )
            )
            content = response.text
        else:
            return {"error": f"Unsupported LLM provider: {provider}"}
            
        content_clean = content.strip()
        if content_clean.startswith("```json"):
            content_clean = content_clean.split("```json")[-1].split("```")[0].strip()
        elif content_clean.startswith("```"):
            content_clean = content_clean.split("```")[1].split("```")[0].strip()
            
        return json.loads(content_clean)
    except Exception as e:
        return {"error": str(e)}

def pair_files(gt_dir, test_dir):
    gt_files = list(Path(gt_dir).rglob("*.md"))
    test_files = list(Path(test_dir).rglob("*.md"))
    
    pairs = []
    for gt_file in gt_files:
        basename = gt_file.stem
        matched_test = None
        for tf in test_files:
            if tf.name == gt_file.name:
                matched_test = tf
                break
            if tf.stem.startswith(basename):
                matched_test = tf
        
        if matched_test:
            pairs.append((gt_file, matched_test))
            test_files.remove(matched_test)
        else:
            logging.warning(f"No match found for GT file: {gt_file}")
            
    return pairs

def main():
    # Đọc cấu hình linh hoạt từ biến môi trường hệ thống
    env_provider = os.environ.get("LLM_SERVICE", "gemini")
    env_api_key = os.environ.get("OPENAI_API_KEY") if env_provider == "openai" else os.environ.get("GEMINI_API_KEY")
    env_base_url = os.environ.get("OPENAI_API_BASE", "http://localhost:1234/v1")
    env_model = os.environ.get("OPENAI_MODEL_NAME", "qwen2-vl-2b-instruct" if env_provider == "openai" else "gemini-1.5-flash")

    parser = argparse.ArgumentParser(description="Evaluate Document Parser Benchmark")
    parser.add_argument("--gt-dir", required=True, help="Ground truth directory")
    parser.add_argument("--test-dir", required=True, help="Test predictions directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for reports")
    parser.add_argument("--mode", choices=["traditional", "llm", "all"], default="traditional", help="Evaluation mode")
    parser.add_argument("--llm-provider", default=env_provider, choices=["gemini", "openai", "ollama"], help="LLM Provider")
    parser.add_argument("--llm-model", default=env_model, help="LLM Model name")
    parser.add_argument("--llm-api-key", default=env_api_key, help="API Key")
    parser.add_argument("--llm-base-url", default=env_base_url, help="Base URL for OpenAI/Ollama")
    args = parser.parse_args()
    
    setup_nltk()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    pairs = pair_files(args.gt_dir, args.test_dir)
    logging.info(f"Found {len(pairs)} file pairs for evaluation.")
    
    results = {}
    
    for gt_path, pred_path in pairs:
        doc_name = gt_path.stem
        logging.info(f"Evaluating {doc_name}...")
        
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_text = f.read()
        with open(pred_path, 'r', encoding='utf-8') as f:
            pred_text = f.read()
            
        doc_results = {}
        
        if args.mode in ["traditional", "all"]:
            gt_clean, gt_tables = extract_tables(gt_text)
            pred_clean, pred_tables = extract_tables(pred_text)
            
            text_metrics = evaluate_text(gt_clean, pred_clean)
            table_metrics = evaluate_tables(gt_tables, pred_tables)
            
            doc_results["traditional"] = {**text_metrics, **table_metrics}
            
        if args.mode in ["llm", "all"]:
            logging.info(f"Running LLM Judge using {args.llm_provider} ({args.llm_model})...")
            llm_metrics = run_llm_judge(gt_text, pred_text, args.llm_provider, args.llm_model, args.llm_api_key, args.llm_base_url)
            doc_results["llm"] = llm_metrics
            
        results[doc_name] = doc_results
        
    json_path = Path(args.output_dir) / "eval_report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    md_path = Path(args.output_dir) / "eval_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Benchmark Evaluation Report\n\n")
        f.write(f"**Mode:** `{args.mode}`\n\n")
        
        if args.mode in ["traditional", "all"]:
            f.write("## Traditional Metrics Summary\n")
            f.write("Metrics evaluated:\n")
            f.write("- **Similarity (Normalized Edit Distance)**: Higher is better.\n")
            f.write("- **CER / WER**: Character/Word Error Rate. Lower is better.\n")
            f.write("- **BLEU / METEOR**: Text semantic matching. Higher is better.\n")
            f.write("- **TEDS (Structure / Content)**: Table layout accuracy. Higher is better.\n")
            f.write("- **GT Tbls / Pred Tbls**: Số lượng bảng gốc (Ground Truth) và số lượng bảng nhận diện được.\n\n")
            
            headers = ["Document", "Sim", "CER", "WER", "BLEU", "METEOR", "TEDS-Str", "TEDS-Con", "GT Tbls", "Pred Tbls"]
            table_data = []
            for doc, res in results.items():
                m = res.get("traditional", {})
                if not m: continue
                table_data.append([
                    doc,
                    f"{m.get('edit_distance_sim', 0):.4f}",
                    f"{m.get('cer', 0):.4f}",
                    f"{m.get('wer', 0):.4f}",
                    f"{m.get('bleu', 0):.4f}",
                    f"{m.get('meteor', 0):.4f}",
                    f"{m.get('teds_struct', 0):.4f}",
                    f"{m.get('teds_content', 0):.4f}",
                    str(m.get('gt_table_count', 0)),
                    str(m.get('pred_table_count', 0))
                ])
            if table_data:
                f.write(tabulate(table_data, headers=headers, tablefmt="pipe"))
                f.write("\n\n")
                
        if args.mode in ["llm", "all"]:
            f.write("## LLM-as-a-Judge Evaluation\n")
            headers = ["Document", "Text", "Layout", "Tables", "Formulas", "Overall"]
            table_data = []
            for doc, res in results.items():
                m = res.get("llm", {})
                if not m or "error" in m: continue
                table_data.append([
                    doc,
                    str(m.get("text_extraction", {}).get("score", "N/A")),
                    str(m.get("layout_and_reading_order", {}).get("score", "N/A")),
                    str(m.get("tables", {}).get("score", "N/A")),
                    str(m.get("formulas", {}).get("score", "N/A")),
                    str(m.get("overall_score", {}).get("score", "N/A"))
                ])
            if table_data:
                f.write(tabulate(table_data, headers=headers, tablefmt="pipe"))
                f.write("\n\n")

    logging.info(f"Evaluation complete. Reports saved to {args.output_dir}")

if __name__ == "__main__":
    main()