import re
import os

with open("mineru/backend/pipeline/batch_analyze.py", "r") as f:
    content = f.read()

# --- Sửa lỗi dùng LLM đọc từng ô nhỏ ---
old_ocr_rec = """            # OCR rec，按照语言分批处理
            for _lang, rec_img_list in rec_img_lang_group.items():
                ocr_engine = atom_model_manager.get_atom_model(
                    atom_model_name=AtomicModel.OCR,
                    det_db_box_thresh=0.5,
                    det_db_unclip_ratio=1.6,
                    lang=_lang,
                    enable_merge_det_boxes=False,
                )
                cropped_img_list = [item["cropped_img"] for item in rec_img_list]
                ocr_res_list = ocr_engine.ocr(cropped_img_list, det=False, tqdm_enable=True, tqdm_desc=f"Table-ocr rec {_lang}")[0]"""

new_ocr_rec = """            # OCR rec，按照语言分批处理
            for _lang, rec_img_list in rec_img_lang_group.items():
                # Dùng trực tiếp mô hình PaddleOCR truyền thống cho lẹ, tránh gọi LLM (LightOnOCR) vào từng dòng chữ
                from mineru.model.ocr.paddle_ocr import PaddleOcr
                ocr_engine = PaddleOcr(lang="vi" if "vi" in _lang else _lang)
                
                cropped_img_list = [item["cropped_img"] for item in rec_img_list]
                ocr_res_list = ocr_engine.ocr(cropped_img_list, det=False, tqdm_enable=True, tqdm_desc=f"Table-cells OCR (Paddle)")[0]"""

if old_ocr_rec in content:
    content = content.replace(old_ocr_rec, new_ocr_rec)
else:
    print("Không tìm thấy old_ocr_rec!")

# --- Sửa tên hiển thị ở thanh tiến trình ---
old_tqdm_desc = """for table_res_dict in tqdm(lighton_tables, desc="LightOnOCR Table with HTML Skeleton"):"""

new_tqdm_desc = """
                import os
                model_name = os.environ.get("OPENAI_MODEL", "LLM OCR").split("/")[-1]
                for table_res_dict in tqdm(lighton_tables, desc=f"{model_name} (with Skeleton)"):"""

if old_tqdm_desc in content:
    content = content.replace(old_tqdm_desc, new_tqdm_desc)
else:
    print("Không tìm thấy old_tqdm_desc!")


with open("mineru/backend/pipeline/batch_analyze.py", "w") as f:
    f.write(content)
print("Đã vá batch_analyze.py thành công!")
