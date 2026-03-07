import re

with open("mineru/backend/pipeline/batch_analyze.py", "r") as f:
    content = f.read()

old_block1 = """            # Process LightOnOCR tables directly
            if lighton_tables:
                from mineru.model.ocr.lighton_ocr import LightOnOCR
                # Config đọc từ env vars (LLM_SERVICE / OPENAI_API_BASE / OPENAI_MODEL ...)
                lighton_ocr = LightOnOCR()
                for table_res_dict in tqdm(lighton_tables, desc="LightOnOCR Table"):
                    bgr_image = cv2.cvtColor(table_res_dict["table_img"], cv2.COLOR_RGB2BGR)
                    _tbl_lang = table_res_dict.get('lang', '')
                    # Truyền lang để tự phát hiện bảng tiếng Việt và dùng VI prompt
                    html_result = lighton_ocr.recognize_table(bgr_image, lang=_tbl_lang)
                    table_res_dict["table_res"]["html"] = html_result
                    # Mark as processed to skip regular table processing
                    table_res_dict["lighton_processed"] = True"""

new_block1 = """            use_wired_skeleton = os.environ.get("USE_WIRED_TABLE_SKELETON", "0") == "1"
            
            # Process LightOnOCR tables directly IF NOT using skeleton mode
            if lighton_tables and not use_wired_skeleton:
                from mineru.model.ocr.lighton_ocr import LightOnOCR
                # Config đọc từ env vars (LLM_SERVICE / OPENAI_API_BASE / OPENAI_MODEL ...)
                lighton_ocr = LightOnOCR()
                for table_res_dict in tqdm(lighton_tables, desc="LightOnOCR Table"):
                    bgr_image = cv2.cvtColor(table_res_dict["table_img"], cv2.COLOR_RGB2BGR)
                    _tbl_lang = table_res_dict.get('lang', '')
                    # Truyền lang để tự phát hiện bảng tiếng Việt và dùng VI prompt
                    html_result = lighton_ocr.recognize_table(bgr_image, lang=_tbl_lang)
                    table_res_dict["table_res"]["html"] = html_result
                    # Mark as processed to skip regular table processing
                    table_res_dict["lighton_processed"] = True"""

if old_block1 in content:
    content = content.replace(old_block1, new_block1)
else:
    print("Không tìm thấy old_block1!")


old_block2 = """                # 检查html_code是否包含'<table>'和'</table>'
                if "<table>" in html_code and "</table>" in html_code:
                    # 选用<table>到</table>的内容，放入table_res_dict['table_res']['html']
                    start_index = html_code.find("<table>")
                    end_index = html_code.rfind("</table>") + len("</table>")
                    table_res_dict["table_res"]["html"] = html_code[start_index:end_index]"""

new_block2 = """                # 检查html_code是否包含'<table>'和'</table>'
                if "<table>" in html_code and "</table>" in html_code:
                    # 选用<table>到</table>的内容，放入table_res_dict['table_res']['html']
                    start_index = html_code.find("<table>")
                    end_index = html_code.rfind("</table>") + len("</table>")
                    table_res_dict["table_res"]["html"] = html_code[start_index:end_index]
                    
            if use_wired_skeleton and lighton_tables:
                from mineru.model.ocr.lighton_ocr import LightOnOCR
                lighton_ocr = LightOnOCR()
                for table_res_dict in tqdm(lighton_tables, desc="LightOnOCR Table with HTML Skeleton"):
                    bgr_image = cv2.cvtColor(table_res_dict["table_img"], cv2.COLOR_RGB2BGR)
                    _tbl_lang = table_res_dict.get('lang', '')
                    skeleton_html = table_res_dict["table_res"].get("html", "")
                    
                    html_result = lighton_ocr.recognize_table(bgr_image, lang=_tbl_lang, skeleton_html=skeleton_html)
                    table_res_dict["table_res"]["html"] = html_result"""

if old_block2 in content:
    content = content.replace(old_block2, new_block2)
else:
    print("Không tìm thấy old_block2!")

with open("mineru/backend/pipeline/batch_analyze.py", "w") as f:
    f.write(content)
print("Đã vá batch_analyze.py thành công!")
