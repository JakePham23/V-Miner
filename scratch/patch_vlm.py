import re

with open("mineru/backend/vlm/vlm_middle_json_mkcontent.py", "r") as f:
    content = f.read()

# Thêm import ở đầu file nếu chưa có
if "from mineru.utils.format_utils import convert_otsl_to_html" not in content:
    content = "from mineru.utils.format_utils import convert_otsl_to_html\n" + content

# Thay thế chỗ gán html: html thành html đã được parse
old_code = "'html': html,"
new_code = "'html': convert_otsl_to_html(html) if ('<fcel>' in html or '<ecel>' in html) else html,"

if old_code in content:
    content = content.replace(old_code, new_code)
    
    with open("mineru/backend/vlm/vlm_middle_json_mkcontent.py", "w") as f:
        f.write(content)
    print("Vá lỗi vlm_middle_json_mkcontent.py thành công!")
else:
    print("Không tìm thấy đoạn code cần vá!")
