import sys
import os
sys.path.append(os.getcwd())

from mineru.utils.format_utils import block_content_to_html

input_file = "output_test_split_pdf_mineru2.5-pro-2605-1.2b/CTDT_Khoa_2024_19Sep2024_CNTN-pages-2/auto/CTDT_Khoa_2024_19Sep2024_CNTN-pages-2.md"
output_file = "output_test_split_pdf_mineru2.5-pro-2605-1.2b/CTDT_Khoa_2024_19Sep2024_CNTN-pages-2/auto/CTDT_Khoa_2024_19Sep2024_CNTN-pages-2_fixed.md"

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

# Hàm này sẽ quét và tự động chuyển đổi các dòng chứa <fcel> sang <table>
fixed_content = block_content_to_html(content)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(fixed_content)

print(f"Đã chuyển đổi xong! Lưu tại: {output_file}")
