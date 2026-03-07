import re

with open("mineru/utils/table_merge.py", "r") as f:
    content = f.read()

safe_int_func = """
import re
def safe_int(val, default=1):
    try:
        if isinstance(val, str):
            m = re.search(r'\\d+', val)
            if m: return int(m.group())
        return int(val)
    except:
        return default
"""

if "def safe_int" not in content:
    # Chèn hàm này ngay sau các dòng import đầu tiên (sau dòng `from bs4 import BeautifulSoup` hoặc tương tự)
    # Nhưng đơn giản nhất là chèn vào đầu file, ngay sau imports
    import_lines = []
    other_lines = []
    lines = content.split('\n')
    for line in lines:
        if line.startswith('import ') or line.startswith('from '):
            import_lines.append(line)
        else:
            other_lines.append(line)
            
    # Chèn an toàn
    new_content = '\n'.join(import_lines) + '\n\n' + safe_int_func + '\n' + '\n'.join(other_lines)
    content = new_content

# Thay thế bằng RegEx cho tất cả các chỗ lấy rowspan/colspan
content = re.sub(
    r'int\(\s*([a-zA-Z0-9_]+)\.get\(\s*("colspan"|\'colspan\'|"rowspan"|\'rowspan\')\s*,\s*1\s*\)\s*\)', 
    r'safe_int(\1.get(\2, 1))', 
    content
)

with open("mineru/utils/table_merge.py", "w") as f:
    f.write(content)
print("Đã vá table_merge.py thành công!")
