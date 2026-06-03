import re

with open('mineru/utils/ocr_utils.py', 'r') as f:
    content = f.read()

# Add a print statement before the trim logic
pattern = r"            # Trim the paste_x and paste_y padding if the box represents the entire crop region"
replacement = r"""            # PRINT FOR DEBUGGING
            print(f"DEBUG OCR BOX: p1={p1}, p3={p3}, new_width={new_width}, new_height={new_height}")
            # Trim the paste_x and paste_y padding if the box represents the entire crop region"""

content = content.replace(pattern, replacement)

with open('mineru/utils/ocr_utils.py', 'w') as f:
    f.write(content)
