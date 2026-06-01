import json
import os
from loguru import logger

# Bật log debug
os.environ["MINERU_TABLE_MERGE_ENABLE"] = "true"

from mineru.utils.table_merge import can_merge_tables, merge_table

with open("output_test_split_pdf/CTDT_Khoa_2024_19Sep2024_CNTN-pages-2/auto/CTDT_Khoa_2024_19Sep2024_CNTN-pages-2_middle.json", "r") as f:
    data = json.load(f)

page_info_list = data.get("pdf_info", [])
print("\n--- TEST CAN MERGE TABLE 3 (Page 2) & TABLE 2 (Page 1) ---")

# Table 2 là bảng cuối của Page 1
table2 = [b for b in page_info_list[0]["para_blocks"] if b["type"] == "table"][-1]
# Table 3 là bảng đầu của Page 2
table3 = [b for b in page_info_list[1]["para_blocks"] if b["type"] == "table"][0]

can_merge, soup1, soup2, current_html, previous_html = can_merge_tables(table3, table2)
print(f"Can merge? {can_merge}")

print("\n--- TEST FULL MERGE_TABLE ---")
merge_table(page_info_list)
print("Merge table loop finished.")
