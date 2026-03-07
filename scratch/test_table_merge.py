import json
import os
import sys

# Add V-Miner to path
sys.path.append(os.path.abspath('.'))

from mineru.utils.table_merge import perform_table_merge, can_merge_tables

# Load JSON
with open('output/CTDT_Khoa_2024_19Sep2024_CNTN/auto/CTDT_Khoa_2024_19Sep2024_CNTN_middle.json', 'r') as f:
    data = json.load(f)

table_blocks_p2 = [b for b in data['pdf_info'][1]['para_blocks'] if b['type']=='table']
table_blocks_p3 = [b for b in data['pdf_info'][2]['para_blocks'] if b['type']=='table']

previous_table_block = table_blocks_p2[-1]
current_table_block = table_blocks_p3[0]

wait_merge_table_footnotes = [
    block for block in current_table_block["blocks"]
    if block["type"] == 'table_footnote'
]

can_merge, soup1, soup2, current_html, previous_html = can_merge_tables(
    current_table_block, previous_table_block
)

print("Can merge:", can_merge)

if can_merge:
    print("Previous HTML length:", len(previous_html))
    print("Current HTML length:", len(current_html))
    
    merged_html = perform_table_merge(
        soup1, soup2, previous_table_block, wait_merge_table_footnotes
    )
    print("Merged HTML length:", len(merged_html))
    print("Merged HTML preview:", merged_html[:200])
