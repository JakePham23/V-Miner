import json

with open("output_test_split_pdf/CTDT_Khoa_2024_19Sep2024_CNTN-pages-2/auto/CTDT_Khoa_2024_19Sep2024_CNTN-pages-2_middle.json", "r") as f:
    data = json.load(f)

# Lấy mã HTML của Table 3 từ PREPROC_BLOCKS của Page 2
table3_preproc = data['pdf_info'][1]['preproc_blocks'][0]['blocks'][0]['lines'][0]['spans'][0]['html']

print(f"Độ dài HTML của Table 3 trong preproc_blocks: {len(table3_preproc)}")
print("\n--- 200 ký tự đầu ---")
print(table3_preproc[:200])
print("\n--- 200 ký tự cuối ---")
print(table3_preproc[-200:])

