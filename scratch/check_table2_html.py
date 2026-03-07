import json

with open("output_test_split_pdf/CTDT_Khoa_2024_19Sep2024_CNTN-pages-2/auto/CTDT_Khoa_2024_19Sep2024_CNTN-pages-2_middle.json", "r") as f:
    data = json.load(f)

table2 = data['pdf_info'][0]['para_blocks'][6]['blocks'][0]['lines'][0]['spans'][0]['html']

# Kiểm tra xem Table 2 (ở Page 1) có chứa dòng ELO 3. 1. 1 của Table 3 không nhé!
if "ELO 3. 1. 1" in table2:
    print("BÙM! Table 2 ĐÃ CHỨA mã HTML của Table 3!")
else:
    print("KHÔNG. Table 2 không chứa Table 3.")
    
print("\nCuối Table 2 HTML trông như thế này:")
print(table2[-500:])

