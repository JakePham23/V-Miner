import json

def find_html(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "html" or k == "text":
                if isinstance(v, str) and "<table" in v:
                    print(f"FOUND in path: {path}['{k}']")
                    print(f"Content starts with: {v[:50]}")
            find_html(v, f"{path}['{k}']")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find_html(v, f"{path}[{i}]")

with open("output_test_split_pdf/CTDT_Khoa_2024_19Sep2024_CNTN-pages-2/auto/CTDT_Khoa_2024_19Sep2024_CNTN-pages-2_middle.json", "r") as f:
    data = json.load(f)

table3 = [b for b in data["pdf_info"][1]["para_blocks"] if b["type"] == "table"][0]
print("Searching in Table 3...")
find_html(table3, "table3")
