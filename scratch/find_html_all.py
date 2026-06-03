import json

def find_html(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ["html", "text"]:
                if isinstance(v, str) and "<table" in v:
                    print(f"FOUND in path: {path}['{k}']")
                    print(f"Content starts with: {v[:50]}")
            find_html(v, f"{path}['{k}']")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find_html(v, f"{path}[{i}]")

with open("output_test_split_pdf/CTDT_Khoa_2024_19Sep2024_CNTN-pages-2/auto/CTDT_Khoa_2024_19Sep2024_CNTN-pages-2_middle.json", "r") as f:
    data = json.load(f)

print("Searching entire middle.json...")
find_html(data, "root")
