import json

with open("output_test_split_pdf/CTDT_Khoa_2024_19Sep2024_CNTN-pages-2/auto/CTDT_Khoa_2024_19Sep2024_CNTN-pages-2_middle.json", "r") as f:
    data = json.load(f)

for page_idx, page in enumerate(data.get("pdf_info", [])):
    blocks = page.get("para_blocks", [])
    tables = [b for b in blocks if b["type"] == "table"]
    for t_idx, t in enumerate(tables):
        html_content = ""
        for sub in t.get("blocks", []):
            if sub["type"] == "table_body" and sub.get("lines"):
                spans = sub["lines"][0].get("spans", [])
                if spans:
                    html_content = spans[0].get("html", "")
        
        print(f"Page {page_idx+1} | Table {t_idx+1}")
        print(f"  bbox = {t.get('bbox')}")
        print(f"  HTML Length = {len(html_content)}")
        if html_content:
            print(f"  Starts with: {html_content[:80]}")
            
