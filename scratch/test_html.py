import json
with open("output_test_split_pdf/CTDT_Khoa_2024_19Sep2024_CNTN-pages-2/auto/CTDT_Khoa_2024_19Sep2024_CNTN-pages-2_middle.json", "r") as f:
    data = json.load(f)
for page_idx in range(2):
    page = data.get("pdf_info", [])[page_idx]
    blocks = page.get("para_blocks", [])
    for b in blocks:
        if b["type"] == "table":
            html_content = ""
            for sub in b.get("blocks", []):
                if sub["type"] == "table_body" and sub.get("lines"):
                    spans = sub["lines"][0].get("spans", [])
                    if spans:
                        html_content = spans[0].get("html", "")
            bbox = b.get("bbox")
            print(f"PAGE {page_idx+1} TABLE bbox={bbox} HTML Length = {len(html_content)}")
            if len(html_content) > 0:
                print(f"  -> Starts with: {html_content[:50]}")
