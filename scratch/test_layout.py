import json
with open("output_test_split_pdf/CTDT_Khoa_2024_19Sep2024_CNTN-pages-2/auto/CTDT_Khoa_2024_19Sep2024_CNTN-pages-2_middle.json", "r") as f:
    data = json.load(f)

for page_idx in range(2):
    page = data.get("pdf_info", [])[page_idx]
    blocks = page.get("para_blocks", [])
    print(f"\n--- PAGE {page_idx+1} ---")
    for b in blocks:
        if b["type"] == "table":
            bbox = b.get("bbox")
            subs = len(b.get("blocks", []))
            print(f"[TABLE] bbox={bbox} spans={subs} sub-blocks")
            for sub in b.get("blocks", []):
                if sub["type"] == "table_caption":
                    text = "".join(s.get("text", "") for l in sub.get("lines", []) for s in l.get("spans", []))
                    print(f"   -> Caption: {text}")
        elif b["type"] in ["text", "title"]:
            t_type = b["type"].upper()
            text = "".join(s.get("text", "") for l in b.get("lines", []) for s in l.get("spans", []))
            bbox = b.get("bbox")
            print(f"[{t_type}] bbox={bbox}: {text[:50]}")
