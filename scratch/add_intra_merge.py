import re

with open("mineru/utils/table_merge.py", "r") as f:
    content = f.read()

# Ta sẽ tạo một phiên bản `merge_table_intra_page` và gọi nó bên trong `merge_table`
intra_page_logic = """
def merge_table_intra_page(page_info_list):
    '''Gộp các bảng bị chẻ đôi trên cùng một trang (Intra-page)'''
    for page_idx in range(len(page_info_list)):
        page_info = page_info_list[page_idx]
        blocks = page_info.get("para_blocks", [])
        
        # Duyệt ngược để an toàn khi xóa/merge
        i = len(blocks) - 1
        while i > 0:
            current_block = blocks[i]
            prev_block = blocks[i-1]
            
            # Nếu cả 2 đều là bảng, xem có thể gộp không
            if current_block["type"] == BlockType.TABLE and prev_block["type"] == BlockType.TABLE:
                # Bỏ qua nếu có text/title chen giữa nằm ngoài lề?
                # Trong list para_blocks, nếu chúng liền kề thì có nghĩa là không có text xen giữa
                
                wait_merge_table_footnotes = [
                    block for block in current_block.get("blocks", [])
                    if block["type"] == BlockType.TABLE_FOOTNOTE
                ]
                
                can_merge, soup1, soup2, current_html, previous_html = can_merge_tables(
                    current_block, prev_block
                )
                
                if can_merge:
                    logger.info(f"Đã gộp thành công 2 bảng trên CÙNG TRANG {page_idx + 1}")
                    merged_html = perform_table_merge(
                        soup1, soup2, prev_block, wait_merge_table_footnotes
                    )
                    
                    for block in prev_block.get("blocks", []):
                        if (block["type"] == BlockType.TABLE_BODY and block.get("lines") and block["lines"][0].get("spans")):
                            block["lines"][0]["spans"][0]["html"] = merged_html
                            break
                    
                    for block in current_block.get("blocks", []):
                        block['lines'] = []
                        block[SplitFlag.LINES_DELETED] = True
            i -= 1
"""

if "merge_table_intra_page" not in content:
    # Chèn hàm này vào trước merge_table
    content = content.replace("def merge_table(page_info_list):", intra_page_logic + "\n\ndef merge_table(page_info_list):")
    # Gọi hàm này ở đầu merge_table
    content = content.replace('"""合并跨页表格"""', '"""合并跨页表格"""\n    merge_table_intra_page(page_info_list)\n')
    
    with open("mineru/utils/table_merge.py", "w") as f:
        f.write(content)
    print("Added intra-page merge logic!")
else:
    print("Already added.")
