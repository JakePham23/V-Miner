import re

with open("mineru/model/ocr/lighton_ocr.py", "r") as f:
    content = f.read()

old_func_def = """    def recognize_table(
        self,
        image: Union[np.ndarray, Image.Image],
        bbox_coords=None,
        *,
        page_img: Optional[np.ndarray] = None,
        poly: Optional[list] = None,
        vietnamese: bool = False,
    ) -> str:"""

new_func_def = """    def recognize_table(
        self,
        image: Union[np.ndarray, Image.Image],
        bbox_coords=None,
        *,
        page_img: Optional[np.ndarray] = None,
        poly: Optional[list] = None,
        vietnamese: bool = False,
        lang: str = "",
        skeleton_html: str = "",
    ) -> str:"""

if old_func_def in content:
    content = content.replace(old_func_def, new_func_def)
else:
    print("Không tìm thấy old_func_def!")


old_prompt = """        if vietnamese:
            prompt = (
                "Đây là bảng tiếng Việt (vietnamese). Trích xuất toàn bộ nội dung bảng. Header của bảng có bao nhiêu cột thì hàng dưới cũng phải có bấy nhiêu cột, không được thiếu. Chỉ có thể ít hơn cột header (nếu có tồn tại) "
                "Đảm bảo chính xác dấu thanh tiếng Việt (á, à, ả, ã, ạ, ắ, ặ, ẳ, ẵ, ề, ế, ệ, ể, ễ, ổ, ỗ, ộ, v.v.). "
                "Output ONLY a valid HTML <table> with <thead> and <tbody>. "
                "Do not add any explanation or preamble."
            )
        else:
            prompt = (
                "Extract the table from this image. "
                "Output ONLY a valid HTML <table> with <thead> and <tbody>. "
                "Do not add any explanation or preamble. "
                "Ensure Vietnamese text is accurate."
            )"""

new_prompt = """        if lang and lang.startswith('vi'):
            vietnamese = True
            
        if skeleton_html and "<table>" in skeleton_html:
            prompt = (
                "Dưới đây là một ảnh chứa bảng và cấu trúc HTML khung của bảng đó đã được dựng sẵn (skeleton). "
                "Hãy nhìn vào ảnh, đọc các chữ tiếng Việt và ĐIỀN ĐÚNG các chữ đó vào các ô <td> tương ứng trong khung HTML này. "
                "Giữ nguyên cấu trúc thẻ <tr>, <td>, rowspan, colspan của khung HTML (trừ khi sai khác quá lớn so với ảnh). "
                "Đảm bảo chính xác dấu thanh tiếng Việt (á, à, ả, ã, ạ, ắ, ặ, ẳ, ẵ, ề, ế, ệ, ể, ễ, ổ, ỗ, ộ, v.v.). "
                "Output ONLY a valid HTML <table>...</table>, KHÔNG kèm giải thích.\n"
                f"SKELETON HTML:\n{skeleton_html}"
            )
        elif vietnamese:
            prompt = (
                "Đây là bảng tiếng Việt (vietnamese). Trích xuất toàn bộ nội dung bảng. Header của bảng có bao nhiêu cột thì hàng dưới cũng phải có bấy nhiêu cột, không được thiếu. Chỉ có thể ít hơn cột header (nếu có tồn tại) "
                "Đảm bảo chính xác dấu thanh tiếng Việt (á, à, ả, ã, ạ, ắ, ặ, ẳ, ẵ, ề, ế, ệ, ể, ễ, ổ, ỗ, ộ, v.v.). "
                "Output ONLY a valid HTML <table> with <thead> and <tbody>. "
                "Do not add any explanation or preamble."
            )
        else:
            prompt = (
                "Extract the table from this image. "
                "Output ONLY a valid HTML <table> with <thead> and <tbody>. "
                "Do not add any explanation or preamble. "
                "Ensure Vietnamese text is accurate."
            )"""

if old_prompt in content:
    content = content.replace(old_prompt, new_prompt)
else:
    print("Không tìm thấy old_prompt!")

with open("mineru/model/ocr/lighton_ocr.py", "w") as f:
    f.write(content)
print("Đã vá lighton_ocr.py thành công!")
