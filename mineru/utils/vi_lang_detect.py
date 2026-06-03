import pypdfium2 as pdfium

def detect_vietnamese(pdf_bytes: bytes) -> bool:
    """
    Tự động nhận diện tiếng Việt bằng cách trích xuất text 5 trang đầu
    và kiểm tra các ký tự có dấu đặc trưng của tiếng Việt.
    """
    try:
        pdf = pdfium.PdfDocument(pdf_bytes)
        sample_text = ""
        for i in range(min(len(pdf), 5)):
            page = pdf[i]
            sample_text += page.get_textpage().get_text_bounded()
        pdf.close()
        
        vietnamese_markers = "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ"
        return any(char in sample_text for char in vietnamese_markers)
    except Exception:
        return False
