import os
from pathlib import Path
from mineru.cli.common import read_fn, do_parse
from mineru.utils.enum_class import MakeMode
from loguru import logger

def process_v_mineru(input_path: str, output_dir: str = "./output"):
    """
    Hàm xử lý V-MinerU duy nhất: 
    Tự động nhận diện PDF chữ/ảnh, ngôn ngữ, và chuyển đổi sang Markdown.
    """
    try:
        # 1. Đọc file (hỗ trợ cả PDF và các định dạng ảnh)
        input_p = Path(input_path)
        if not input_p.exists():
            logger.error(f"File không tồn tại: {input_path}")
            return

        logger.info(f"Đang bắt đầu xử lý: {input_p.name}")
        pdf_bytes = read_fn(input_p)
        
        # 2. Gọi hàm parse duy nhất (Sử dụng logic V-MinerU đã tối ưu)
        # Mặc định: backend='pipeline', parse_method='auto', lang='auto'
        do_parse(
            output_dir=output_dir,
            pdf_file_names=[input_p.stem],
            pdf_bytes_list=[pdf_bytes],
            p_lang_list=["auto"],
            backend="pipeline",
            parse_method="auto", # Tự động nhận diện PDF chữ hay ảnh để OCR
            f_make_md_mode=MakeMode.MM_MD
        )
        
        logger.info(f"Xử lý thành công! Kết quả lưu tại: {output_dir}/{input_p.stem}")
        
    except Exception as e:
        logger.exception(f"Lỗi trong quá trình xử lý: {e}")

if __name__ == "__main__":
    # Thay đổi đường dẫn file của bạn tại đây
    FILE_TO_PROCESS = "demo3.pdf" 
    
    process_v_mineru(FILE_TO_PROCESS)
