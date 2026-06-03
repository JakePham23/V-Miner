# Copyright (c) Opendatalab. All rights reserved.
import os
import time
from loguru import logger
from tqdm import tqdm

from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.pdf_image_tools import load_images_from_pdf
from mineru.utils.enum_class import ImageType
from mineru.version import __version__
from mineru.model.ocr.lighton_ocr import LightOnOCR

def process_vlm_api(
        output_dir,
        pdf_file_names,
        pdf_bytes_list,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_md,
        f_dump_middle_json,
        f_dump_model_output,
        f_dump_orig_pdf,
        f_dump_content_list,
        f_make_md_mode,
        server_url=None,
        **kwargs,
):
    """Xử lý OCR toàn trang bằng cách gửi ảnh trực tiếp qua VLM API (không load model local)."""
    parse_method = "vlm_api"
    # Tắt vẽ bbox vì chúng ta không chạy Layout model local nào cả
    f_draw_layout_bbox = False
    f_draw_span_bbox = False
    f_dump_model_output = False # Không chạy model local nên không có model output

    # Import hàm ghi file dùng chung từ common
    from mineru.cli.common import prepare_env, _process_output

    # Khởi tạo API Client
    logger.info("[VLM-API] Khởi động OCR toàn trang qua API...")
    lighton = LightOnOCR(server_url=server_url)

    for idx, pdf_bytes in enumerate(pdf_bytes_list):
        pdf_file_name = pdf_file_names[idx]
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
        image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

        # Chuyển đổi PDF thành ảnh PIL (chạy bằng CPU cực nhẹ)
        images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)

        middle_json = {
            "pdf_info": [],
            "_backend": "vlm-api",
            "_version_name": __version__
        }

        # Gọi VLM API cho từng trang
        for page_idx, img_dict in enumerate(tqdm(images_list, desc="VLM-API OCR Progress")):
            img_pil = img_dict["img_pil"]
            
            try:
                # Gửi ảnh trang giấy lên API LLM
                raw_md = lighton.recognize_page(img_pil)
            except Exception as e:
                logger.error(f"[VLM-API] Lỗi khi xử lý trang {page_idx + 1}: {e}")
                raw_md = f"\n\n[ERROR: Failed to OCR page {page_idx + 1} via API]\n\n"

            page_info = {
                "page_idx": page_idx,
                "raw_markdown": raw_md,
                "page_size": list(img_pil.size)
            }
            middle_json["pdf_info"].append(page_info)

        pdf_info = middle_json["pdf_info"]

        # Ghi kết quả ra file markdown
        _process_output(
            pdf_info, pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
            md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_pdf,
            f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
            f_make_md_mode, middle_json, model_output=None, is_pipeline=False
        )

        pdf_doc.close()
