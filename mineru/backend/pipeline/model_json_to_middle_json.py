# Copyright (c) Opendatalab. All rights reserved.
import copy
import os

from tqdm import tqdm

from mineru.backend.utils.html_image_utils import replace_inline_table_images
from mineru.backend.utils.runtime_utils import cross_page_table_merge
from mineru.utils.config_reader import get_device
from mineru.backend.pipeline.model_init import AtomModelSingleton
from mineru.backend.pipeline.para_split import para_split
from mineru.utils.char_utils import full_to_half
from mineru.utils.cut_image import cut_image_and_table
from mineru.utils.enum_class import ContentType, BlockType
from mineru.utils.title_level_postprocess import apply_title_leveling_to_pdf_info
from mineru.utils.model_utils import clean_memory
from mineru.backend.pipeline.pipeline_magic_model import MagicModel
from mineru.utils.ocr_utils import OcrConfidence, rotate_vertical_crop_if_needed
from mineru.version import __version__
from mineru.utils.hash_utils import bytes_md5
from mineru.utils.pdfium_guard import close_pdfium_document, pdfium_guard


def page_model_info_to_page_info(page_model_info, image_dict, page, image_writer, page_index, ocr_enable=False):
def page_model_info_to_page_info(page_model_info, image_dict, page, image_writer, page_index, ocr_enable=False, formula_enabled=True, lang=None):
    scale = image_dict["scale"]
    page_pil_img = image_dict["img_pil"]
    page_img_md5 = bytes_md5(page_pil_img.tobytes())
    with pdfium_guard():
        page_w, page_h = map(int, page.get_size())
    magic_model = MagicModel(
        page_model_info,
        page,
        scale,
        page_pil_img,
        page_w,
        page_h,
        ocr_enable
    )

    """从magic_model对象中获取后面会用到的区块信息"""
    preproc_blocks = magic_model.get_preproc_blocks()
    discarded_blocks = magic_model.get_discarded_blocks()
    all_image_spans = magic_model.get_all_image_spans()

    # 对image/table/chart/interline_equation的span截图
    for span in all_image_spans:
        if span["type"] in [
            ContentType.IMAGE,
            ContentType.TABLE,
            ContentType.CHART,
            ContentType.INTERLINE_EQUATION
        ]:
            span = cut_image_and_table(span, page_pil_img, page_img_md5, page_index, image_writer, scale=scale)
    img_groups = magic_model.get_imgs()
    table_groups = magic_model.get_tables()

    """对image和table的区块分组"""
    img_body_blocks, img_caption_blocks, img_footnote_blocks, maybe_text_image_blocks = process_groups(
        img_groups, 'image_body', 'image_caption_list', 'image_footnote_list'
    )

    table_body_blocks, table_caption_blocks, table_footnote_blocks, _ = process_groups(
        table_groups, 'table_body', 'table_caption_list', 'table_footnote_list'
    )

    """获取所有的spans信息"""
    spans = magic_model.get_all_spans()

    """某些图可能是文本块，通过简单的规则判断一下"""
    if len(maybe_text_image_blocks) > 0:
        for block in maybe_text_image_blocks:
            should_add_to_text_blocks = False

            if ocr_enable:
                pass  # dùng spans từ OCR đã chạy
            else:
                # Sửa lỗi gọi txt_spans_extract(...) với tham số không hợp lệ
                spans = txt_spans_extract(page, spans, page_pil_img, scale, [], [])
                # 找到与当前block重叠的text spans
                span_in_block_list = [
                    span for span in spans
                    if span['type'] == 'text' and
                       calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], block['bbox']) > 0.7
                ]

                if len(span_in_block_list) > 0:
                    # 计算spans总面积
                    spans_area = sum(
                        (span['bbox'][2] - span['bbox'][0]) * (span['bbox'][3] - span['bbox'][1])
                        for span in span_in_block_list
                    )

                    # 计算block面积
                    block_area = (block['bbox'][2] - block['bbox'][0]) * (block['bbox'][3] - block['bbox'][1])

                    # 判断是否符合文本图条件
                    if block_area > 0 and spans_area / block_area > 0.25:
                        should_add_to_text_blocks = True

            # 根据条件决定添加到哪个列表
            if should_add_to_text_blocks:
                block.pop('group_id', None)  # 移除group_id
                text_blocks.append(block)
            else:
                img_body_blocks.append(block)


    """将所有区块的bbox整理到一起"""
    if formula_enabled:
        interline_equation_blocks = []

    if len(interline_equation_blocks) > 0:

        for block in interline_equation_blocks:
            spans.append({
                "type": ContentType.INTERLINE_EQUATION,
                'score': block['score'],
                "bbox": block['bbox'],
                "content": "",
            })

        all_bboxes, all_discarded_blocks, footnote_blocks = prepare_block_bboxes(
            img_body_blocks, img_caption_blocks, img_footnote_blocks,
            table_body_blocks, table_caption_blocks, table_footnote_blocks,
            discarded_blocks,
            text_blocks,
            title_blocks,
            interline_equation_blocks,
            page_w,
            page_h,
        )
    else:
        all_bboxes, all_discarded_blocks, footnote_blocks = prepare_block_bboxes(
            img_body_blocks, img_caption_blocks, img_footnote_blocks,
            table_body_blocks, table_caption_blocks, table_footnote_blocks,
            discarded_blocks,
            text_blocks,
            title_blocks,
            interline_equations,
            page_w,
            page_h,
        )

    """在删除重复span之前，应该通过image_body和table_body的block过滤一下image和table的span"""
    """顺便删除大水印并保留abandon的span"""
    spans = remove_outside_spans(spans, all_bboxes, all_discarded_blocks)

    """删除重叠spans中置信度较低的那些"""
    spans, dropped_spans_by_confidence = remove_overlaps_low_confidence_spans(spans)
    """删除重叠spans中较小的那些"""
    spans, dropped_spans_by_span_overlap = remove_overlaps_min_spans(spans)

    """根据parse_mode，构造spans，主要是文本类的字符填充"""
    if ocr_enable:
        pass
    else:
        """使用新版本的混合ocr方案."""
        spans = txt_spans_extract(page, spans, page_pil_img, scale, all_bboxes, all_discarded_blocks)

    """先处理不需要排版的discarded_blocks"""
    discarded_block_with_spans, spans = fill_spans_in_blocks(
        all_discarded_blocks, spans, 0.4
    )
    fix_discarded_blocks = fix_discarded_block(discarded_block_with_spans)

    """如果当前页面没有有效的bbox则跳过"""
    if len(all_bboxes) == 0 and len(fix_discarded_blocks) == 0:
        return None

    """对image/table/interline_equation截图"""
    for span in spans:
        if span['type'] in [ContentType.IMAGE, ContentType.TABLE, ContentType.INTERLINE_EQUATION]:
            span = cut_image_and_table(
                span, page_pil_img, page_img_md5, page_index, image_writer, scale=scale
            )

    """span填充进block"""
    block_with_spans, spans = fill_spans_in_blocks(all_bboxes, spans, 0.5)

    """对未分配的orphaned spans进行回收，避免VLM OCR文本丢失"""
    orphaned_blocks = _recover_orphaned_spans(spans, all_bboxes)
    block_with_spans.extend(orphaned_blocks)

    """对block进行fix操作"""
    fix_blocks = fix_block_spans(block_with_spans)

    """对block进行排序"""
    """Sử dụng OCR để điền nội dung cho các block (Title/Text) còn trống"""
    _ocr_fill_empty_blocks(page_model_info, image_dict, fix_blocks, lang, scale)
    sorted_blocks = sort_blocks_by_bbox(fix_blocks, page_w, page_h, footnote_blocks)

    """构造page_info"""
    replace_inline_table_images(preproc_blocks, image_writer, page_index)

    page_info = make_page_info_dict(preproc_blocks, page_index, page_w, page_h, discarded_blocks)

    return page_info


def build_page_model_info(page_layout_dets, page_index, pil_img):
    page_info_dict = {'page_no': page_index, 'width': pil_img.width, 'height': pil_img.height}
    return {'layout_dets': page_layout_dets, 'page_info': page_info_dict}


def append_page_model_infos_to_middle_json(
    middle_json,
    page_model_infos,
    images_list,
    pdf_doc,
    image_writer,
    page_start_index=0,
    ocr_enable=False,
    progress_bar=None,
):
    for offset, (page_model_info, image_dict) in enumerate(zip(page_model_infos, images_list)):
        page_index = page_start_index + offset
        with pdfium_guard():
            page = pdf_doc[page_index]
        page_info = page_model_info_to_page_info(
            copy.deepcopy(page_model_info),
            image_dict,
            page,
            image_writer,
            page_index,
            ocr_enable=ocr_enable,
=======
            page_model_info, image_dict, page, image_writer, page_index,
            ocr_enable=ocr_enable, formula_enabled=formula_enabled, lang=lang
        )
        if page_info is None:
            with pdfium_guard():
                page_w, page_h = map(int, pdf_doc[page_index].get_size())
            page_info = make_page_info_dict([], page_index, page_w, page_h, [])
        middle_json["pdf_info"].append(page_info)
        if progress_bar is not None:
            progress_bar.update(1)


def append_batch_results_to_middle_json(
    middle_json,
    batch_results,
    images_list,
    pdf_doc,
    image_writer,
    page_start_index=0,
    ocr_enable=False,
    model_list=None,
    progress_bar=None,
):
    page_model_infos = []
    for offset, (image_dict, page_layout_dets) in enumerate(zip(images_list, batch_results)):
        page_index = page_start_index + offset
        page_model_info = build_page_model_info(page_layout_dets, page_index, image_dict['img_pil'])
        page_model_infos.append(page_model_info)

    if model_list is not None:
        model_list.extend(page_model_infos)

    append_page_model_infos_to_middle_json(
        middle_json,
        page_model_infos,
        images_list,
        pdf_doc,
        image_writer,
        page_start_index=page_start_index,
        ocr_enable=ocr_enable,
        progress_bar=progress_bar,
    )


def _extract_text_from_block(block):
    text_parts = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            if span.get("type") == ContentType.TEXT:
                text_parts.append(span.get("content", ""))
    return "".join(text_parts).strip()


def _iter_block_spans(block):
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            yield span

    for sub_block in block.get("blocks", []):
        yield from _iter_block_spans(sub_block)


def _normalize_formula_tag_content(tag_content):
    tag_content = full_to_half(tag_content.strip())
    if tag_content.startswith("("):
        tag_content = tag_content[1:].strip()
    if tag_content.endswith(")"):
        tag_content = tag_content[:-1].strip()
    return tag_content


def _get_interline_equation_span(block):
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            if span.get("type") == ContentType.INTERLINE_EQUATION:
                return span
    return None


def _append_formula_number_tag(equation_block, formula_number_block):
    equation_span = _get_interline_equation_span(equation_block)
    tag_content = _normalize_formula_tag_content(_extract_text_from_block(formula_number_block))
    if equation_span is not None:
        formula = equation_span.get("content", "")
        equation_span["content"] = f"{formula}\\tag{{{tag_content}}}"


def _optimize_formula_number_blocks(pdf_info_list):
    for page_info in pdf_info_list:
        optimized_blocks = []
        blocks = page_info.get("preproc_blocks", [])
        for index, block in enumerate(blocks):
            if block.get("type") != BlockType.FORMULA_NUMBER:
                optimized_blocks.append(block)
                continue

            prev_block = blocks[index - 1] if index > 0 else None
            if prev_block and prev_block.get("type") == BlockType.INTERLINE_EQUATION:
                _append_formula_number_tag(prev_block, block)
                continue

            next_block = blocks[index + 1] if index + 1 < len(blocks) else None
            next_next_block = blocks[index + 2] if index + 2 < len(blocks) else None
            if (
                next_block
                and next_block.get("type") == BlockType.INTERLINE_EQUATION
                and (next_next_block is None or next_next_block.get("type") != BlockType.FORMULA_NUMBER)
            ):
                _append_formula_number_tag(next_block, block)
                continue

            block["type"] = BlockType.TEXT
            optimized_blocks.append(block)

        page_info["preproc_blocks"] = optimized_blocks


def _apply_post_ocr(pdf_info_list, lang=None):
    need_ocr_list = []
    img_crop_list = []

    for page_info in pdf_info_list:
        for block in page_info.get('preproc_blocks', []):
            for span in _iter_block_spans(block):
                if 'np_img' in span:
                    need_ocr_list.append(span)
                    # Keep post-OCR rec aligned with the main OCR pipeline for vertical tall crops.
                    img_crop_list.append(rotate_vertical_crop_if_needed(span['np_img']))
                    span.pop('np_img')
=======
    text_block_list = []
    for page_info in middle_json["pdf_info"]:
        for block in page_info['preproc_blocks']:
            if block['type'] in ['table', 'image']:
                for sub_block in block['blocks']:
                    if sub_block['type'] in ['image_caption', 'image_footnote', 'table_caption', 'table_footnote']:
                        text_block_list.append(sub_block)
            elif block['type'] in ['text', 'title']:
                text_block_list.append(block)
        for block in page_info['discarded_blocks']:
            text_block_list.append(block)
    for block in text_block_list:  # chỉ lấy 'text' và 'title'
        for line in block['lines']:
            for span in line['spans']:
                if 'np_img' in span:  # ← title spans không bao giờ có np_img
                    need_ocr_list.append(span)
    if len(img_crop_list) > 0:
        atom_model_manager = AtomModelSingleton()
        ocr_model = atom_model_manager.get_atom_model(
            atom_model_name='ocr',
            det_db_box_thresh=0.3,
            lang=lang
        )
        ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]
        assert len(ocr_res_list) == len(
            need_ocr_list), f'ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(need_ocr_list)}'
        for index, span in enumerate(need_ocr_list):
            ocr_text, ocr_score = ocr_res_list[index]
            if ocr_score > OcrConfidence.min_confidence:
                span['content'] = ocr_text
                span['score'] = float(f"{ocr_score:.3f}")
            else:
                span['content'] = ''
                span['score'] = 0.0

        for block in page_info.get('discarded_blocks', []):
            for span in _iter_block_spans(block):
                if 'np_img' in span:
                    need_ocr_list.append(span)
                    # Keep post-OCR rec aligned with the main OCR pipeline for vertical tall crops.
                    img_crop_list.append(rotate_vertical_crop_if_needed(span['np_img']))
                    span.pop('np_img')

    if len(img_crop_list) == 0:
        return

    atom_model_manager = AtomModelSingleton()
    ocr_model = atom_model_manager.get_atom_model(
        atom_model_name='ocr',
        det_db_box_thresh=0.3,
        lang=lang
    )
    ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]
    assert len(ocr_res_list) == len(
        need_ocr_list), f'ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(need_ocr_list)}'
    for index, span in enumerate(need_ocr_list):
        ocr_text, ocr_score = ocr_res_list[index]
        if ocr_score > OcrConfidence.min_confidence:
            span['content'] = ocr_text
            span['score'] = float(f"{ocr_score:.3f}")
        else:
            span['content'] = ''
            span['score'] = 0.0


def _post_block_process(pdf_info_list):
    for page_info in pdf_info_list:
        for block_key in ["preproc_blocks", "para_blocks"]:
            for block in page_info.get(block_key, []):
                block_type = block.get("type")
                if block_type == BlockType.DOC_TITLE:
                    block["type"] = BlockType.TITLE
                    block["level"] = 1
                elif block_type == BlockType.PARAGRAPH_TITLE:
                    block["type"] = BlockType.TITLE
                    block["level"] = 2
                elif block_type == BlockType.VERTICAL_TEXT:
                    block["type"] = BlockType.TEXT


def apply_server_side_postprocess(pdf_info_list, lang=None):
    """执行只能在服务端完成的后处理；目前仅包含依赖 OCR 模型的 post-OCR。"""
    _apply_post_ocr(pdf_info_list, lang=lang)


def finalize_middle_json_from_preproc(pdf_info_list):
    """从 preproc_blocks 执行确定性 finalize，供服务端完整路径和客户端复用。"""
    _optimize_formula_number_blocks(pdf_info_list)
    para_split(pdf_info_list)
    cross_page_table_merge(pdf_info_list)
    apply_title_leveling_to_pdf_info(pdf_info_list)
    _post_block_process(pdf_info_list)


def finalize_middle_json(
    pdf_info_list,
    lang=None,
):
    """Apply document-level post processing once all page_info entries are ready."""
    apply_server_side_postprocess(pdf_info_list, lang=lang)
    finalize_middle_json_from_preproc(pdf_info_list)
    if os.getenv('MINERU_DONOT_CLEAN_MEM') is None and len(pdf_info_list) >= 10:
        clean_memory(get_device())


def init_middle_json():
    return {"pdf_info": [], "_backend": "pipeline", "_version_name": __version__}


def result_to_middle_json(model_list, images_list, pdf_doc, image_writer, lang=None, ocr_enable=False, formula_enable=None):
    middle_json = init_middle_json()
    with tqdm(total=len(model_list), desc="Processing pages") as progress_bar:
        append_page_model_infos_to_middle_json(
            middle_json,
            model_list,
            images_list,
            pdf_doc,
            image_writer,
            ocr_enable=ocr_enable,
            progress_bar=progress_bar,
        )

    finalize_middle_json(middle_json["pdf_info"], lang=lang)
    close_pdfium_document(pdf_doc)
    return middle_json


def make_page_info_dict(blocks, page_id, page_w, page_h, discarded_blocks):
    return_dict = {
        'preproc_blocks': blocks,
        'page_idx': page_id,
        'page_size': [page_w, page_h],
        'discarded_blocks': discarded_blocks,
    }
    return return_dict

def _ocr_fill_empty_blocks(page_model_info, image_dict, fix_blocks, lang, scale):
    """
    Hậu xử lý: Với mỗi block Title hoặc Text không có span nội dung nào,
    sẽ crop ảnh từ page image và OCR trực tiếp để lấy content.
    
    Điều này giúp khắc phục lỗi: Block được Layout nhận diện tốt nhưng OCR span 
    bị vứt bỏ do overlap filter hoặc tọa độ lệch.
    """
    import cv2
    import numpy as np
    from mineru.backend.pipeline.model_init import AtomModelSingleton
    from mineru.utils.ocr_utils import OcrConfidence
    from mineru.utils.enum_class import BlockType

    # Lấy page image numpy
    page_pil = image_dict.get("img_pil")
    if page_pil is None:
        return
    page_np = np.array(page_pil)  # RGB

    atom_manager = AtomModelSingleton()
    try:
        ocr_model = atom_manager.get_atom_model(
            atom_model_name="ocr",
            det_db_box_thresh=0.3,
            lang=lang,
        )
    except Exception as e:
        from loguru import logger
        logger.warning(f"[ocr_fill] Không load được OCR model: {e}")
        return

    for block in fix_blocks:
        # Chỉ xử lý các loại block văn bản/tiêu đề
        if block.get("type") not in [BlockType.TITLE, BlockType.TEXT]:
            continue

        # Kiểm tra xem block đã có nội dung chưa
        has_content = any(
            span.get("content", "").strip()
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        )
        if has_content:
            continue

        # Lấy bbox (PDF points) và convert về pixel để crop
        bbox = block.get("bbox")
        if not bbox:
            continue

        x0 = int(bbox[0] * scale)
        y0 = int(bbox[1] * scale)
        x1 = int(bbox[2] * scale)
        y1 = int(bbox[3] * scale)

        # Thêm padding nhỏ để OCR nhận diện tốt hơn (tránh cắt sát quá)
        pad = 8
        h, w = page_np.shape[:2]
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(w, x1 + pad)
        y1 = min(h, y1 + pad)

        if x1 <= x0 or y1 <= y0:
            continue

        crop = page_np[y0:y1, x0:x1]
        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        try:
            # Kiểm tra xem ocr_model có text_detector không (ví dụ PaddleOCR)
            has_text_det = hasattr(ocr_model, "text_detector") or getattr(ocr_model, "has_text_detector", False)
            if has_text_det:
                # Paddle: thực hiện cả det+rec trong vùng crop
                ocr_result = ocr_model.ocr(bgr)[0]
                if ocr_result:
                    texts = [item[1][0] for item in ocr_result if len(item) >= 2 and item[1][0].strip()]
                    text = " ".join(texts)
                else:
                    text = ""
            else:
                # Engine khác (EasyOCR / LightOnOCR): OCR toàn bộ vùng crop
                ocr_result = ocr_model.ocr(bgr, det=True, rec=True)[0]
                if ocr_result:
                    texts = [item[1][0] for item in ocr_result if len(item) >= 2 and item[1][0].strip()]
                    text = " ".join(texts)
                else:
                    text = ""
        except Exception as e:
            from loguru import logger
            logger.warning(f"[ocr_fill] OCR lỗi tại bbox={bbox}: {e}")
            text = ""

        if text.strip():
            from loguru import logger
            logger.debug(f"[ocr_fill] Filled {block.get('type')} content: {repr(text[:60])}")
            # Tạo span mới và gán vào line đầu tiên của block
            new_span = {
                "bbox": bbox,
                "type": "text",
                "content": text.strip(),
                "score": 0.9,
            }
            if block.get("lines"):
                block["lines"][0].setdefault("spans", []).append(new_span)
            else:
                block["lines"] = [{
                    "bbox": bbox,
                    "spans": [new_span],
                }]

def _recover_orphaned_spans(spans, all_bboxes):
    """
    回收未分配给任何block的spans，将其转换为独立的text blocks。
    主要用于VLM生成的OcrText span (category 15)，因为layout模型可能漏检这些区域。
    """
    from mineru.utils.enum_class import BlockType, ContentType
    from mineru.utils.boxbase import calculate_overlap_area_in_bbox1_area_ratio
    orphaned_blocks = []
    
    # 过滤出有内容的文本类span
    valid_spans = [
        span for span in spans 
        if span.get('type') in [ContentType.TEXT, ContentType.INLINE_EQUATION]
        and span.get('content', '').strip()
    ]
    
    for span in valid_spans:
        span_bbox = span['bbox']
        
        # 检查是否与现有任何 block 有显著重叠，如果有则认为是已经被覆盖的（或者是 layout 模型认为该丢弃的）
        is_covered = False
        for block_bbox_full in all_bboxes:
            block_bbox = block_bbox_full[:4]
            if calculate_overlap_area_in_bbox1_area_ratio(span_bbox, block_bbox) > 0.1:
                is_covered = True
                break
        
        if not is_covered:
            # 构造符合 fix_block_spans 输入格式 of block dictionary
            block = {
                'type': BlockType.TEXT,
                'bbox': span_bbox,
                'spans': [span],
            }
            orphaned_blocks.append(block)
    
    return orphaned_blocks
