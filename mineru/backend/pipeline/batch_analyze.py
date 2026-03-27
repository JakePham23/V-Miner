import html

import cv2
from loguru import logger
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import os 
from .model_init import AtomModelSingleton
from .model_list import AtomicModel
from ...utils.config_reader import get_formula_enable, get_table_enable
from ...utils.model_utils import crop_img, get_res_list_from_layout_res, clean_vram
from ...utils.ocr_utils import merge_det_boxes, update_det_boxes, sorted_boxes
from ...utils.ocr_utils import get_adjusted_mfdetrec_res, get_ocr_result_list, OcrConfidence, get_rotate_crop_image
from ...utils.pdf_image_tools import get_crop_np_img

YOLO_LAYOUT_BASE_BATCH_SIZE = 1
MFD_BASE_BATCH_SIZE = 1
MFR_BASE_BATCH_SIZE = 16
OCR_DET_BASE_BATCH_SIZE = 16
TABLE_ORI_CLS_BATCH_SIZE = 16
TABLE_Wired_Wireless_CLS_BATCH_SIZE = 16


import re as _re

def _fix_layout_res(layout_res: list) -> list:
    """
    Post-process YOLO layout_res cho một trang:

    Fix 1 — Promote plain_text → title
        YOLO đôi khi classify heading có nền màu là plain_text (cat=1).
        Nếu text khớp pattern heading (số + dấu chấm / chữ hoa toàn bộ / có ➤)
        thì promote lên title (cat=0).
        NOTE: chỉ áp dụng cho block cao ≤ 60px (heading thường là 1 dòng).

    Fix 2 — Dedup bbox trùng
        YOLO đôi khi trả về cùng bbox với 2 type khác nhau (NMS chưa lọc sạch).
        Giữ lại block có score cao nhất cho mỗi bbox.
    """
    # ── Fix 2: dedup bbox ────────────────────────────────────────────────────
    seen: dict = {}  # key = rounded bbox tuple → block có score cao nhất
    for block in layout_res:
        poly = block.get("poly", [])
        if len(poly) < 6:
            continue
        # Round về 10px để bắt các bbox gần giống nhau (sub-pixel diff)
        key = tuple(round(x / 10) for x in [poly[0], poly[1], poly[4], poly[5]])
        score = block.get("score", 0)
        if key not in seen or score > seen[key].get("score", 0):
            seen[key] = block

    deduped = list(seen.values())
    # Giữ thứ tự gốc (sort theo y rồi x)
    deduped.sort(key=lambda b: (b.get("poly", [0, 0])[1], b.get("poly", [0])[0]))

    # ── Fix 1: promote plain_text → title ───────────────────────────────────
    # Pattern heading tiếng Việt:
    #   "2. THỜI GIAN ĐÀO TẠO"  → số + dấu chấm + khoảng trắng + CHỮ HOA
    #   "7.1.1. Kiến thức Tin học" → số thứ cấp + dấu chấm
    #   "Về kiến thức chuyên môn" → có prefix đặc biệt (➤ / ►)
    HEADING_RE = _re.compile(
        r'^(\d+[\.\d]*\.\s+\S)'           # "2. X" hoặc "7.1.1. X"
        r'|^[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠ-Ỹ\s]{6,}$'  # TOÀN CHỮ HOA ≥ 6 ký tự
        r'|^[➤►▶>]\s*\S'                  # prefix mũi tên
    )

    result = []
    for block in deduped:
        cat_id = block.get("category_id")
        if cat_id == 1:  # plain_text
            poly = block.get("poly", [])
            if len(poly) >= 6:
                height = abs(poly[5] - poly[1])
                # Chỉ xét block cao ≤ 80px (1–2 dòng heading)
                if height <= 80:
                    # Lấy text đã OCR nếu có (từ bước trước), hoặc để pipeline tự OCR
                    existing_text = block.get("text", "")
                    if existing_text and HEADING_RE.match(existing_text.strip()):
                        from loguru import logger as _logger
                        _logger.debug(
                            f"[layout_fix] Promote plain_text→title: "
                            f"bbox=[{poly[0]:.0f},{poly[1]:.0f}→{poly[4]:.0f},{poly[5]:.0f}] "
                            f"text={repr(existing_text[:60])}"
                        )
                        block = dict(block)
                        block["category_id"] = 0  # title
        result.append(block)

    return result


class BatchAnalyze:
    def __init__(self, model_manager, batch_ratio: int, formula_enable, table_enable, enable_ocr_det_batch: bool = True):
        self.batch_ratio = batch_ratio
        self.formula_enable = get_formula_enable(formula_enable)
        self.table_enable = get_table_enable(table_enable)
        self.model_manager = model_manager
        self.enable_ocr_det_batch = enable_ocr_det_batch

    def __call__(self, images_with_extra_info: list) -> list:
        if len(images_with_extra_info) == 0:
            return []

        images_layout_res = []

        self.model = self.model_manager.get_model(
            lang=None,
            formula_enable=self.formula_enable,
            table_enable=self.table_enable,
        )
        atom_model_manager = AtomModelSingleton()

        pil_images = [image for image, _, _ in images_with_extra_info]

        np_images = [np.asarray(image) for image, _, _ in images_with_extra_info]

        # doclayout_yolo

        images_layout_res += self.model.layout_model.batch_predict(
            pil_images, YOLO_LAYOUT_BASE_BATCH_SIZE
        )

        if self.formula_enable:
            # 公式检测
            images_mfd_res = self.model.mfd_model.batch_predict(
                np_images, MFD_BASE_BATCH_SIZE
            )

            # 公式识别
            images_formula_list = self.model.mfr_model.batch_predict(
                images_mfd_res,
                np_images,
                batch_size=self.batch_ratio * MFR_BASE_BATCH_SIZE,
            )
            mfr_count = 0
            for image_index in range(len(np_images)):
                images_layout_res[image_index] += images_formula_list[image_index]
                mfr_count += len(images_formula_list[image_index])

        # 清理显存
        clean_vram(self.model.device, vram_threshold=8)

        ocr_res_list_all_page = []
        table_res_list_all_page = []
        for index in range(len(np_images)):
            _, ocr_enable, _lang = images_with_extra_info[index]
            layout_res = images_layout_res[index]
            np_img = np_images[index]

            # ── Post-process layout_res ────────────────────────────────────────
            layout_res = _fix_layout_res(layout_res)
            images_layout_res[index] = layout_res
            # ──────────────────────────────────────────────────────────────────

            ocr_res_list, table_res_list, single_page_mfdetrec_res = (
                get_res_list_from_layout_res(layout_res)
            )

            ocr_res_list_all_page.append({'ocr_res_list':ocr_res_list,
                                          'lang':_lang,
                                          'ocr_enable':ocr_enable,
                                          'np_img':np_img,
                                          'single_page_mfdetrec_res':single_page_mfdetrec_res,
                                          'layout_res':layout_res,
                                          })

            for table_res in table_res_list:
                def get_crop_table_img(scale):
                    crop_xmin, crop_ymin = int(table_res['poly'][0]), int(table_res['poly'][1])
                    crop_xmax, crop_ymax = int(table_res['poly'][4]), int(table_res['poly'][5])
                    bbox = (int(crop_xmin / scale), int(crop_ymin / scale), int(crop_xmax / scale), int(crop_ymax / scale))
                    return get_crop_np_img(bbox, np_img, scale=scale)

                wireless_table_img = get_crop_table_img(scale = 1)
                wired_table_img = get_crop_table_img(scale = 10/3)

                table_res_list_all_page.append({'table_res':table_res,
                                                'lang':_lang,
                                                'table_img':wireless_table_img,
                                                'wired_table_img':wired_table_img,
                                                'ocr_enable': ocr_enable,
                                              })

        # 表格识别 table recognition
        if self.table_enable:
            
            # Check if any tables use LightOnOCR (vi-light-ocr, vi-vision-light, vi-hybrid, or vi-custom with lighton backend)
            lighton_langs = ['vi-light-ocr', 'vi-vision-light', 'vi-hybrid']
            
            # Check for vi-custom with lighton backend
            is_lighton_backend = os.getenv('MINERU_TABLE_BACKEND', 'paddle').lower() == 'lighton'
            
            lighton_tables = [
                t for t in table_res_list_all_page 
                if (t.get('lang') in lighton_langs or 
                (t.get('lang') and t.get('lang').startswith('vi') and is_lighton_backend))
                and t.get('ocr_enable', False)
            ]
            
            other_tables = [
                t for t in table_res_list_all_page 
                if not (t.get('lang') in lighton_langs or 
                (t.get('lang') and t.get('lang').startswith('vi') and is_lighton_backend))
                or not t.get('ocr_enable', False)
            ]


            
            # Process LightOnOCR tables directly
            if lighton_tables:
                from mineru.model.ocr.lighton_ocr import LightOnOCR
                lighton_ocr = LightOnOCR(
                    server_url=os.getenv('LIGHTON_SERVER_URL', 'http://localhost:1234/v1/chat/completions'),
                    model_name=os.getenv('LIGHTON_MODEL_NAME', 'lightonai/LightOnOCR-2-1B')
                )
                for table_res_dict in tqdm(lighton_tables, desc="LightOnOCR Table"):
                    bgr_image = cv2.cvtColor(table_res_dict["table_img"], cv2.COLOR_RGB2BGR)
                    html_result = lighton_ocr.recognize_table(bgr_image)
                    table_res_dict["table_res"]["html"] = html_result
                    # Mark as processed to skip regular table processing
                    table_res_dict["lighton_processed"] = True
            
            # Process remaining tables with standard pipeline
            standard_table_list = [t for t in table_res_list_all_page if not t.get("lighton_processed", False)]
            
            # Skip standard processing if no tables left
            if not standard_table_list:
                pass  # All tables were processed by LightOnOCR
            else:
                # 图片旋转批量处理
                img_orientation_cls_model = atom_model_manager.get_atom_model(
                    atom_model_name=AtomicModel.ImgOrientationCls,
                )
                try:
                    if self.enable_ocr_det_batch:
                        img_orientation_cls_model.batch_predict(standard_table_list,
                                                                det_batch_size=self.batch_ratio * OCR_DET_BASE_BATCH_SIZE,
                                                                batch_size=TABLE_ORI_CLS_BATCH_SIZE)
                    else:
                        for table_res in standard_table_list:
                            rotate_label = img_orientation_cls_model.predict(table_res['table_img'])
                            img_orientation_cls_model.img_rotate(table_res, rotate_label)
                except Exception as e:
                    logger.warning(
                        f"Image orientation classification failed: {e}, using original image"
                    )

                # 表格分类
                table_cls_model = atom_model_manager.get_atom_model(
                    atom_model_name=AtomicModel.TableCls,
                )
                try:
                    table_cls_model.batch_predict(standard_table_list,
                                                  batch_size=TABLE_Wired_Wireless_CLS_BATCH_SIZE)
                except Exception as e:
                    logger.warning(
                    f"Table classification failed: {e}, using default model"
                )

            # OCR det 过程，顺序执行
            rec_img_lang_group = defaultdict(list)
            det_ocr_engine = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.OCR,
                det_db_box_thresh=0.5,
                det_db_unclip_ratio=1.6,
                enable_merge_det_boxes=False,
            )
            for index, table_res_dict in enumerate(
                    tqdm(standard_table_list, desc="Table-ocr det")
            ):
                bgr_image = cv2.cvtColor(table_res_dict["table_img"], cv2.COLOR_RGB2BGR)
                ocr_result = det_ocr_engine.ocr(bgr_image, rec=False)[0]
                # 构造需要 OCR 识别的图片字典，包括cropped_img, dt_box, table_id，并按照语言进行分组
                for dt_box in ocr_result:
                    rec_img_lang_group[_lang].append(
                        {
                            "cropped_img": get_rotate_crop_image(
                                bgr_image, np.asarray(dt_box, dtype=np.float32)
                            ),
                            "dt_box": np.asarray(dt_box, dtype=np.float32),
                            "table_id": index,
                        }
                    )

            # OCR rec，按照语言分批处理
            for _lang, rec_img_list in rec_img_lang_group.items():
                ocr_engine = atom_model_manager.get_atom_model(
                    atom_model_name=AtomicModel.OCR,
                    det_db_box_thresh=0.5,
                    det_db_unclip_ratio=1.6,
                    lang=_lang,
                    enable_merge_det_boxes=False,
                )
                cropped_img_list = [item["cropped_img"] for item in rec_img_list]
                ocr_res_list = ocr_engine.ocr(cropped_img_list, det=False, tqdm_enable=True, tqdm_desc=f"Table-ocr rec {_lang}")[0]
                # 按照 table_id 将识别结果进行回填
                for img_dict, ocr_res in zip(rec_img_list, ocr_res_list):
                    if standard_table_list[img_dict["table_id"]].get("ocr_result"):
                        standard_table_list[img_dict["table_id"]]["ocr_result"].append(
                            [img_dict["dt_box"], html.escape(ocr_res[0]), ocr_res[1]]
                        )
                    else:
                        standard_table_list[img_dict["table_id"]]["ocr_result"] = [
                            [img_dict["dt_box"], html.escape(ocr_res[0]), ocr_res[1]]
                        ]

            clean_vram(self.model.device, vram_threshold=8)

            # 先对所有表格使用无线表格模型，然后对分类为有线的表格使用有线表格模型
            wireless_table_model = atom_model_manager.get_atom_model(
                atom_model_name=AtomicModel.WirelessTable,
            )
            wireless_table_model.batch_predict(standard_table_list)

            # 单独拿出有线表格进行预测
            wired_table_res_list = []
            for table_res_dict in standard_table_list:
                # logger.debug(f"Table classification result: {table_res_dict["table_res"]["cls_label"]} with confidence {table_res_dict["table_res"]["cls_score"]}")
                if (
                    (table_res_dict["table_res"]["cls_label"] == AtomicModel.WirelessTable and table_res_dict["table_res"]["cls_score"] < 0.9)
                    or table_res_dict["table_res"]["cls_label"] == AtomicModel.WiredTable
                ):
                    wired_table_res_list.append(table_res_dict)
                del table_res_dict["table_res"]["cls_label"]
                del table_res_dict["table_res"]["cls_score"]
            if wired_table_res_list:
                for table_res_dict in tqdm(
                        wired_table_res_list, desc="Table-wired Predict"
                ):
                    if not table_res_dict.get("ocr_result", None):
                        continue

                    wired_table_model = atom_model_manager.get_atom_model(
                        atom_model_name=AtomicModel.WiredTable,
                        lang=table_res_dict["lang"],
                    )
                    table_res_dict["table_res"]["html"] = wired_table_model.predict(
                        table_res_dict["wired_table_img"],
                        table_res_dict["ocr_result"],
                        table_res_dict["table_res"].get("html", None)
                    )

            # 表格格式清理
            for table_res_dict in table_res_list_all_page:
                html_code = table_res_dict["table_res"].get("html", "") or ""

                # 检查html_code是否包含'<table>'和'</table>'
                if "<table>" in html_code and "</table>" in html_code:
                    # 选用<table>到</table>的内容，放入table_res_dict['table_res']['html']
                    start_index = html_code.find("<table>")
                    end_index = html_code.rfind("</table>") + len("</table>")
                    table_res_dict["table_res"]["html"] = html_code[start_index:end_index]

        # OCR det
        if self.enable_ocr_det_batch:
            # 批处理模式 - 按语言和分辨率分组
            # 收集所有需要OCR检测的裁剪图像
            all_cropped_images_info = []

            for ocr_res_list_dict in ocr_res_list_all_page:
                _lang = ocr_res_list_dict['lang']

                for res in ocr_res_list_dict['ocr_res_list']:
                    new_image, useful_list = crop_img(
                        res, ocr_res_list_dict['np_img'], crop_paste_x=50, crop_paste_y=50
                    )
                    adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                        ocr_res_list_dict['single_page_mfdetrec_res'], useful_list
                    )

                    # BGR转换
                    bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

                    all_cropped_images_info.append((
                        bgr_image, useful_list, ocr_res_list_dict, res, adjusted_mfdetrec_res, _lang
                    ))

            # 按语言分组
            lang_groups = defaultdict(list)
            for crop_info in all_cropped_images_info:
                lang = crop_info[5]
                lang_groups[lang].append(crop_info)

            # 对每种语言按分辨率分组并批处理
            for lang, lang_crop_list in lang_groups.items():
                if not lang_crop_list:
                    continue

                # logger.info(f"Processing OCR detection for language {lang} with {len(lang_crop_list)} images")

                # 获取OCR模型
                ocr_model = atom_model_manager.get_atom_model(
                    atom_model_name=AtomicModel.OCR,
                    det_db_box_thresh=0.3,
                    lang=lang
                )

                # 按分辨率分组并同时完成padding
                # RESOLUTION_GROUP_STRIDE = 32
                RESOLUTION_GROUP_STRIDE = 64

                resolution_groups = defaultdict(list)
                for crop_info in lang_crop_list:
                    cropped_img = crop_info[0]
                    h, w = cropped_img.shape[:2]
                    # 直接计算目标尺寸并用作分组键
                    target_h = ((h + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
                    target_w = ((w + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
                    group_key = (target_h, target_w)
                    resolution_groups[group_key].append(crop_info)

                # 对每个分辨率组进行批处理
                for (target_h, target_w), group_crops in tqdm(resolution_groups.items(), desc=f"OCR-det {lang}"):
                    # Check if this OCR model supports batch detection (PaddleOCR)
                    # or needs full-image OCR (EasyOCR, LightOnOCR, ConfigurableHybridOCR)
                    has_text_det = hasattr(ocr_model, 'text_detector')
                    if not has_text_det:
                        # Check if wrapped in ConfigurableHybridOCR
                        has_text_det = getattr(ocr_model, 'has_text_detector', False)

                    if has_text_det:
                        # --- Fast Paddle batch path ---
                        batch_images = []
                        for crop_info in group_crops:
                            img = crop_info[0]
                            h, w = img.shape[:2]
                            padded_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
                            padded_img[:h, :w] = img
                            batch_images.append(padded_img)

                        det_batch_size = min(len(batch_images), self.batch_ratio * OCR_DET_BASE_BATCH_SIZE)
                        batch_results = ocr_model.text_detector.batch_predict(batch_images, det_batch_size)

                        for crop_info, (dt_boxes, _) in zip(group_crops, batch_results):
                            bgr_image, useful_list, ocr_res_list_dict, res, adjusted_mfdetrec_res, _lang = crop_info

                            if dt_boxes is not None and len(dt_boxes) > 0:
                                dt_boxes_sorted = sorted_boxes(dt_boxes)
                                dt_boxes_merged = merge_det_boxes(dt_boxes_sorted) if dt_boxes_sorted else []
                                dt_boxes_final = (update_det_boxes(dt_boxes_merged, adjusted_mfdetrec_res)
                                                  if dt_boxes_merged and adjusted_mfdetrec_res
                                                  else dt_boxes_merged)

                                if dt_boxes_final:
                                    ocr_res = [box.tolist() if hasattr(box, 'tolist') else box for box in dt_boxes_final]
                                    ocr_result_list = get_ocr_result_list(
                                        ocr_res, useful_list, ocr_res_list_dict['ocr_enable'], bgr_image, _lang
                                    )
                                    ocr_res_list_dict['layout_res'].extend(ocr_result_list)

                    else:
                        # --- Full-image OCR path for EasyOCR / LightOnOCR ---
                        # These backends detect + recognize in ONE step.
                        # We inject the text directly as final spans (no re-recognition needed).
                        for crop_info in group_crops:
                            bgr_image, useful_list, ocr_res_list_dict, res, adjusted_mfdetrec_res, _lang = crop_info
                            cat_id = res.get('category_id', '?')
                            h_img, w_img = bgr_image.shape[:2]

                            try:
                                ocr_result = ocr_model.ocr(bgr_image, det=True, rec=True,
                                                           mfd_res=adjusted_mfdetrec_res)[0]
                            except Exception as e:
                                logger.warning(f"Full OCR failed for {_lang}: {e}")
                                ocr_result = None

                            if ocr_result is None:
                                logger.warning(f"[OCR-DET] cat={cat_id} img={w_img}x{h_img} → EMPTY (no span added)")
                            else:
                                # Build [box_4pts, (text, score)] list for get_ocr_result_list.
                                # get_ocr_result_list with ocr_enable=False injects text as FINAL spans
                                # (no re-recognition in OCR-rec phase), fixing empty headers.
                                compatible_res = []
                                for item in ocr_result:
                                    if len(item) == 2:
                                        box, (text, score) = item
                                        if not text or not text.strip():
                                            continue
                                        logger.info(f"[OCR-DET] cat={cat_id} img={w_img}x{h_img} → text={repr(text[:50])}")
                                        box_arr = np.array(box, dtype=np.float32)
                                        if box_arr.shape == (4, 2):
                                            pts = box_arr.tolist()
                                        else:
                                            x0, y0, x1, y1 = box_arr.flatten()[:4]
                                            pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                                        compatible_res.append([pts, (text, score)])

                                if compatible_res:
                                    # ocr_enable=False: text is already known, skip rec phase
                                    ocr_result_list = get_ocr_result_list(
                                        compatible_res, useful_list,
                                        False, bgr_image, _lang
                                    )
                                    ocr_res_list_dict['layout_res'].extend(ocr_result_list)
                                else:
                                    logger.warning(f"[OCR-DET] cat={cat_id} img={w_img}x{h_img} → all items empty text (filtered)")


        else:
            # 原始单张处理模式
            for ocr_res_list_dict in tqdm(ocr_res_list_all_page, desc="OCR-det Predict"):
                # Process each area that requires OCR processing
                _lang = ocr_res_list_dict['lang']
                # Get OCR results for this language's images
                ocr_model = atom_model_manager.get_atom_model(
                    atom_model_name=AtomicModel.OCR,
                    ocr_show_log=False,
                    det_db_box_thresh=0.3,
                    lang=_lang
                )

                # Check if backend supports separate detection (Paddle) or full-image OCR
                has_text_det = hasattr(ocr_model, 'text_detector')
                if not has_text_det:
                    has_text_det = getattr(ocr_model, 'has_text_detector', False)

                for res in ocr_res_list_dict['ocr_res_list']:
                    new_image, useful_list = crop_img(
                        res, ocr_res_list_dict['np_img'], crop_paste_x=50, crop_paste_y=50
                    )
                    adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                        ocr_res_list_dict['single_page_mfdetrec_res'], useful_list
                    )
                    bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

                    if has_text_det:
                        # --- Paddle: detection-only, then recognize separately ---
                        ocr_res = ocr_model.ocr(
                            bgr_image, mfd_res=adjusted_mfdetrec_res, rec=False
                        )[0]
                        if ocr_res:
                            ocr_result_list = get_ocr_result_list(
                                ocr_res, useful_list, ocr_res_list_dict['ocr_enable'], bgr_image, _lang
                            )
                            ocr_res_list_dict['layout_res'].extend(ocr_result_list)
                    else:
                        # --- EasyOCR / LightOnOCR: full OCR in one step ---
                        # Inject text directly as final spans (no re-recognition).
                        try:
                            ocr_result = ocr_model.ocr(bgr_image, det=True, rec=True,
                                                       mfd_res=adjusted_mfdetrec_res)[0]
                        except Exception as e:
                            logger.warning(f"Full OCR failed for {_lang}: {e}")
                            ocr_result = None

                        if ocr_result:
                            compatible_res = []
                            for item in ocr_result:
                                if len(item) == 2:
                                    box, (text, score) = item
                                    if not text or not text.strip():
                                        continue
                                    box_arr = np.array(box, dtype=np.float32)
                                    if box_arr.shape == (4, 2):
                                        pts = box_arr.tolist()
                                    else:
                                        x0, y0, x1, y1 = box_arr.flatten()[:4]
                                        pts = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                                    compatible_res.append([pts, (text, score)])

                            if compatible_res:
                                # ocr_enable=False: text known, skip rec phase
                                ocr_result_list = get_ocr_result_list(
                                    compatible_res, useful_list,
                                    False, bgr_image, _lang
                                )
                                ocr_res_list_dict['layout_res'].extend(ocr_result_list)

        # OCR rec
        # Create dictionaries to store items by language
        need_ocr_lists_by_lang = {}  # Dict of lists for each language
        img_crop_lists_by_lang = {}  # Dict of lists for each language

        for layout_res in images_layout_res:
            for layout_res_item in layout_res:
                if layout_res_item['category_id'] in [15]:
                    if 'np_img' in layout_res_item and 'lang' in layout_res_item:
                        lang = layout_res_item['lang']

                        # Initialize lists for this language if not exist
                        if lang not in need_ocr_lists_by_lang:
                            need_ocr_lists_by_lang[lang] = []
                            img_crop_lists_by_lang[lang] = []

                        # Add to the appropriate language-specific lists
                        need_ocr_lists_by_lang[lang].append(layout_res_item)
                        img_crop_lists_by_lang[lang].append(layout_res_item['np_img'])

                        # Remove the fields after adding to lists
                        layout_res_item.pop('np_img')
                        layout_res_item.pop('lang')

        if len(img_crop_lists_by_lang) > 0:

            # Process OCR by language
            total_processed = 0

            # Process each language separately
            for lang, img_crop_list in img_crop_lists_by_lang.items():
                if len(img_crop_list) > 0:
                    # Get OCR results for this language's images

                    ocr_model = atom_model_manager.get_atom_model(
                        atom_model_name=AtomicModel.OCR,
                        det_db_box_thresh=0.3,
                        lang=lang
                    )
                    ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]

                    # Verify we have matching counts
                    assert len(ocr_res_list) == len(
                        need_ocr_lists_by_lang[lang]), f'ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(need_ocr_lists_by_lang[lang])} for lang: {lang}'

                    # Process OCR results for this language
                    for index, layout_res_item in enumerate(need_ocr_lists_by_lang[lang]):
                        ocr_text, ocr_score = ocr_res_list[index]
                        layout_res_item['text'] = ocr_text
                        layout_res_item['score'] = float(f"{ocr_score:.3f}")
                        if ocr_score < OcrConfidence.min_confidence:
                            layout_res_item['category_id'] = 16
                        else:
                            layout_res_bbox = [layout_res_item['poly'][0], layout_res_item['poly'][1],
                                               layout_res_item['poly'][4], layout_res_item['poly'][5]]
                            layout_res_width = layout_res_bbox[2] - layout_res_bbox[0]
                            layout_res_height = layout_res_bbox[3] - layout_res_bbox[1]
                            if (
                                    ocr_text in [
                                        '（204号', '（20', '（2', '（2号', '（20号', '号', '（204',
                                        '(cid:)', '(ci:)', '(cd:1)', 'cd:)', 'c)', '(cd:)', 'c', 'id:)',
                                        ':)', '√:)', '√i:)', '−i:)', '−:', 'i:)',
                                    ]
                                    and ocr_score < 0.8
                                    and layout_res_width < layout_res_height
                            ):
                                layout_res_item['category_id'] = 16

                    total_processed += len(img_crop_list)

        return images_layout_res
