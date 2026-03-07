# Copyright (c) Opendatalab. All rights reserved.
"""
Vietnamese language detection utility for MinerU.
Detects if a PDF contains Vietnamese text by analyzing character distribution.
"""
import re
from io import BytesIO

import pypdfium2 as pdfium
from loguru import logger


# Vietnamese-specific diacritic characters (unique to Vietnamese)
# These chars are relatively unique to Vietnamese vs other Latin-based languages
_VI_PATTERN = re.compile(
    r'[àáâãèéêìíòóôõùúýăắặằẵẩẫẻẽẹệếềểễịỉỏọộốổỗớờởỡụủứừửữặẳẫơượưừưảẦẤẬẮẰẲẴẺẸẾỀỂỄỈỊỌỐỔỖỚỜỞỠỤỦỨỪỬỮỳỷỹỵỶỴỹđĐ]',
    re.UNICODE
)

_VI_CHARS = set(
    'àáâãèéêìíòóôõùúýăắặằẵẩẫẻẽẹệếềểễịỉỏọộốổỗớờởỡụủứừửữặẳẫơượưừưảđĐ'
    'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂẮẶẰẴẨẪẺẼẸỆẾỀỂỄỊỈỎỌỘỐỔỖỚỜỞỠỤỦỨỪỬỮẶẲẪƠƯỢƯỪƯẢĐ'
    'ỳỷỹỵỶỴ'
)


def detect_vietnamese(pdf_bytes: bytes, pages_to_check: int = 5, threshold: float = 0.05) -> bool:
    """
    Detect if PDF content is Vietnamese by sampling text from first N pages.

    Args:
        pdf_bytes: Raw PDF bytes 
        pages_to_check: Number of pages to scan (default 5)
        threshold: Minimum ratio of Vietnamese chars to total chars (default 5%)

    Returns:
        True if Vietnamese text is detected, False otherwise
    """
    try:
        pdf = pdfium.PdfDocument(pdf_bytes)
        page_count = len(pdf)

        if page_count == 0:
            pdf.close()
            return False

        pages = min(page_count, pages_to_check)
        total_chars = 0
        vi_chars = 0

        for i in range(pages):
            try:
                page = pdf[i]
                text_page = page.get_textpage()
                text = text_page.get_text_bounded()
                if not text:
                    continue

                # Count total non-whitespace chars
                clean_text = re.sub(r'\s+', '', text)
                total_chars += len(clean_text)

                # Count Vietnamese-unique characters
                for ch in clean_text:
                    if ch in _VI_CHARS:
                        vi_chars += 1
            except Exception as page_err:
                logger.debug(f"Error reading page {i}: {page_err}")
                continue

        pdf.close()

        if total_chars == 0:
            return False

        ratio = vi_chars / total_chars
        logger.debug(f"Vietnamese detection: {vi_chars}/{total_chars} chars = {ratio:.3f} (threshold={threshold})")

        if ratio >= threshold:
            logger.info(f"Vietnamese text detected (ratio={ratio:.3f}), switching to vi pipeline")
            return True

        return False

    except Exception as e:
        logger.warning(f"Vietnamese detection failed: {e}, defaulting to False")
        return False
