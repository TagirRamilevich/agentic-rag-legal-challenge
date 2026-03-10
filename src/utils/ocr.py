try:
    import pytesseract
    from PIL import Image
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

try:
    import fitz
    _FITZ_AVAILABLE = True
except ImportError:
    _FITZ_AVAILABLE = False


def ocr_page(pdf_path: str, page_number: int, dpi: int = 300) -> str:
    if not _OCR_AVAILABLE or not _FITZ_AVAILABLE:
        return ""
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    try:
        return pytesseract.image_to_string(img)
    except Exception:
        return ""
