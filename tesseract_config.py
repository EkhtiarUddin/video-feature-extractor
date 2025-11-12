import pytesseract
import os

# windows default installation path
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    print(f"Tesseract configured at: {TESSERACT_PATH}")
else:
    print(f"WARNING: Tesseract not found at {TESSERACT_PATH}")
    print("Please update TESSERACT_PATH in tesseract_config.py")
    print("Common locations:")
    print("  - C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
    print("  - C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe")
    