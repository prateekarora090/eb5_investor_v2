import PyPDF2
import pytesseract
from pdf2image import convert_from_bytes
from io import BytesIO
import cv2
import numpy as np
import hashlib
import os
import json

def read_pdf(file_content, max_pages=50):
    file_hash = hashlib.md5(file_content).hexdigest()
    cache_file = f"cache/{file_hash}.json"
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    text_content = extract_text(file_content, max_pages)
    visual_content = extract_visual_content(file_content, max_pages)
    
    result = {
        "text_content": text_content,
        "visual_content": visual_content
    }
    
    os.makedirs("cache", exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(result, f)
    
    return result

def extract_text(file_content, max_pages):
    reader = PyPDF2.PdfReader(BytesIO(file_content))
    text = ""
    for i, page in enumerate(reader.pages):
        if i >= max_pages:
            break
        text += page.extract_text()
    return text

def extract_visual_content(file_content, max_pages):
    images = convert_from_bytes(file_content, last_page=max_pages)
    visual_content = ""
    for i, image in enumerate(images):
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # OCR for text in images
        text = pytesseract.image_to_string(gray)
        visual_content += f"Page {i+1} Image Text:\n{text}\n"
        
        # Basic shape detection (for charts/tables)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        if lines is not None:
            visual_content += f"Page {i+1} contains potential charts/tables.\n"
    
    return visual_content