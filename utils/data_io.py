import os 
import uuid 
import pandas as pd 
from typing import Tuple
from config import UPLOAD_FOLDER, ALLOWED_EXTS

os.makedirs(UPLOAD_FOLDER, exists_ok=True)

def allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTS

def save_upload(file_storage) -> str:
    """Save uploaded file with unique name; return full path."""
    original = file_storage.filename
    _, ext = os.path.splitext(original)
    uid = uuid.uuid4().hex
    safe_name = f"{uid}{ext.lower()}"
    path = os.path.join(UPLOAD_FOLDER, safe_name)
    file_storage.save(path)
    return path

def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".txt":
        return pd.read_csv(path, sep='\t') if '\t' in open(path, 'r', encoding='utf-8', errors='ignore').read(1024) else pd.read_csv(path)
    if ext == '.xlsx':
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file extension: {ext}")