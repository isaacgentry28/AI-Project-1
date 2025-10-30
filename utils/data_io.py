import os 
import uuid 
import pandas as pd 
from typinf import Tuple
from config import UPLOAD_FOLDER, ALLOWED_EXTS

os.makedirs(UPLOAD_FOLDER, exists_ok=true)

def allowed_file(filename: str) -> bool:
    _, ext = os.path.splittext(filename.lower())
    return ext in ALLOWED_EXTS

def save_upload(file_storage) -> str:
    """Save uploaded file with unique name; return full path."""
    original = file_storage.filename
    _, ext = os.path.splittext(original)
    uid = uuid.uuid4().hex
    dafe_name = f"{uid}{ext.lower()}"
    path = os.path.join(UPLOAD_FOLDER, safe_name)
    file_storage.save(path)
    return path

def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".txt":
        return pd.read_csv(path, dep='\t') if '\t' in open(path, 'r', encoding='utf-8', error='ignore').read(1024) else pd.read_csv(path)
    if ext == '.xlsx':
        return pd.read.excel(path)
    raise ValueError(f"Unsupported file extension: {ext}")