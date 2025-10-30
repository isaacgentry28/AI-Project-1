import os


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ARTIFACTS_FOLDER = os.path.join(BASE_DIR, 'artifacts')
DB_PATH = os.path.join(BASE_DIR, 'user.db')


ALLOWED_EXTS = {'.csv', '.txt', '.xlsx'}


MAX_CONTENT_LENGTH_MB = int(os.getenv('MAX_CONTENT_LENGTH_MB', '25'))
MAX_CONTENT_LENGTH = MAX_CONTENT_LENGTH_MB * 1024 * 1024