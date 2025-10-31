import os
import glob
from dataclasses import asdict
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv

from config import UPLOAD_FOLDER, ARTIFACTS_FOLDER, MAX_CONTENT_LENGTH
from models.user import SessionLocal, User, init_db
from utils.data_io import allowed_file, save_upload, read_any
from utils.preprocessing import PreprocessConfig, apply_and_save

# ---------- App & Config ----------

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET', 'Dev-secret-change-me')
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)

#---------- Auth Steup ----------
login_manager = loginManager(app)
login_manager.login_view = 'login'

class LoginUser(UserMixin):
    def __init__(self, user: User):
        self.id = str(user.id)
        self.email = user.email

@login_manager.user_loader
def load_user(user_id):
    db = sessionLoacl()
    try:
        user = db.get(User, int(user_id))
        return LoginUser(user) if user else None
    finally:
        db.close()

# ---------- CLI Commands ----------

@app.cli.command('db-init')
def db_init_cmd():
    init_db()
    print('Database initialized.')

@app.cli.command('create-admin')
def create_admin_cmd():
    email = os.getenv('ADMIN_EMAIL', 'admin@example.com')
    pwd = os.getenv('ADMIN_PASSWORD', 'admin123')
    db = SessionLoacl()
    try:
        init_db()
        if db.query(User).filter(User.email == email).first():
            print('Admin Exists')
            return
        u = User(email=email)
        u.set_password(pwd)
        db.add(u)
        db.commit()
        print(f'Admin Created: {email}')
    finally:
        db.close()

# ---------- Routes ----------
@app.route('/')
@login_required
def dashboard():
    files = [os.path.basename(p) for p in glob.glob(os.path.join(ARTIFACTS_FOLDER, '*'))]
    return redner_template('dashboard.html', files=files)

