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
from sklearn.preprocessing import LabelEncoder
import json
import joblib
import numpy as np

from models.classical import make_model, ModelSpec
from models.dnn import train_dnn, DNNConfig
from utils.metrics import classification_metrics, regression_metrics, confusion
from utils.plots import plot_confusion, plot_roc, plot_pr

PLOTS_DIR = os.path.join(ARTIFACTS_FOLDER, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)



# ---------- App & Config ----------
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET', 'dev-secret-change-me')
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)

# ---------- Auth Setup ----------
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class LoginUser(UserMixin):
    def __init__(self, user: User):
        self.id = str(user.id)
        self.email = user.email

@login_manager.user_loader
def load_user(user_id):
    db = SessionLocal()
    try:
        user = db.get(User, int(user_id))
        return LoginUser(user) if user else None
    finally:
        db.close()
# ---------- Leaderboard Utils ----------
LEADERBOARD_JSON = os.path.join(ARTIFACTS_FOLDER, 'leaderboard.json')

def _load_splits():
    X_train = np.loadtxt(os.path.join(ARTIFACTS_FOLDER, 'X_train.csv'), delimiter=',', skiprows=1)
    X_val = np.loadtxt(os.path.join(ARTIFACTS_FOLDER, 'X_val.csv'), delimiter=',', skiprows=1)
    X_test = np.loadtxt(os.path.join(ARTIFACTS_FOLDER, 'X_test.csv'), delimiter=',', skiprows=1)
    y_train = np.loadtxt(os.path.join(ARTIFACTS_FOLDER, 'y_train.csv'), delimiter=',', skiprows=1)
    y_val = np.loadtxt(os.path.join(ARTIFACTS_FOLDER, 'y_val.csv'), delimiter=',', skiprows=1)
    y_test = np.loadtxt(os.path.join(ARTIFACTS_FOLDER, 'y_test.csv'), delimiter=',', skiprows=1)
    return X_train, X_val, X_test, y_train, y_val, y_test


def _update_leaderboard(entry):
    lb = []
    if os.path.exists(LEADERBOARD_JSON):
        with open(LEADERBOARD_JSON, 'r') as f:
            lb = json.load(f)
    lb.append(entry)
    with open(LEADERBOARD_JSON, 'w') as f:
        json.dump(lb, f, indent=2)

# ---------- CLI Commands ----------
@app.cli.command('db-init')
def db_init_cmd():
    init_db()
    print('DB initialized.')

@app.cli.command('create-admin')
def create_admin_cmd():
    email = os.getenv('ADMIN_EMAIL', 'admin@example.com')
    pwd = os.getenv('ADMIN_PASSWORD', 'admin123')
    db = SessionLocal()
    try:
        init_db()
        if db.query(User).filter(User.email == email).first():
            print('Admin exists.')
            return
        u = User(email=email)
        u.set_password(pwd)
        db.add(u)
        db.commit()
        print(f'Admin created: {email}')
    finally:
        db.close()

# ---------- Routes ----------
@app.route('/')
@login_required
def dashboard():
    files = [os.path.basename(p) for p in glob.glob(os.path.join(ARTIFACTS_FOLDER, '*'))]
    return render_template('dashboard.html', files=files)

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email','').strip().lower()
        password = request.form.get('password','')
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == email).first()
            if not user or not user.check_password(password):
                flash('Invalid credentials', 'error')
                return render_template('login.html')
            login_user(LoginUser(user))
            return redirect(url_for('dashboard'))
        finally:
            db.close()
    return render_template('login.html')

@app.route('/train', methods=['GET','POST'])
@login_required
def train():
    context = {}
    if request.method == 'POST':
        task = request.form.get('task')  # 'classification' | 'regression'
        model_name = request.form.get('model')
        params = {}
        # Exposed params (optional)
        if model_name in ('tree','random_forest','boosting'):
            try:
                params['max_depth'] = int(request.form.get('max_depth')) if request.form.get('max_depth') else None
            except: pass
            try:
                params['n_estimators'] = int(request.form.get('n_estimators')) if request.form.get('n_estimators') else None
            except: pass
        if model_name == 'svm':
            C = request.form.get('C')
            if C: params['C'] = float(C)
            kernel = request.form.get('kernel')
            if kernel: params['kernel'] = kernel

        try:
            X_train, X_val, X_test, y_train, y_val, y_test = _load_splits()
        except Exception:
            flash('Please run preprocessing with a target first.', 'error')
            return render_template('train.html', **context)

        # Label-encode non-integer class labels
        if task == 'classification' and (y_train.dtype.kind not in ('i','b')):
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_val = le.transform(y_val)
            y_test = le.transform(y_test)

        if model_name == 'dnn':
            hidden = request.form.get('hidden','128,64')
            hidden = [int(x.strip()) for x in hidden.split(',') if x.strip()]
            epochs = int(request.form.get('epochs','30'))
            lr = float(request.form.get('lr','0.001'))
            cfg = DNNConfig(hidden=hidden, epochs=epochs, lr=lr)
            model, predict, predict_proba, task_type = train_dnn(X_train, y_train, X_val, y_val, cfg)
            y_pred = predict(X_test)
            y_proba = None
            if task_type != 'regression':
                proba = predict_proba(X_test)
                if proba.shape[1] == 2:
                    y_proba = proba[:,1]
            metrics = regression_metrics(y_test, y_pred) if task_type == 'regression' else classification_metrics(y_test, y_pred, y_proba)
        else:
            spec = ModelSpec(name=model_name, task=task, params=params)
            model = make_model(spec)
            model.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))
            if task == 'regression':
                y_pred = model.predict(X_test)
                metrics = regression_metrics(y_test, y_pred)
                y_proba = None
            else:
                y_pred = model.predict(X_test)
                y_proba = None
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_test)
                    if proba.shape[1] == 2:
                        y_proba = proba[:,1]
                metrics = classification_metrics(y_test, y_pred, y_proba)

        # Plots
        run_name = f"{model_name}_{task}"
        plot_paths = {}
        if task == 'classification':
            cm = confusion(y_test, y_pred)
            labels = [str(l) for l in sorted(list(set(y_test)))]
            cm_path = os.path.join(PLOTS_DIR, f'{run_name}_cm.png')
            plot_confusion(cm, labels, cm_path)
            plot_paths['confusion'] = cm_path
            if y_proba is not None:
                from sklearn.metrics import roc_curve, precision_recall_curve
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                pr, rc, _ = precision_recall_curve(y_test, y_proba)
                roc_path = os.path.join(PLOTS_DIR, f'{run_name}_roc.png')
                pr_path = os.path.join(PLOTS_DIR, f'{run_name}_pr.png')
                plot_roc(fpr, tpr, roc_path)
                plot_pr(pr, rc, pr_path)
                plot_paths['roc'] = roc_path
                plot_paths['pr'] = pr_path

        # Leaderboard entry
        entry = {
            'model': model_name,
            'task': task,
            'metrics': metrics,
            'plots': plot_paths,
        }
        _update_leaderboard(entry)
        flash('Training & evaluation complete. See Compare page for leaderboard.', 'success')
        return redirect(url_for('compare'))

    return render_template('train.html')


@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email','').strip().lower()
        password = request.form.get('password','')
        if not email or not password:
            flash('Email and password required', 'error')
            return render_template('register.html')
        db = SessionLocal()
        try:
            if db.query(User).filter(User.email == email).first():
                flash('Email already registered', 'error')
                return render_template('register.html')
            u = User(email=email)
            u.set_password(password)
            db.add(u)
            db.commit()
            flash('Account created, please login', 'success')
            return redirect(url_for('login'))
        finally:
            db.close()
    return render_template('register.html')

@app.route('/compare')
@login_required
def compare():
    lb = []
    if os.path.exists(LEADERBOARD_JSON):
        with open(LEADERBOARD_JSON, 'r') as f:
            lb = json.load(f)
    def score(item):
        m = item['metrics']
        return m.get('roc_auc', m.get('accuracy', m.get('r2', -1.0)))
    lb_sorted = sorted(lb, key=score, reverse=True)
    return render_template('compare.html', runs=lb_sorted)


@app.route('/static_proxy')
@login_required
def static_proxy():
    path = request.args.get('path','')
    if not path or not os.path.abspath(path).startswith(os.path.abspath(ARTIFACTS_FOLDER)):
        flash('Invalid file path', 'error')
        return redirect(url_for('compare'))
    from flask import send_file
    return send_file(path)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/upload', methods=['GET','POST'])
@login_required
def upload():
    uploaded = None
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'error')
            return render_template('upload.html', uploaded=None)
        f = request.files['file']
        if f.filename == '':
            flash('No selected file', 'error')
            return render_template('upload.html', uploaded=None)
        if not allowed_file(f.filename):
            flash('Unsupported file type. Use .csv, .txt, or .xlsx', 'error')
            return render_template('upload.html', uploaded=None)
        try:
            path = save_upload(f)
            # quick schema check: at least 1 column & 1 row
            df = read_any(path)
            if df.shape[0] == 0 or df.shape[1] == 0:
                os.remove(path)
                flash('File appears empty or invalid schema', 'error')
            else:
                uploaded = os.path.basename(path)
                flash('Upload successful', 'success')
        except Exception as e:
            flash(f'Upload error: {e}', 'error')
    return render_template('upload.html', uploaded=uploaded)

@app.route('/preprocess', methods=['GET','POST'])
@login_required
def preprocess():
    stats = None
    if request.method == 'POST':
        target = request.form.get('target') or None
        impute_strategy = request.form.get('impute_strategy','median')
        scale_numeric = request.form.get('scale_numeric','true') == 'true'
        one_hot_categorical = request.form.get('one_hot_categorical','true') == 'true'
        test_size = float(request.form.get('test_size','0.2'))
        val_size = float(request.form.get('val_size','0.1'))
        random_state = int(request.form.get('random_state','42'))

        # Use most recent uploaded file
        try:
            latest = sorted(glob.glob(os.path.join(UPLOAD_FOLDER, '*')), key=os.path.getmtime, reverse=True)[0]
        except IndexError:
            flash('Please upload a dataset first', 'error')
            return render_template('preprocess.html', stats=None)

        try:
            df = read_any(latest)
            cfg = PreprocessConfig(
                target=target,
                impute_strategy=impute_strategy,
                scale_numeric=scale_numeric,
                one_hot_categorical=one_hot_categorical,
                test_size=test_size,
                val_size=val_size,
                random_state=random_state,
            )
            out = apply_and_save(df, cfg)
            stats = type('Obj', (), out)  # simple dot-access in template
            flash('Preprocessing complete. Artifacts saved.', 'success')
        except Exception as e:
            flash(f'Preprocessing error: {e}', 'error')
    return render_template('preprocess.html', stats=stats)

if __name__ == '__main__':
    app.run(debug=True)


