# AI-Project-1
# CS456 Machine Learning Web Platform

## 🚀 Overview
This project is a complete **web-based Machine Learning platform** built using Flask, scikit-learn, and PyTorch.  
It allows users to **upload datasets**, **preprocess data**, **train ML models (classical and DNN)**, and **compare results** via metrics and visualizations.

---

## 🧩 Features
- User authentication and session management (Flask-Login + SQLite)
- Upload `.csv`, `.txt`, `.xlsx` datasets (validated on upload)
- Preprocessing pipeline with imputing, scaling, encoding, and deterministic train/val/test split
- Classical ML Models: Linear/Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM
- Deep Neural Network (PyTorch MLP) with configurable layers, epochs, learning rate
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, MSE, MAE, R²
- Visualizations: ROC, PR, Confusion Matrix
- Leaderboard comparison of model runs
- Docker support for reproducibility

---

## 🛠️ Setup Instructions

### Option 1: Local (No Docker)
```bash
# 1. Clone the repository
git clone <your_repo_url>
cd ml_platform

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize database & admin user
export FLASK_APP=app.py  # (Windows PowerShell: $env:FLASK_APP="app.py")
flask db-init
flask create-admin

# 5. Run application
flask run
```
Visit the app at: **http://127.0.0.1:5000**  
Default credentials (if using `.env.example`):  
> Email: `admin@example.com`  
> Password: `admin123`

---

### Option 2: Docker Compose
```bash
docker compose up --build
```
App will be available at **http://localhost:5000**.

---

## ⚙️ Workflow

### 1️⃣ Register / Login
Create a user account or use the admin credentials.

### 2️⃣ Upload Dataset
Supported formats: `.csv`, `.txt`, `.xlsx`.  
The system validates the file before saving.

### 3️⃣ Preprocess
- Specify target column
- Choose imputation strategy, scaling, encoding, and split ratios
- Runs a scikit-learn `ColumnTransformer` and saves train/val/test splits

### 4️⃣ Train Models
- Choose task type (classification/regression)
- Select model type and hyperparameters
- For DNN, define hidden layers, epochs, and learning rate

### 5️⃣ Evaluate & Compare
- View metrics and plots in the **Compare** page
- Metrics saved to `artifacts/leaderboard.json`
- Visuals stored in `artifacts/plots/`

---

## 📦 File Structure
```
ml_platform/
├── app.py
├── config.py
├── models/
│   ├── classical.py
│   └── dnn.py
├── utils/
│   ├── data_io.py
│   ├── preprocessing.py
│   ├── metrics.py
│   └── plots.py
├── templates/
│   ├── base.html
│   ├── login.html
│   ├── register.html
│   ├── upload.html
│   ├── preprocess.html
│   ├── train.html
│   ├── compare.html
│   └── dashboard.html
├── static/
│   └── main.css
├── uploads/
├── artifacts/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🧠 Responsible AI & Data Handling
- Only use publicly shareable datasets without PII.
- Train/validation/test splits are deterministic using a fixed seed.
- Pipeline preprocessing prevents data leakage.
- Evaluate class imbalance with PR-AUC where applicable.

---



---

## 🧹 Cleaning Up
To remove cached data and artifacts:
```bash
rm -rf uploads/* artifacts/*
```
---

© 2025 CS456 Project 1 — Isaac Gentry
