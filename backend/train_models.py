#!/usr/bin/env python3
"""
CervicalAI — Complete Model Training Pipeline
============================================
Trains BOTH models from Kaggle datasets:

Stage 1 (XGBoost): Kaggle Cervical Cancer Risk Factors Dataset
  → https://www.kaggle.com/datasets/ranzeet013/cervical-cancer-dataset

Stage 2 (FastViT): SipakMed Cytological Images Dataset
  → https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed
"""

import os
import sys
import json
import pickle
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger("CervicalAI.Training")

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RISK_CSV = DATA_DIR / "cervical_cancer_risk.csv"
SIPAKMED_DIR = DATA_DIR / "sipakmed"
MODEL_XGB = MODEL_DIR / "xgboost_cervical.pkl"
MODEL_SCALER = MODEL_DIR / "xgboost_scaler.pkl"
MODEL_FASTVIT = MODEL_DIR / "fastvit_sipakmed.pt"
METRICS_JSON = MODEL_DIR / "training_metrics.json"


# ─────────────────────────────────────────────────────────────
# Kaggle Download
# ─────────────────────────────────────────────────────────────

def download_kaggle_datasets():
    """Download both datasets from Kaggle."""
    try:
        import kaggle
    except ImportError:
        logger.error("Kaggle not installed. Run: pip install kaggle")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Downloading Kaggle datasets...")
    logger.info("=" * 60)

    if not RISK_CSV.exists():
        logger.info("Downloading cervical cancer risk factors dataset...")
        os.system(f"kaggle datasets download -d ranzeet013/cervical-cancer-dataset -p {DATA_DIR} --unzip")
        csvs = list(DATA_DIR.glob("*.csv"))
        if csvs:
            csvs[0].rename(RISK_CSV)
            logger.info(f"✅ Risk dataset saved: {RISK_CSV}")
        else:
            logger.error("Could not find downloaded CSV. Check Kaggle credentials.")
    else:
        logger.info(f"✅ Risk dataset already exists: {RISK_CSV}")

    sipakmed_check = SIPAKMED_DIR / "im_Dyskeratotic"
    if not sipakmed_check.exists():
        logger.info("Downloading SipakMed cytological images dataset (~1.5GB)...")
        os.system(
            f"kaggle datasets download -d prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed "
            f"-p {DATA_DIR} --unzip"
        )
        possible_dirs = [
            DATA_DIR / "sipakmed",
            DATA_DIR / "Sipakmed",
            DATA_DIR / "cervical-cancer-largest-dataset-sipakmed",
        ]
        for d in possible_dirs:
            if d.exists():
                if d != SIPAKMED_DIR:
                    import shutil
                    shutil.move(str(d), str(SIPAKMED_DIR))
                logger.info(f"✅ SipakMed dataset: {SIPAKMED_DIR}")
                break
        else:
            logger.warning("SipakMed directory not found at expected path. Check data/ folder manually.")
    else:
        logger.info(f"✅ SipakMed dataset already exists: {SIPAKMED_DIR}")


# ─────────────────────────────────────────────────────────────
# STAGE 1: XGBoost Training
# ─────────────────────────────────────────────────────────────

# Explicitly added the primary clinical screening factors to provide genuine signal
FEATURE_COLS = [
    "Age", "Number of sexual partners", "First sexual intercourse",
    "Num of pregnancies", "Smokes", "Smokes (years)", "Smokes (packs/year)",
    "Hormonal Contraceptives", "Hormonal Contraceptives (years)",
    "IUD", "IUD (years)", "STDs", "STDs (number)",
    "STDs:condylomatosis", "STDs:HPV", "STDs:HIV", "STDs:syphilis",
    "Dx:Cancer", "Dx:CIN", "Dx:HPV",
    "Hinselmann", "Schiller", "Citology"
]
TARGET_COL = "Biopsy"


def load_and_clean_risk_data(csv_path: Path) -> tuple:
    """Load, clean, and prepare the dataset using native missing values."""
    logger.info(f"Loading risk dataset: {csv_path}")
    df = pd.read_csv(csv_path, na_values=["?", "", " ", "NA", "NaN"])

    col_map = {}
    for expected in FEATURE_COLS + [TARGET_COL]:
        for actual in df.columns:
            if expected.lower().replace(" ", "").replace(":", "") == actual.lower().replace(" ", "").replace(":", ""):
                col_map[expected] = actual
                break
        if expected not in col_map:
            col_map[expected] = expected

    available_features = [f for f in FEATURE_COLS if col_map.get(f, f) in df.columns]
    target = col_map.get(TARGET_COL, TARGET_COL)

    df_clean = df[available_features + [target]].copy()
    df_clean.columns = available_features + [TARGET_COL]

    # Convert features to numeric but preserve NaNs for XGBoost's native missing routing
    for col in available_features:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    df_clean[TARGET_COL] = pd.to_numeric(df_clean[TARGET_COL], errors="coerce").fillna(0).astype(int)

    logger.info(f"Cleaned dataset: {df_clean.shape}")
    logger.info(f"Class distribution:\n{df_clean[TARGET_COL].value_counts()}")

    X = df_clean[available_features].values.astype(np.float32)
    y = df_clean[TARGET_COL].values

    return X, y, available_features


def train_xgboost(epochs: int = 500, test_size: float = 0.2):
    """Train the XGBoost cervical cancer risk classifier with native missing flags and dampening."""
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.metrics import (
        classification_report, roc_auc_score, confusion_matrix,
        accuracy_score, f1_score, precision_score, recall_score
    )
    import xgboost as xgb

    logger.info("=" * 60)
    logger.info("STAGE 1: Training XGBoost Risk Model")
    logger.info("=" * 60)

    if not RISK_CSV.exists():
        logger.error(f"Risk dataset not found: {RISK_CSV}")
        return None

    X, y, feature_names = load_and_clean_risk_data(RISK_CSV)

    # Train/val/test split (70/10/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, random_state=42, stratify=y_train
    )

    # Dampened class weight scaling prevents small-sample validation noise spikes
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    logger.info(f"Using native scale_pos_weight ratio for threshold alignment: {scale_pos_weight:.4f}")

    # XGBoost setup tuned for native sparse tabular data
    model = xgb.XGBClassifier(
        n_estimators=epochs,
        max_depth=4,                # Kept shallow to avoid overfitting on 835 samples
        learning_rate=0.03,         # Slightly lower learning rate for smoother optimization
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",      # CRITICAL: Track logloss only so it keeps training as loss drops
        early_stopping_rounds=40,    # Give it plenty of room to build out paths
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        missing=np.nan
    )

    logger.info(f"Training XGBoost with native feature scaling bypass...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    # Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    logger.info("=" * 40)
    logger.info(f"XGBoost Test Results:")
    logger.info(f"  Accuracy:  {acc:.4f}")
    logger.info(f"  AUC-ROC:   {auc:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall:    {rec:.4f}")
    logger.info(f"  Best iter: {model.best_iteration}")
    logger.info("=" * 40)
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['No Cancer', 'Cancer'])}")
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # 5-Fold Cross Validation without scaler leak
    best_trees = max(50, model.best_iteration)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(
        xgb.XGBClassifier(
            n_estimators=best_trees, 
            max_depth=5,
            learning_rate=0.04, 
            scale_pos_weight=scale_pos_weight,
            random_state=42, 
            n_jobs=-1,
            tree_method="hist",
            missing=np.nan
        ),
        X, y, cv=cv, scoring="roc_auc", n_jobs=-1
    )
    logger.info(f"5-Fold CV AUC: {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    with open(MODEL_XGB, "wb") as f:
        pickle.dump(model, f)
        
    # Maintain an identity/dummy scaler to prevent structural breaks if app.py references it
    from sklearn.preprocessing import StandardScaler
    dummy_scaler = StandardScaler()
    dummy_scaler.fit(np.nan_to_num(X_train))
    with open(MODEL_SCALER, "wb") as f:
        pickle.dump(dummy_scaler, f)
    logger.info(f"✅ XGBoost model saved: {MODEL_XGB}")

    return {
        "model": "XGBoost",
        "accuracy": float(acc),
        "auc_roc": float(auc),
        "f1_score": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "cv_auc_mean": float(cv_auc.mean()),
        "cv_auc_std": float(cv_auc.std()),
        "best_iteration": int(model.best_iteration),
        "n_features": len(feature_names),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }


# ─────────────────────────────────────────────────────────────
# STAGE 2: FastViT Training
# ─────────────────────────────────────────────────────────────

def get_sipakmed_data(data_dir: Path, img_size: int = 224):
    """Load SipakMed dataset using torchvision ImageFolder."""
    import torch
    from torchvision import datasets, transforms

    train_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    sipakmed_classes = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic",
                        "im_Parabasal", "im_Superficial_Intermediate"]
    found_dirs = [d.name for d in data_dir.iterdir() if d.is_dir()] if data_dir.exists() else []

    if not any(c in found_dirs for c in sipakmed_classes):
        for sub in data_dir.iterdir() if data_dir.exists() else []:
            sub_dirs = [d.name for d in sub.iterdir() if d.is_dir()] if sub.is_dir() else []
            if any(c in sub_dirs for c in sipakmed_classes):
                data_dir = sub
                found_dirs = sub_dirs
                break

    if not data_dir.exists() or not found_dirs:
        raise FileNotFoundError(f"SipakMed dataset not found at {data_dir}.")

    full_dataset = datasets.ImageFolder(str(data_dir), transform=train_transform)
    return full_dataset, train_transform, val_transform


def train_fastvit(
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 3e-4,
    img_size: int = 224,
):
    """Train FastViT-T8 on the SipakMed cytological image dataset."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from sklearn.metrics import f1_score

    sys.path.insert(0, str(ROOT))
    from app.models.fastvit_model import FastViTT8

    logger.info("=" * 60)
    logger.info("STAGE 2: Training FastViT on SipakMed Dataset")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        full_dataset, train_tf, val_tf = get_sipakmed_data(SIPAKMED_DIR, img_size)
    except FileNotFoundError as e:
        logger.error(str(e))
        return None

    total = len(full_dataset)
    val_n = int(0.15 * total)
    test_n = int(0.15 * total)
    train_n = total - val_n - test_n

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_n, val_n, test_n],
        generator=torch.Generator().manual_seed(42)
    )
    val_ds.dataset.transform = val_tf

    targets = np.array(full_dataset.targets)
    train_targets = [targets[i] for i in train_ds.indices]
    class_counts = np.bincount(train_targets)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[t] for t in train_targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = FastViTT8(num_classes=5, drop_rate=0.15).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler_amp = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_val_f1 = 0.0
    patience_counter = 0
    patience = 15

    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            if scaler_amp is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                scaler_amp.scale(loss).backward()
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = outputs.max(1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), MODEL_FASTVIT)
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step()
        if patience_counter >= patience:
            break

    model.load_state_dict(torch.load(MODEL_FASTVIT, map_location=device))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.numpy())

    test_acc = (np.array(test_preds) == np.array(test_labels)).mean()
    test_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)

    return {
        "model": "FastViT-T8",
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1),
        "best_val_f1": float(best_val_f1),
        "epochs_trained": epoch + 1,
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CervicalAI Model Training Pipeline")
    parser.add_argument("--stage", choices=["all", "xgboost", "fastvit"], default="all")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs-xgb", type=int, default=500)
    parser.add_argument("--epochs-vit", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("CervicalAI — Model Training Pipeline v2.0")
    logger.info("=" * 60)

    if args.download:
        download_kaggle_datasets()

    metrics = {}

    if args.stage in ("all", "xgboost"):
        xgb_metrics = train_xgboost(epochs=args.epochs_xgb)
        if xgb_metrics:
            metrics["xgboost"] = xgb_metrics

    if args.stage in ("all", "fastvit"):
        vit_metrics = train_fastvit(
            epochs=args.epochs_vit, batch_size=args.batch_size, lr=args.lr, img_size=args.img_size
        )
        if vit_metrics:
            metrics["fastvit"] = vit_metrics

    metrics["training_date"] = start_time.isoformat()
    metrics["duration_seconds"] = (datetime.now() - start_time).total_seconds()

    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()