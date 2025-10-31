import json
import os
from dataclasses import dataclass
from typing import List, Optional, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import ARTIFACTS_FOLDER

os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)


@dataclass
class PreprocessConfig:
    target: Optional[str]
    impute_strategy: str = 'median'  # 'mean'|'median'|'most_frequent'
    scale_numeric: bool = True
    one_hot_categorical: bool = True
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42


def build_pipeline(df: pd.DataFrame, cfg: PreprocessConfig) -> ColumnTransformer:
    features = df.drop(columns=[cfg.target]) if cfg.target and cfg.target in df.columns else df.copy()
    num_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in features.columns if c not in num_cols]

    transformers = []
    num_steps = [('imputer', SimpleImputer(strategy=cfg.impute_strategy))]
    if cfg.scale_numeric:
        num_steps.append(('scaler', StandardScaler()))
    if num_cols:
        transformers.append(('num', Pipeline(num_steps), num_cols))

    cat_steps = [('imputer', SimpleImputer(strategy='most_frequent'))]
    if cfg.one_hot_categorical and cat_cols:
        cat_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
    if cat_cols:
        transformers.append(('cat', Pipeline(cat_steps), cat_cols))

    return ColumnTransformer(transformers=transformers, remainder='drop')


# Local lightweight Pipeline wrapper to avoid importing sklearn.pipeline at top-level until needed
from sklearn.pipeline import Pipeline


def apply_and_save(df: pd.DataFrame, cfg: PreprocessConfig) -> Dict:
    """Build pipeline, split data, transform X, and save artifacts to disk."""
    meta = {
        'target': cfg.target,
        'impute_strategy': cfg.impute_strategy,
        'scale_numeric': cfg.scale_numeric,
        'one_hot_categorical': cfg.one_hot_categorical,
        'test_size': cfg.test_size,
        'val_size': cfg.val_size,
        'random_state': cfg.random_state,
    }

    if cfg.target and cfg.target in df.columns:
        y = df[cfg.target]
        X = df.drop(columns=[cfg.target])
    else:
        y = None
        X = df
    pipe = build_pipeline(df, cfg)

    # Split train/val/test deterministically
    if y is not None:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(cfg.test_size + cfg.val_size), random_state=cfg.random_state, stratify=None
        )
        rel_val = cfg.val_size / (cfg.test_size + cfg.val_size) if (cfg.test_size + cfg.val_size) > 0 else 0
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(cfg.test_size / (cfg.test_size + cfg.val_size)) if (cfg.test_size + cfg.val_size) else 0,
            random_state=cfg.random_state, stratify=None
        )
    else:
        # no target – just split features
        X_train, X_temp = train_test_split(X, test_size=(cfg.test_size + cfg.val_size), random_state=cfg.random_state)
        X_val, X_test = train_test_split(X_temp, test_size=(cfg.test_size / (cfg.test_size + cfg.val_size)) if (cfg.test_size + cfg.val_size) else 0,
                                         random_state=cfg.random_state)
        y_train = y_val = y_test = None

    pipe_fit = pipe.fit(X_train)

    # Transform and assemble back into DataFrames
    def tf_to_df(X_part, name):
        if X_part is None:
            return None
        arr = pipe_fit.transform(X_part)
        if isinstance(arr, np.ndarray):
            cols = [f"f_{i}" for i in range(arr.shape[1])]
            return pd.DataFrame(arr, columns=cols, index=X_part.index)
        else:
            # fallback
            arr = np.asarray(arr)
            cols = [f"f_{i}" for i in range(arr.shape[1])]
            return pd.DataFrame(arr, columns=cols, index=X_part.index)

    X_train_t = tf_to_df(X_train, 'train')
    X_val_t   = tf_to_df(X_val, 'val')
    X_test_t  = tf_to_df(X_test, 'test')

    # Persist artifacts
    os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)
    joblib.dump(pipe_fit, os.path.join(ARTIFACTS_FOLDER, 'pipeline.joblib'))

    out = {'train': X_train_t, 'val': X_val_t, 'test': X_test_t}
    for split, Xp in out.items():
        if Xp is not None:
            Xp.to_csv(os.path.join(ARTIFACTS_FOLDER, f'X_{split}.csv'), index=False)
    if y_train is not None:
        pd.Series(y_train).to_csv(os.path.join(ARTIFACTS_FOLDER, 'y_train.csv'), index=False)
        pd.Series(y_val).to_csv(os.path.join(ARTIFACTS_FOLDER, 'y_val.csv'), index=False)
        pd.Series(y_test).to_csv(os.path.join(ARTIFACTS_FOLDER, 'y_test.csv'), index=False)

    with open(os.path.join(ARTIFACTS_FOLDER, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    return {
        'n_train': 0 if X_train_t is None else len(X_train_t),
        'n_val': 0 if X_val_t is None else len(X_val_t),
        'n_test': 0 if X_test_t is None else len(X_test_t),
        'has_target': y is not None,
        'artifacts': ARTIFACTS_FOLDER,
    }