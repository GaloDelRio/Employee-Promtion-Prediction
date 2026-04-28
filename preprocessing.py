from __future__ import annotations

import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import (
    COLUMNS_TO_DROP,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    VAL_SIZE_WITHIN_TEMP,
)


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{TARGET_COLUMN}' en el dataset.")
    return df


def summarize_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        'rows': int(df.shape[0]),
        'columns': int(df.shape[1]),
        'column_names': df.columns.tolist(),
        'null_values_per_column': {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
        'target_distribution_count': {str(k): int(v) for k, v in df[TARGET_COLUMN].value_counts().to_dict().items()},
        'target_distribution_percent': {
            str(k): float(v) for k, v in (df[TARGET_COLUMN].value_counts(normalize=True) * 100).to_dict().items()
        },
    }


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in COLUMNS_TO_DROP if c in df.columns]
    return df.drop(columns=cols)


def split_data(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=VAL_SIZE_WITHIN_TEMP,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def identify_feature_types(X_train: pd.DataFrame) -> Tuple[List[str], List[str]]:
    numeric_features = X_train.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return numeric_features, categorical_features


def _safe_fit_label_encoder(series_list: List[pd.Series]) -> LabelEncoder:
    le = LabelEncoder()
    all_values = pd.concat([s.astype(str) for s in series_list], axis=0)
    le.fit(pd.concat([all_values, pd.Series(['__UNK__'])], axis=0))
    return le


def _safe_transform_label_encoder(le: LabelEncoder, series: pd.Series) -> np.ndarray:
    known = set(le.classes_)
    values = series.astype(str).apply(lambda x: x if x in known else '__UNK__')
    return le.transform(values)


def fit_and_transform_preprocessing(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Any]:
    numeric_features, categorical_features = identify_feature_types(X_train)

    X_train_proc = X_train.copy()
    X_val_proc = X_val.copy()
    X_test_proc = X_test.copy()

    num_imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    if numeric_features:
        X_train_proc[numeric_features] = num_imputer.fit_transform(X_train_proc[numeric_features])
        X_val_proc[numeric_features] = num_imputer.transform(X_val_proc[numeric_features])
        X_test_proc[numeric_features] = num_imputer.transform(X_test_proc[numeric_features])

        X_train_proc[numeric_features] = scaler.fit_transform(X_train_proc[numeric_features])
        X_val_proc[numeric_features] = scaler.transform(X_val_proc[numeric_features])
        X_test_proc[numeric_features] = scaler.transform(X_test_proc[numeric_features])

    cat_imputer = SimpleImputer(strategy='most_frequent')
    label_encoders: Dict[str, LabelEncoder] = {}
    cat_dims: List[int] = []

    if categorical_features:
        X_train_proc[categorical_features] = cat_imputer.fit_transform(X_train_proc[categorical_features])
        X_val_proc[categorical_features] = cat_imputer.transform(X_val_proc[categorical_features])
        X_test_proc[categorical_features] = cat_imputer.transform(X_test_proc[categorical_features])

        for col in categorical_features:
            le = _safe_fit_label_encoder([X_train_proc[col], X_val_proc[col], X_test_proc[col]])
            X_train_proc[col] = _safe_transform_label_encoder(le, X_train_proc[col])
            X_val_proc[col] = _safe_transform_label_encoder(le, X_val_proc[col])
            X_test_proc[col] = _safe_transform_label_encoder(le, X_test_proc[col])
            label_encoders[col] = le
            cat_dims.append(len(le.classes_))

    if categorical_features:
        X_train_cat = X_train_proc[categorical_features].to_numpy(dtype=np.int32)
        X_val_cat = X_val_proc[categorical_features].to_numpy(dtype=np.int32)
        X_test_cat = X_test_proc[categorical_features].to_numpy(dtype=np.int32)
    else:
        X_train_cat = np.empty((len(X_train_proc), 0), dtype=np.int32)
        X_val_cat = np.empty((len(X_val_proc), 0), dtype=np.int32)
        X_test_cat = np.empty((len(X_test_proc), 0), dtype=np.int32)

    if numeric_features:
        X_train_num = X_train_proc[numeric_features].to_numpy(dtype=np.float32)
        X_val_num = X_val_proc[numeric_features].to_numpy(dtype=np.float32)
        X_test_num = X_test_proc[numeric_features].to_numpy(dtype=np.float32)
    else:
        X_train_num = np.empty((len(X_train_proc), 0), dtype=np.float32)
        X_val_num = np.empty((len(X_val_proc), 0), dtype=np.float32)
        X_test_num = np.empty((len(X_test_proc), 0), dtype=np.float32)

    artifacts = {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'feature_order': numeric_features + categorical_features,
        'num_imputer': num_imputer,
        'scaler': scaler,
        'cat_imputer': cat_imputer if categorical_features else None,
        'label_encoders': label_encoders,
        'cat_dims': cat_dims,
        'columns_to_drop': COLUMNS_TO_DROP,
        'target_column': TARGET_COLUMN,
        'X_train_cat': X_train_cat,
        'X_val_cat': X_val_cat,
        'X_test_cat': X_test_cat,
        'X_train_num': X_train_num,
        'X_val_num': X_val_num,
        'X_test_num': X_test_num,
        'y_train_np': y_train.to_numpy(dtype=np.float32),
        'y_val_np': y_val.to_numpy(dtype=np.float32),
        'y_test_np': y_test.to_numpy(dtype=np.float32),
    }
    return artifacts


def save_preprocessor(artifacts: Dict[str, Any], path=PREPROCESSOR_PATH) -> None:
    serializable = {
        'numeric_features': artifacts['numeric_features'],
        'categorical_features': artifacts['categorical_features'],
        'feature_order': artifacts['feature_order'],
        'num_imputer': artifacts['num_imputer'],
        'scaler': artifacts['scaler'],
        'cat_imputer': artifacts['cat_imputer'],
        'label_encoders': artifacts['label_encoders'],
        'cat_dims': artifacts['cat_dims'],
        'columns_to_drop': artifacts['columns_to_drop'],
        'target_column': artifacts['target_column'],
    }
    with open(path, 'wb') as f:
        pickle.dump(serializable, f)


def load_preprocessor(path=PREPROCESSOR_PATH) -> Dict[str, Any]:
    with open(path, 'rb') as f:
        return pickle.load(f)


def transform_new_data(df: pd.DataFrame, preprocessor: Dict[str, Any]):
    numeric_features = preprocessor['numeric_features']
    categorical_features = preprocessor['categorical_features']
    feature_order = preprocessor['feature_order']

    work_df = df.copy()

    for col in feature_order:
        if col not in work_df.columns:
            work_df[col] = np.nan

    work_df = work_df[feature_order]

    if numeric_features:
        for col in numeric_features:
            work_df[col] = pd.to_numeric(work_df[col], errors='coerce')
        work_df[numeric_features] = preprocessor['num_imputer'].transform(work_df[numeric_features])
        work_df[numeric_features] = preprocessor['scaler'].transform(work_df[numeric_features])
        x_num = work_df[numeric_features].to_numpy(dtype=np.float32)
    else:
        x_num = np.empty((len(work_df), 0), dtype=np.float32)

    if categorical_features:
        cat_block = work_df[categorical_features].copy()
        cat_block[categorical_features] = preprocessor['cat_imputer'].transform(cat_block[categorical_features])
        for col in categorical_features:
            le = preprocessor['label_encoders'][col]
            cat_block[col] = _safe_transform_label_encoder(le, cat_block[col])
        x_cat = cat_block[categorical_features].to_numpy(dtype=np.int32)
    else:
        x_cat = np.empty((len(work_df), 0), dtype=np.int32)

    return x_cat, x_num