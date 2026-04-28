from __future__ import annotations

import argparse
from pathlib import Path

from tensorflow import keras

from config import (
    BATCH_SIZE,
    DATASET_SUMMARY_PATH,
    DROPOUT,
    EMB_DIM,
    EPOCHS,
    HISTORY_CSV_PATH,
    HISTORY_JSON_PATH,
    LEARNING_RATE,
    METRICS_JSON_PATH,
    MODEL_PATH,
    MODEL_SUMMARY_PATH,
    NUM_HEADS,
    NUM_LAYERS,
    PATIENCE,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    REPORTS_JSON_PATH,
    TEST_CM_PATH,
    THRESHOLD,
    TRAINING_PLOT_PATH,
    VALIDATION_CM_PATH,
)
from model import build_saint_like_model, get_model_summary_text
from preprocessing import (
    drop_columns,
    fit_and_transform_preprocessing,
    load_dataset,
    save_preprocessor,
    split_data,
    summarize_dataset,
)
from utils import (
    evaluate_metrics,
    get_predictions,
    get_runtime_device,
    make_model_inputs,
    plot_training_curve,
    report_dict,
    save_confusion_matrix,
    save_json,
    save_training_history,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Entrenamiento del modelo SAINT-like con Keras para promoción de empleados.')
    parser.add_argument('--data-path', type=str, required=True, help='Ruta al CSV del dataset.')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--emb-dim', type=int, default=EMB_DIM)
    parser.add_argument('--num-heads', type=int, default=NUM_HEADS)
    parser.add_argument('--num-layers', type=int, default=NUM_LAYERS)
    parser.add_argument('--dropout', type=float, default=DROPOUT)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(RANDOM_STATE)

    device = get_runtime_device()
    print(f'Dispositivo detectado: {device}', flush=True)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f'No se encontró el dataset en: {data_path}')

    df = load_dataset(str(data_path))
    print(f'Dataset cargado: {df.shape}', flush=True)
    save_json(summarize_dataset(df), DATASET_SUMMARY_PATH)

    df = drop_columns(df)
    print(f'Dataset después de drop_columns: {df.shape}', flush=True)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    print('Split completado', flush=True)
    print(f'X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}', flush=True)

    artifacts = fit_and_transform_preprocessing(X_train, X_val, X_test, y_train, y_val, y_test)
    save_preprocessor(artifacts, PREPROCESSOR_PATH)
    print('Preprocesamiento completado', flush=True)
    print(f"Numéricas: {len(artifacts['numeric_features'])}", flush=True)
    print(f"Categóricas: {len(artifacts['categorical_features'])}", flush=True)

    y_train_np = artifacts['y_train_np']
    neg = float((y_train_np == 0).sum())
    pos = float((y_train_np == 1).sum())
    positive_weight = neg / max(pos, 1.0)
    class_weight = {0: 1.0, 1: positive_weight}
    print(f'class_weight: {class_weight}', flush=True)

    model = build_saint_like_model(
        cat_dims=artifacts['cat_dims'],
        num_numeric=len(artifacts['numeric_features']),
        emb_dim=args.emb_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='binary_crossentropy',
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='roc_auc'),
            keras.metrics.AUC(name='pr_auc', curve='PR'),
        ],
    )

    summary_text = get_model_summary_text(model)
    with open(MODEL_SUMMARY_PATH, 'w', encoding='utf-8') as f:
        f.write(summary_text)

    print('\nResumen del modelo:', flush=True)
    print(summary_text, flush=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        )
    ]

    history = model.fit(
        x=make_model_inputs(artifacts['X_train_cat'], artifacts['X_train_num']),
        y=artifacts['y_train_np'],
        validation_data=(
            make_model_inputs(artifacts['X_val_cat'], artifacts['X_val_num']),
            artifacts['y_val_np'],
        ),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(MODEL_PATH)

    history_df = save_training_history(history, HISTORY_CSV_PATH, HISTORY_JSON_PATH)
    plot_training_curve(history_df, TRAINING_PLOT_PATH)

    y_train_true, y_train_pred, y_train_proba = get_predictions(
        model,
        artifacts['X_train_cat'],
        artifacts['X_train_num'],
        artifacts['y_train_np'],
        batch_size=args.batch_size,
        threshold=THRESHOLD,
    )
    y_val_true, y_val_pred, y_val_proba = get_predictions(
        model,
        artifacts['X_val_cat'],
        artifacts['X_val_num'],
        artifacts['y_val_np'],
        batch_size=args.batch_size,
        threshold=THRESHOLD,
    )
    y_test_true, y_test_pred, y_test_proba = get_predictions(
        model,
        artifacts['X_test_cat'],
        artifacts['X_test_num'],
        artifacts['y_test_np'],
        batch_size=args.batch_size,
        threshold=THRESHOLD,
    )

    metrics = {
        'train': evaluate_metrics(y_train_true, y_train_pred, y_train_proba),
        'validation': evaluate_metrics(y_val_true, y_val_pred, y_val_proba),
        'test': evaluate_metrics(y_test_true, y_test_pred, y_test_proba),
    }
    save_json(metrics, METRICS_JSON_PATH)

    reports = {
        'validation': report_dict(y_val_true, y_val_pred),
        'test': report_dict(y_test_true, y_test_pred),
    }
    save_json(reports, REPORTS_JSON_PATH)

    save_confusion_matrix(y_val_true, y_val_pred, 'Matriz de confusión - Validation', VALIDATION_CM_PATH)
    save_confusion_matrix(y_test_true, y_test_pred, 'Matriz de confusión - Test', TEST_CM_PATH)

    print('\nEntrenamiento finalizado.', flush=True)
    print(f'Modelo guardado en: {MODEL_PATH}', flush=True)
    print(f'Preprocesamiento guardado en: {PREPROCESSOR_PATH}', flush=True)
    print(f'Resumen del modelo guardado en: {MODEL_SUMMARY_PATH}', flush=True)
    print(f'Curva de entrenamiento guardada en: {TRAINING_PLOT_PATH}', flush=True)
    print(f'Métricas guardadas en: {METRICS_JSON_PATH}', flush=True)


if __name__ == '__main__':
    main()