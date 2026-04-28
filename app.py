from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import gradio as gr
import pandas as pd

from config import (
    BATCH_PREDICTIONS_PATH,
    DATA_DIR,
    METRICS_JSON_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    TARGET_COLUMN,
)
from inference import PromotionPredictor
from preprocessing import drop_columns, load_dataset, load_preprocessor


def build_predictor() -> PromotionPredictor | None:
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        return None
    return PromotionPredictor(MODEL_PATH, PREPROCESSOR_PATH)


def load_metrics() -> Dict:
    if not METRICS_JSON_PATH.exists():
        return {}
    with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


predictor = build_predictor()
preprocessor = load_preprocessor(PREPROCESSOR_PATH) if PREPROCESSOR_PATH.exists() else None
metrics_data = load_metrics()

DATASET_PATH = DATA_DIR / "employee_promotion_prediction.csv"


def load_source_dataset() -> pd.DataFrame | None:
    if not DATASET_PATH.exists() or preprocessor is None:
        return None

    df = load_dataset(str(DATASET_PATH))
    df = drop_columns(df)

    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])

    feature_order = preprocessor["feature_order"]
    valid_cols = [c for c in feature_order if c in df.columns]
    df = df[valid_cols].copy()
    return df


source_df = load_source_dataset()


def default_values_for_ui() -> List:
    if preprocessor is None:
        return []

    values = []

    for _ in preprocessor["numeric_features"]:
        values.append(0)

    for col in preprocessor["categorical_features"]:
        classes = [c for c in preprocessor["label_encoders"][col].classes_.tolist() if c != "__UNK__"]
        values.append(classes[0] if classes else None)

    return values


def random_employee_values() -> List:
    if preprocessor is None:
        return []

    if source_df is None or source_df.empty:
        return default_values_for_ui()

    sample_row = source_df.sample(1).iloc[0]
    values = []

    for col in preprocessor["numeric_features"]:
        value = sample_row.get(col, 0)
        if pd.isna(value):
            value = 0
        values.append(float(value))

    for col in preprocessor["categorical_features"]:
        value = sample_row.get(col, None)
        if pd.isna(value):
            value = None
        else:
            value = str(value)

        classes = [c for c in preprocessor["label_encoders"][col].classes_.tolist() if c != "__UNK__"]
        if value not in classes:
            value = classes[0] if classes else None

        values.append(value)

    return values


def clear_employee_values() -> List:
    return default_values_for_ui()


def predict_single(*values):
    if predictor is None or preprocessor is None:
        raise gr.Error("No se encontró un modelo entrenado. Ejecuta train.py primero.")

    numeric_features: List[str] = preprocessor["numeric_features"]
    categorical_features: List[str] = preprocessor["categorical_features"]

    input_data = {}
    idx = 0

    for col in numeric_features:
        input_data[col] = values[idx]
        idx += 1

    for col in categorical_features:
        input_data[col] = values[idx]
        idx += 1

    pred, proba = predictor.predict_one(input_data)
    label = "Promovido" if pred == 1 else "No promovido"

    details = {
        "predicted_class": pred,
        "predicted_label": label,
        "probability_promoted": round(proba, 6),
        "threshold": predictor.threshold,
    }
    return label, details


def csv_format_view(option: str):
    if preprocessor is None:
        return "No hay preprocesador cargado.", pd.DataFrame()

    columns = preprocessor["feature_order"]

    if option == "Solo columnas requeridas":
        info = (
            "El CSV debe contener estas columnas.\n\n"
            "- No necesitas incluir `promoted`.\n"
            "- El orden no importa, pero los nombres sí.\n"
            "- Si mandas columnas extra, el sistema las ignora."
        )
        preview_df = pd.DataFrame(columns=columns)
        return info, preview_df

    if option == "Ejemplo vacío":
        row = {}
        for col in preprocessor["numeric_features"]:
            row[col] = 0
        for col in preprocessor["categorical_features"]:
            classes = [c for c in preprocessor["label_encoders"][col].classes_.tolist() if c != "__UNK__"]
            row[col] = classes[0] if classes else ""
        preview_df = pd.DataFrame([row])[columns]
        info = "Este es un ejemplo mínimo con una fila."
        return info, preview_df

    if option == "Ejemplo con fila real":
        if source_df is not None and not source_df.empty:
            preview_df = source_df.sample(1).reset_index(drop=True)[columns]
            info = "Este ejemplo usa una fila real del dataset de entrenamiento."
            return info, preview_df
        else:
            info = "No se encontró el dataset original para mostrar una fila real."
            preview_df = pd.DataFrame(columns=columns)
            return info, preview_df

    return "Selecciona una opción.", pd.DataFrame(columns=columns)


def predict_batch(file_obj):
    if predictor is None:
        raise gr.Error("No se encontró un modelo entrenado. Ejecuta train.py primero.")
    if file_obj is None:
        raise gr.Error("Sube un archivo CSV para continuar.")

    df = pd.read_csv(file_obj.name)
    result = predictor.predict_dataframe(df)
    result.to_csv(BATCH_PREDICTIONS_PATH, index=False)
    return result, str(BATCH_PREDICTIONS_PATH)


def metric_card(title: str, value, description: str) -> str:
    value_text = "N/A" if value is None else f"{value:.4f}"
    return f"""
    <div style="border:1px solid #ddd; border-radius:12px; padding:14px; margin-bottom:10px;">
        <h4 style="margin:0 0 8px 0;">{title}</h4>
        <div style="font-size:24px; font-weight:bold; margin-bottom:8px;">{value_text}</div>
        <div style="font-size:14px; line-height:1.4;">{description}</div>
    </div>
    """


def build_metrics_html(metrics: Dict, split_name: str) -> str:
    if not metrics or split_name not in metrics:
        return "<p>No hay métricas disponibles todavía.</p>"

    split_metrics = metrics[split_name]

    return (
        metric_card(
            "Accuracy",
            split_metrics.get("accuracy"),
            "Mide la proporción total de predicciones correctas; es útil como referencia general, pero debe interpretarse con cuidado en problemas desbalanceados."
        )
        + metric_card(
            "Precision",
            split_metrics.get("precision"),
            "Indica qué proporción de los casos predichos como promovidos realmente correspondían a empleados promovidos."
        )
        + metric_card(
            "Recall",
            split_metrics.get("recall"),
            "Mide qué proporción de los empleados realmente promovidos fue detectada correctamente por el modelo."
        )
        + metric_card(
            "F1-score",
            split_metrics.get("f1_score"),
            "Resume el equilibrio entre precision y recall en una sola métrica."
        )
    )


with gr.Blocks(title="Employee Promotion Prediction") as demo:
    gr.Markdown(
        "# Employee Promotion Prediction\n"
        "Usa una de estas dos opciones: crear un empleado manualmente o subir un CSV."
    )

    if metrics_data:
        gr.Markdown("## Métricas del modelo")

        with gr.Tabs():
            with gr.Tab("Train"):
                gr.HTML(build_metrics_html(metrics_data, "train"))
            with gr.Tab("Validation"):
                gr.HTML(build_metrics_html(metrics_data, "validation"))
            with gr.Tab("Test"):
                gr.HTML(build_metrics_html(metrics_data, "test"))

    if predictor is None or preprocessor is None:
        gr.Markdown(
            "## Falta entrenar el modelo\n"
            "Primero ejecuta:\n\n"
            "`python train.py --data-path data/employee_promotion_prediction.csv`"
        )
    else:
        employee_inputs = []

        with gr.Tab("Crear empleado"):
            gr.Markdown("Puedes capturar un empleado desde cero o cargar uno aleatorio del dataset.")

            with gr.Column():
                if preprocessor["numeric_features"]:
                    gr.Markdown("### Variables numéricas")
                    for col in preprocessor["numeric_features"]:
                        comp = gr.Number(label=col, value=0)
                        employee_inputs.append(comp)

                if preprocessor["categorical_features"]:
                    gr.Markdown("### Variables categóricas")
                    for col in preprocessor["categorical_features"]:
                        classes = [c for c in preprocessor["label_encoders"][col].classes_.tolist() if c != "__UNK__"]
                        default_value = classes[0] if classes else None
                        comp = gr.Dropdown(
                            choices=classes,
                            value=default_value,
                            label=col,
                        )
                        employee_inputs.append(comp)

            with gr.Row():
                random_btn = gr.Button("Generar empleado random")
                clear_btn = gr.Button("Limpiar")
                predict_btn = gr.Button("Predecir", variant="primary")

            prediction_label = gr.Textbox(label="Resultado")
            prediction_json = gr.JSON(label="Detalle de la predicción")

            random_btn.click(
                fn=random_employee_values,
                inputs=[],
                outputs=employee_inputs,
            )

            clear_btn.click(
                fn=clear_employee_values,
                inputs=[],
                outputs=employee_inputs,
            )

            predict_btn.click(
                fn=predict_single,
                inputs=employee_inputs,
                outputs=[prediction_label, prediction_json],
            )

        with gr.Tab("Subir CSV"):
            gr.Markdown("Sube un CSV para predecir varios empleados a la vez.")

            csv_view_dropdown = gr.Dropdown(
                choices=[
                    "Solo columnas requeridas",
                    "Ejemplo vacío",
                    "Ejemplo con fila real",
                ],
                value="Solo columnas requeridas",
                label="Cómo debe verse el CSV",
            )

            csv_info = gr.Markdown()
            csv_preview = gr.Dataframe(label="Vista previa del formato")

            csv_view_dropdown.change(
                fn=csv_format_view,
                inputs=csv_view_dropdown,
                outputs=[csv_info, csv_preview],
            )

            file_input = gr.File(label="Archivo CSV", file_types=[".csv"])
            batch_btn = gr.Button("Procesar archivo", variant="primary")
            batch_df = gr.Dataframe(label="Resultados")
            batch_file = gr.File(label="Descargar predicciones")

            batch_btn.click(
                fn=predict_batch,
                inputs=file_input,
                outputs=[batch_df, batch_file],
            )

            demo.load(
                fn=lambda: csv_format_view("Solo columnas requeridas"),
                inputs=[],
                outputs=[csv_info, csv_preview],
            )


if __name__ == "__main__":
    demo.launch()