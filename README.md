# Employee Promotion Prediction

**Objetivo:** Clasificación binaria - Predecir si un empleado será promovido.

## Leer la documentación

La documentación del proyecto se encuentra en [documentation.md](./documentation.md)

## Instalar y Ejecutar

Antes que nada para poder correrlo necesitas estar en un entorno virtual.
```bash
.\.venv\Scripts\Activate.ps1    
```


### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Entrenar el modelo
```bash
python train.py
```
Esto genera el modelo entrenado en `outputs/model.keras`

### 3. Usar la app web
```bash
python app.py
```
Abre la interfaz Gradio en tu navegador para hacer predicciones.

### 4. Predicción por lotes
```bash
python inference.py
```
Procesa múltiples registros desde un archivo CSV.

---

## Archivos del Proyecto

| Archivo | Descripción |
|---------|-------------|
| [`train.py`](./train.py) | Entrena el modelo con los datos procesados |
| [`app.py`](./app.py) | Interfaz web (Gradio) para predicciones individuales |
| [`inference.py`](./inference.py) | Predicción en lote desde CSV |
| [`preprocessing.py`](./preprocessing.py) | Limpieza y transformación de datos |
| [`config.py`](./config.py) | Configuración: rutas, hiperparámetros, columnas |
| [`model.py`](./model.py) | Estructura y arquitectura del modelo |
| [`dataset.py`](./dataset.py) | Carga y gestión del dataset |
| [`utils.py`](./utils.py) | Funciones auxiliares |
| [`requirements.txt`](./requirements.txt) | Dependencias del proyecto |
| [`Employee_Promotion_Prediction.ipynb`](./Employee_Promotion_Prediction.ipynb) | Análisis exploratorio de datos |
| [`documentation.md`](./documentation.md) | Documentación detallada |
| [`data/`](./data/) | Dataset original en CSV |
| [`outputs/`](./outputs/) | Modelo entrenado y resultados |
---

## Dataset

Dataset: [`data/employee_promotion_prediction.csv`](./data/employee_promotion_prediction.csv)

Dataset original disponible en Kaggle: [Employee Promtion Prediction Dataset](https://www.kaggle.com/datasets/rohit8527kmr7518/employee-promtion-prediction)

