# Reporte del dataset y preprocesamiento realizado

## Dataset utilizado

Para este proyecto se utilizó el dataset **Employee Promtion Prediction Dataset**, publicado en Kaggle por **Rohit Kumar**. Este conjunto de datos está orientado al análisis de variables relacionadas con el desempeño, compromiso y características laborales de empleados, con el objetivo de predecir si un colaborador será promovido o no. Este contiene alrededor de **100,000 registros**,con **43 variables** y una variable objetivo llamada **promoted**.

Además, la tasa de promoción aproximada de **10%**, lo que lo convierte en un problema de clasificación binaria con desbalance de clases.
En términos generales, el dataset reúne información que puede representar distintos factores asociados a una promoción laboral, como desempeño, historial reciente, participación en proyectos, entrenamiento, asistencia y otros indicadores internos. Debido a esto, resulta adecuado para un problema de *machine learning* supervisado enfocado en clasificación, donde la meta es aprender patrones que permitan distinguir entre empleados promovidos y no promovidos.

---

## Objetivo del análisis

Hasta este punto, el trabajo realizado se ha enfocado en seleccionar, ejecutar y evaluar un modelo para, posteriormente, analizar sus métricas con el propósito de medir su desempeño. Esto permitirá, en el futuro, realizar mejoras y comprender mejor el funcionamiento de estos elementos de machine learning.

---

## Librerías utilizadas

Para el procesamiento de datos se instalaron y utilizaron las siguientes librerías:

- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `scikit-learn`
- `imbalanced-learn`

---

## Carga del dataset

El archivo fue cargado desde Google Colab 

---

## Exploración inicial de los datos

Primero se realizó una revisión general del dataset para entender su estructura, verificar el tipo de variables y revisar si existían valores faltantes. En esta exploración se observó que no habían columnas vacías ni un problema importante de datos nulos, por lo que no fue necesario eliminar registros ni hacer una imputación previa a nivel global.

También se analizó la distribución de la variable objetivo **`promoted`** y verifique que el dataset estaba **desbalanceado**: la mayoría de los registros pertenecían a empleados no promovidos y una minoría a empleados promovidos (10%/90%). Esto era importante detectarlo desde el inicio, porque un modelo entrenado directamente con esta distribución podría aprender a favorecer la clase mayoritaria y fallar al identificar correctamente los casos de promoción.

---

## Análisis de correlación y selección de variables

Después de la exploración inicial, se construyó una **matriz de correlación** con las variables numéricas para entender qué atributos tenían relación con la variable objetivo y cuáles aportaban muy poca información.


<img width="1648" height="1492" alt="image" src="https://github.com/user-attachments/assets/013560d0-b3a4-4f80-88fc-9e2304608d59" />



A partir de este análisis se identifice que varias columnas tenían una correlación prácticamente nula con `promoted`, por lo que era razonable considerar que su aporte al modelo sería muy bajo. Entre ellas estaban variables como:

- `team_size`
- `remote_work_ratio`
- `deadline_adherence_rate`
- `cross_department_projects`
- `mentoring_sessions`
- `internal_mobility_score`
- `attendance_rate`
- `training_hours_last_year`
- `certifications_count`

Además, se detectó que algunas variables contenían información muy parecida entre sí, especialmente las relacionadas con desempeño. Por ejemplo, `performance_score`, `performance_last_year` y `performance_two_years_ago` mostraban una relación muy alta entre ellas. Esto indicaba **redundancia**, y mantenerlas juntas podía introducir multicolinealidad y ruido innecesario.

Con base en el análisis anterior, se eliminaron columnas que no aportaban significativamente al objetivo o que duplicaban información de otras variables. Esta reducción se hizo con tres propósitos principales:

1. Disminuir ruido en los datos.
2. Evitar multicolinealidad entre variables muy parecidas.
3. Simplificar el conjunto de entrada para facilitar el entrenamiento e interpretación del modelo más adelante.

Después de esta limpieza, el dataset quedó compuesto únicamente por las variables consideradas más útiles para explicar la promoción.

---

## Separación entre variables de entrada y variable objetivo

Una vez reducido el dataset, se separó la información en:

- **Variables de entrada (`X`)**: todas las características del empleado.
- **Variable objetivo (`y`)**: la columna `promoted`.

Esta separación era necesaria para preparar el problema de aprendizaje supervisado, donde el modelo recibirá las variables de entrada y tratará de aprender patrones para predecir la variable objetivo.

---

## División en conjunto de entrenamiento y prueba

El siguiente paso fue dividir los datos en dos grupos:

- Conjunto de entrenamiento para ajustar el modelo posteriormente.
- Conjunto de prueba para evaluar su desempeño con datos no vistos.

La división se hizo usando una proporción de **80% para entrenamiento** y **20% para prueba**, manteniendo la proporción original de la variable objetivo en ambos conjuntos. Esto último fue importante porque, al estar desbalanceadas las clases, era necesario conservar una distribución similar en entrenamiento y prueba para que la evaluación futura fuera representativa del problema real.

---

## Identificación de variables numéricas y categóricas

Antes de transformar los datos, se identificó qué variables eran **numéricas** y cuáles eran **categóricas**. Esto fue importante porque ambos tipos de datos necesitan un tratamiento distinto.

Las variables numéricas requieren normalmente escalamiento para que sus magnitudes sean comparables, mientras que las variables categóricas deben convertirse a una representación numérica que pueda ser entendida por los algoritmos de machine learning.

---

## Preprocesamiento de los datos

Después se construyó una etapa de preprocesamiento diferenciando el tratamiento según el tipo de variable.

### Variables numéricas

En las variables numéricas se aplicó:

- Imputación por mediana como medida preventiva en caso de encontrar valores faltantes.
- Estandarización para que todas las variables quedaran en una escala comparable.

Esto ayuda a que ciertas variables no dominen a otras solo por tener magnitudes más grandes y hace que muchos algoritmos funcionen de forma más estable.

### Variables categóricas

En las variables categóricas se aplicó:

- Imputación por moda como medida preventiva en caso de encontrar valores faltantes.
- Codificación One-Hot para transformar categorías en columnas binarias.

Esta transformación permite representar la información categórica en un formato numérico sin introducir un orden artificial entre categorías.

### Aplicación correcta del preprocesamiento

El preprocesamiento se ajustó únicamente con el conjunto de entrenamiento y después se aplicó al conjunto de prueba. Esto fue importante para evitar fuga de información, ya que el conjunto de prueba no debe influir en la construcción de las transformaciones.

---

## Balanceo de clases con SMOTE

Una vez preprocesados los datos, se aplicó **SMOTE** únicamente sobre el conjunto de entrenamiento. Esta técnica fue necesaria porque la variable `promoted` estaba desbalanceada, con aproximadamente un **90% de casos negativos** y un **10% de casos positivos**.

SMOTE genera nuevas instancias sintéticas de la clase minoritaria a partir de los ejemplos existentes. En este caso, su uso permitió equilibrar la cantidad de empleados promovidos y no promovidos dentro del conjunto de entrenamiento.

La razón de hacer esto fue mejorar la capacidad del futuro modelo para aprender patrones de la clase minoritaria. Si no se balancea el entrenamiento, el modelo podría inclinarse demasiado hacia la clase mayoritaria y dar buenos resultados aparentes en exactitud, pero un mal desempeño al detectar promociones reales.

Es importante señalar que SMOTE solo se aplicó al conjunto de entrenamiento, no al conjunto de prueba. Esto se hizo para que la evaluación futura refleje mejor el comportamiento del modelo ante datos reales y no alterados artificialmente.

---

## Elección de modelo primera iteración

Para esta etapa se seleccionó un modelo de Random Forest como primer modelo de clasificación. La elección se hizo porque este algoritmo suele dar buenos resultados en problemas de clasificación tabular, especialmente cuando existen relaciones no lineales entre variables y cuando se busca un modelo robusto frente al ruido.

Random Forest funciona mediante un conjunto de árboles de decisión entrenados sobre distintas muestras de los datos. La predicción final se obtiene combinando el resultado de todos esos árboles, lo que normalmente produce modelos más estables que un solo árbol individual.

<img width="1400" height="1000" alt="image" src="https://github.com/user-attachments/assets/92ce523d-43c5-48a5-9ae5-a5e80a764d93" />

---

## Estado actual del proyecto

En esta etapa todavía no se ha entrenado ningún modelo. Todo el trabajo realizado corresponde a la fase de preparación de datos, que es fundamental antes del modelado. Gracias a este procesamiento, el dataset ya quedó listo para la siguiente fase, donde se podrá entrenar y evaluar un modelo de clasificación con una base de datos más limpia, más consistente y mejor balanceada.

---

## Métricas de evaluación seleccionadas

Debido a que el problema presenta desbalance de clases, no era suficiente evaluar el modelo únicamente con accuracy. Si se utilizara solo esa métrica, podría parecer que el modelo funciona bien aunque en realidad falle al detectar correctamente a los empleados promovidos, que corresponden a la clase minoritaria.

1. Accuracy
- Mide la proporción total de predicciones correctas. Aunque es útil como referencia general, en este caso debe interpretarse con cuidado debido al desbalance entre clases.

3. Precision
- Indica qué proporción de los casos que el modelo predijo como promovidos realmente correspondían a empleados promovidos. Esta métrica ayuda a medir qué tan confiables son las predicciones positivas.

4. Recall
- Mide qué proporción de los empleados que realmente fueron promovidos fue detectada por el modelo. Esta métrica es especialmente importante cuando interesa identificar la mayor cantidad posible de promociones reales.

5. F1-score
- Es la media armónica entre precision y recall. Se utilizó porque resume el equilibrio entre ambas métricas, lo cual es útil cuando ninguna de las dos por sí sola describe completamente el desempeño.

Además de las métricas anteriores, se utilizó la matriz de confusión para analizar cuántos casos fueron clasificados correctamente y cuántos errores se cometieron en cada clase.

---

## Resultados obtenidos

El modelo fue evaluado en entrenamiento, validación y prueba. Los resultados obtenidos fueron los siguientes:

### Entrenamiento

- Accuracy: 0.9192
- Precision: 0.9146
- Recall: 0.9249
- F1-score: 0.9197


### Validación

- Accuracy: 0.8626
- Precision: 0.3741
- Recall: 0.5382
- F1-score: 0.4414


### Prueba

- Accuracy: 0.8620
- Precision: 0.3703
- Recall: 0.5199
- F1-score: 0.4325


Además, el reporte de clasificación mostró que el modelo tuvo un desempeño alto sobre la clase 0, correspondiente a los empleados no promovidos, pero un desempeño considerablemente menor sobre la clase 1, correspondiente a los promovidos.

###  Reporte de clasificación: Validación

| Clase | Precision | Recall | F1-score | Support |
|------|-----------:|-------:|---------:|--------:|
| 0 | 0.95 | 0.90 | 0.92 | 2684 |
| 1 | 0.37 | 0.54 | 0.44 | 301 |

### Reporte de clasificación: Prueba

| Clase | Precision | Recall | F1-score | Support |
|------|-----------:|-------:|---------:|--------:|
| 0 | 0.94 | 0.90 | 0.92 | 2684 |
| 1 | 0.37 | 0.52 | 0.43 | 302 |

## Interpretación de los resultados

Los resultados muestran que el modelo logró aprender patrones útiles del conjunto de entrenamiento, ya que en ese subconjunto alcanzó valores altos en todas las métricas. Sin embargo, al pasar a validación y prueba se observó una disminución importante en el desempeño, especialmente en precision, recall y F1-score de la clase positiva.

Esto sugiere que el modelo presenta cierto nivel de sobreajuste. En otras palabras, aprendió muy bien el conjunto de entrenamiento, pero no mantuvo el mismo nivel de generalización en datos nuevos.

Aun así, los resultados de validación y prueba fueron muy parecidos entre sí, lo cual es una señal positiva. Esto indica que el comportamiento del modelo fue estable fuera del entrenamiento y que la evaluación final es consistente.

También se observó que el valor de accuracy se mantuvo alrededor de 0.86 en validación y prueba. Aunque este resultado parece alto, no debe interpretarse como prueba suficiente de buen desempeño, ya que el conjunto de datos está desbalanceado y la mayoría de los registros pertenecen a la clase no promovida.

La parte más importante del análisis está en la clase positiva. El modelo logró identificar aproximadamente la mitad de los empleados promovidos reales, lo cual se refleja en un recall cercano a 0.52–0.54. Sin embargo, la precision de 0.37 indica que una parte considerable de los casos predichos como promovidos realmente no lo eran. Como consecuencia, el F1-score de la clase positiva quedó en un nivel intermedio, alrededor de 0.43–0.44.

En conjunto, esto significa que el modelo sí tiene capacidad para detectar promociones, pero todavía comete bastantes errores al hacerlo. Por lo tanto, puede considerarse una primera aproximación funcional, aunque aún hay espacio de mejora antes de tomarlo como modelo final.

---

## Implementación de modelo basado en un articulo del estado del arte

Para esta etapa del proyecto se seleccionó un modelo **SAINT-like** implementado en **Keras**, inspirado en la arquitectura **SAINT (Self-Attention and Intersample Attention Transformer)**, la cual fue propuesta para trabajar con datos tabulares. Se tomó como referencia este artículo [*Enhancing Employee Promotion Prediction with Hybrid Deep Learning and SAINT-based Transformers*](https://arxiv.org/pdf/2604.10337) , ya que el modelo que propone ,resultaba útil para entender cómo este tipo de arquitectura puede aplicarse a variables laborales y organizacionales sin depender únicamente de modelos clásicos de árboles.

---

## ¿Qué es este modelo?

El modelo SAINT-like es una red neuronal diseñada para trabajar con **datos tabulares**, es decir, datos organizados en filas y columnas como los de este dataset de empleados. A diferencia de una red densa tradicional, este enfoque trata cada variable como una unidad de información separada y después aprende cómo se relaciona con las demás.

En este proyecto, el modelo recibe variables **categóricas** y **numéricas** del empleado. Las variables categóricas se convierten en representaciones vectoriales mediante **embeddings**, mientras que las numéricas se transforman mediante capas densas para llevarlas a un formato compatible. Después, todas esas representaciones se combinan como si fueran una secuencia de tokens y pasan por un bloque de atención tipo Transformer.

Este modelo se eligió porque el problema planteado es de tipo **clasificación tabular** y contiene tanto variables numéricas como categóricas, por lo que resultaba interesante probar una arquitectura que pudiera capturar relaciones entre ambas de forma más flexible que un modelo tradicional.

Además, SAINT fue considerado como una opción relevante porque en la literatura reciente se presenta como una arquitectura enfocada precisamente en datos tabulares y en la generación de representaciones contextuales de las variables. Esto lo vuelve atractivo en escenarios donde no solo importa el valor individual de cada columna, sino también la relación entre varias de ellas.

- Sin embargo, el artículo tomado como referencia no fue usado como una receta exacta para replicar, sino como una guía conceptual. De hecho, ese trabajo reporta que los modelos basados en árboles mantuvieron un mejor desempeño que SAINT y que los enfoques híbridos evaluados. Por esta razón, en este proyecto el uso de SAINT-like se entiende principalmente como una alternativa experimental y de comparación, útil para analizar otra familia de modelos sobre el mismo problema.

---

## ¿Cómo funciona?

El funcionamiento general del modelo puede resumirse en cuatro pasos:

1. **Entrada de datos**  
   El modelo recibe por separado las variables categóricas y las variables numéricas del empleado. Esta separación es importante porque ambos tipos de variables requieren un tratamiento distinto antes de entrar a la red.

2. **Transformación de variables**  
   Cada variable categórica se convierte en un embedding, es decir, en un vector numérico aprendido por el modelo. Por su parte, cada variable numérica pasa por una proyección densa que la transforma al mismo tamaño que los embeddings. De esta manera, todas las columnas quedan representadas en un espacio numérico comparable y pueden ser tratadas como una secuencia de tokens.

3. **Bloque de atención**  
   Una vez construidos los tokens, estos pasan por un bloque tipo Transformer. En esta parte, el mecanismo de atención permite que cada variable "observe" a las demás y aprenda qué tan relevantes son entre sí para la predicción. Por ejemplo, el modelo puede aprender que el efecto de una variable como `performance_score` cambia dependiendo de otras como `department`, `salary` o `years_at_company`.  
   
   Esto es importante porque en problemas tabulares no siempre basta con analizar cada columna de forma aislada; muchas veces la información útil está en la combinación entre varias variables. El bloque de atención ayuda precisamente a capturar esas interacciones. Además, dentro de este bloque también se incluyen conexiones residuales, normalización y una pequeña red feedforward, lo que permite refinar la representación aprendida sin perder la información original.
   
   En esta parte del modelo también aparecen varios de los hiperparámetros más importantes:
   - **`EMB_DIM`**: define el tamaño de los embeddings y de los tokens internos del modelo.
   - **`NUM_HEADS`**: indica cuántas cabezas de atención se usan, es decir, cuántas formas distintas tiene el modelo de analizar relaciones entre variables al mismo tiempo.
   - **`NUM_LAYERS`**: define cuántos bloques Transformer se apilan.
   - **`DROPOUT`**: controla el nivel de regularización, apagando aleatoriamente parte de las conexiones durante entrenamiento para reducir sobreajuste.

4. **Clasificación final**  
   La salida del bloque de atención se aplana y se envía a un clasificador denso. Este clasificador resume toda la información aprendida y genera una probabilidad entre 0 y 1 mediante una función sigmoide. Esa probabilidad representa qué tan probable es que el empleado sea promovido.
   
   En esta etapa también influyen hiperparámetros relacionados con el entrenamiento:
   - **`BATCH_SIZE`**: determina cuántos ejemplos procesa el modelo por lote.
   - **`EPOCHS`**: indica el número máximo de vueltas al conjunto de entrenamiento.
   - **`LEARNING_RATE`**: controla qué tan grandes son los ajustes que hace el optimizador en cada actualización.
   - **`PATIENCE`**: define cuántas épocas sin mejora en validación se permiten antes de detener automáticamente el entrenamiento.
---

## Estructura general del proyecto

El proyecto se organizó en varios archivos para separar responsabilidades y hacer más claro el flujo completo, desde la preparación de datos hasta la inferencia con la interfaz.

`config.py`

Este archivo concentra la configuración general del proyecto. Aquí se definen rutas, nombres de archivos de salida, columna objetivo y los hiperparámetros principales del modelo, como `EMB_DIM`, `NUM_HEADS`, `NUM_LAYERS`, `DROPOUT`, `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE` y `PATIENCE`.

- definición de rutas de trabajo (`DATA_DIR`, `OUTPUT_DIR`)
- columna objetivo (`TARGET_COLUMN`)
- columnas que se eliminan (`COLUMNS_TO_DROP`)
- hiperparámetros del modelo
- rutas donde se guardan modelo, métricas, historial y gráficas

---

`preprocessing.py`

Este archivo se encarga de cargar el dataset, limpiarlo, separar entrenamiento, validación y prueba, identificar variables numéricas y categóricas, y aplicar las transformaciones necesarias antes de entrenar o predecir.

- `load_dataset()`: carga el archivo CSV
- `summarize_dataset()`: genera un resumen general del dataset
- `drop_columns()`: elimina columnas que no se usarán
- `split_data()`: separa los datos en entrenamiento, validación y prueba
- `identify_feature_types()`: detecta cuáles columnas son numéricas y cuáles categóricas
- `fit_and_transform_preprocessing()`: ajusta imputadores, escalado y codificación, y transforma los datos
- `save_preprocessor()` y `load_preprocessor()`: guardan y recuperan el preprocesamiento
- `transform_new_data()`: transforma nuevos datos para inferencia usando el mismo preprocesamiento del entrenamiento

---

`model.py`

Aquí se define la arquitectura del modelo SAINT-like en Keras. Es el archivo donde se construye la red neuronal.

- `SliceColumn`: capa personalizada que extrae una columna específica de la entrada
- `transformer_block()`: define el bloque de atención tipo Transformer
- `build_saint_like_model()`: arma el modelo completo con entradas categóricas, numéricas, bloque Transformer y clasificador final
- `get_model_summary_text()`: guarda el resumen del modelo como texto

---

`utils.py`

Este archivo contiene funciones auxiliares para entrenamiento, evaluación, guardado de resultados y generación de gráficas.

- `seed_everything()`: fija semillas para reproducibilidad
- `get_runtime_device()`: detecta si se usa CPU o GPU
- `make_model_inputs()`: acomoda las entradas en el formato que espera Keras
- `get_predictions()`: genera probabilidades y clases predichas
- `evaluate_metrics()`: calcula accuracy, precision, recall, F1, ROC-AUC y PR-AUC
- `report_dict()`: genera el classification report
- `save_training_history()`: guarda el historial del entrenamiento
- `plot_training_curve()`: genera la gráfica de pérdida
- `save_confusion_matrix()`: guarda la matriz de confusión como imagen
- `save_json()`: guarda resultados en archivos JSON

---

`train.py`

Este archivo controla el proceso de entrenamiento completo. Toma el dataset, aplica preprocesamiento, construye el modelo, lo entrena, evalúa resultados y guarda todos los artefactos.

- `parse_args()`: permite cambiar hiperparámetros desde la terminal
- `main()`: ejecuta todo el flujo de entrenamiento
- carga del dataset y limpieza inicial
- preprocesamiento y guardado del preprocesador
- cálculo de `class_weight` para manejar el desbalance
- compilación del modelo con Adam y binary crossentropy
- entrenamiento con `EarlyStopping`
- guardado de modelo, métricas, reportes y gráficas

---

`inference.py`

Este archivo sirve para usar el modelo ya entrenado sobre nuevos datos. No entrena nada, solo carga el modelo y genera predicciones.

- `PromotionPredictor.__init__()`: carga el modelo y el preprocesador
- `predict_dataframe()`: predice varias filas y devuelve probabilidad y clase
- `predict_one()`: predice un solo registro a partir de un diccionario

---

`app.py`

Este archivo crea la interfaz con Gradio para usar el modelo de forma visual.

**Partes importantes:**
- `build_predictor()`: inicializa el predictor si ya existe modelo entrenado
- `load_source_dataset()`: carga el dataset base para generar empleados aleatorios
- `default_values_for_ui()` y `clear_employee_values()`: controlan valores por defecto en la interfaz
- `random_employee_values()`: llena el formulario con un empleado aleatorio
- `predict_single()`: hace la predicción individual
- `csv_format_view()`: muestra cómo debe verse el CSV
- `predict_batch()`: procesa archivos CSV con varios empleados
- definición de las pestañas de Gradio: una para crear empleado y otra para subir CSV

---
## Flujo del proyecto

El flujo general del proyecto empieza en train.py, donde se cargan los datos y se aplican las funciones definidas en preprocessing.py para prepararlos correctamente. Con esos datos ya transformados, model.py construye la arquitectura del modelo SAINT-like. Después, el mismo train.py se encarga de entrenarlo y utiliza funciones de utils.py para calcular métricas, evaluar el desempeño y guardar los resultados. Una vez terminado este proceso, tanto el modelo entrenado como el preprocesador se almacenan en la carpeta outputs/. Más adelante, inference.py carga esos archivos para poder hacer predicciones sobre datos nuevos, y finalmente app.py usa esa parte de inferencia para mostrar las predicciones dentro de la interfaz desarrollada en Gradio.

---



---

## Fuente del dataset

- Rohit Kumar. *Employee Promtion Prediction Dataset*. Kaggle.

