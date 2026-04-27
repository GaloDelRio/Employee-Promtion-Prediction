# Reporte del dataset y preprocesamiento realizado

## Dataset utilizado

Para este proyecto se utilizó el dataset **Employee Promtion Prediction Dataset**, publicado en Kaggle por **Rohit Kumar**. Este conjunto de datos está orientado al análisis de variables relacionadas con el desempeño, compromiso y características laborales de empleados, con el objetivo de predecir si un colaborador será promovido o no. Este contiene alrededor de **100,000 registros**,con **43 variables** y una variable objetivo llamada **promoted**.

Además, la tasa de promoción aproximada de **10%**, lo que lo convierte en un problema de clasificación binaria con desbalance de clases.
En términos generales, el dataset reúne información que puede representar distintos factores asociados a una promoción laboral, como desempeño, historial reciente, participación en proyectos, entrenamiento, asistencia y otros indicadores internos. Debido a esto, resulta adecuado para un problema de *machine learning* supervisado enfocado en clasificación, donde la meta es aprender patrones que permitan distinguir entre empleados promovidos y no promovidos.

---

## Exploración inicial de los datos

Primero se realizó una revisión general del dataset para entender su estructura, verificar el tipo de variables y revisar si existían valores faltantes. En esta exploración se observó que no habían columnas vacías ni un problema importante de datos nulos, por lo que no fue necesario eliminar registros ni hacer una imputación previa a nivel global.

También se analizó la distribución de la variable objetivo **`promoted`** y verifique que el dataset estaba **desbalanceado**: la mayoría de los registros pertenecían a empleados no promovidos y una minoría a empleados promovidos (10%/90%). Esto era importante detectarlo desde el inicio, porque un modelo entrenado directamente con esta distribución podría aprender a favorecer la clase mayoritaria y fallar al identificar correctamente los casos de promoción.


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

Una vez reducido el dataset, se separó la información en:

- **Variables de entrada (`X`)**: todas las características del empleado.
- **Variable objetivo (`y`)**: la columna `promoted`.

Esta separación era necesaria para preparar el problema de aprendizaje supervisado, donde el modelo recibirá las variables de entrada y tratará de aprender patrones para predecir la variable objetivo.


El siguiente paso fue dividir los datos en dos grupos:

- Conjunto de entrenamiento para ajustar el modelo posteriormente.
- Conjunto de prueba para evaluar su desempeño con datos no vistos.

La división se hizo usando una proporción de **80% para entrenamiento** y **20% para prueba**, manteniendo la proporción original de la variable objetivo en ambos conjuntos. Esto último fue importante porque, al estar desbalanceadas las clases, era necesario conservar una distribución similar en entrenamiento y prueba para que la evaluación futura fuera representativa del problema real.

Antes de transformar los datos, se identificó qué variables eran **numéricas** y cuáles eran **categóricas**. Esto fue importante porque ambos tipos de datos necesitan un tratamiento distinto.

Las variables numéricas requieren normalmente escalamiento para que sus magnitudes sean comparables, mientras que las variables categóricas deben convertirse a una representación numérica que pueda ser entendida por los algoritmos de machine learning.

---

## Preprocesamiento de los datos

Después se construyó una etapa de preprocesamiento diferenciando el tratamiento según el tipo de variable.

#### Variables numéricas

En las variables numéricas se aplicó:

- Imputación por mediana como medida preventiva en caso de encontrar valores faltantes.
- Estandarización para que todas las variables quedaran en una escala comparable.

Esto ayuda a que ciertas variables no dominen a otras solo por tener magnitudes más grandes y hace que muchos algoritmos funcionen de forma más estable.

#### Variables categóricas

En las variables categóricas se aplicó:

- Imputación por moda como medida preventiva en caso de encontrar valores faltantes.
- Codificación One-Hot para transformar categorías en columnas binarias.

Esta transformación permite representar la información categórica en un formato numérico sin introducir un orden artificial entre categorías.

El preprocesamiento se ajustó únicamente con el conjunto de entrenamiento y después se aplicó al conjunto de prueba. Esto fue importante para evitar fuga de información, ya que el conjunto de prueba no debe influir en la construcción de las transformaciones.

#### Balanceo de clases con SMOTE

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

## Métricas de evaluación seleccionadas y resultados obtenidos

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


Los resultados muestran que el modelo logró aprender patrones útiles del conjunto de entrenamiento, ya que en ese subconjunto alcanzó valores altos en todas las métricas. Sin embargo, al pasar a validación y prueba se observó una disminución importante en el desempeño, especialmente en precision, recall y F1-score de la clase positiva.

Esto sugiere que el modelo presenta cierto nivel de sobreajuste. En otras palabras, aprendió muy bien el conjunto de entrenamiento, pero no mantuvo el mismo nivel de generalización en datos nuevos.

Aun así, los resultados de validación y prueba fueron muy parecidos entre sí, lo cual es una señal positiva. Esto indica que el comportamiento del modelo fue estable fuera del entrenamiento y que la evaluación final es consistente.

También se observó que el valor de accuracy se mantuvo alrededor de 0.86 en validación y prueba. Aunque este resultado parece alto, no debe interpretarse como prueba suficiente de buen desempeño, ya que el conjunto de datos está desbalanceado y la mayoría de los registros pertenecen a la clase no promovida.

La parte más importante del análisis está en la clase positiva. El modelo logró identificar aproximadamente la mitad de los empleados promovidos reales, lo cual se refleja en un recall cercano a 0.52–0.54. Sin embargo, la precision de 0.37 indica que una parte considerable de los casos predichos como promovidos realmente no lo eran. Como consecuencia, el F1-score de la clase positiva quedó en un nivel intermedio, alrededor de 0.43–0.44.

En conjunto, esto significa que el modelo sí tiene capacidad para detectar promociones, pero todavía comete bastantes errores al hacerlo. Por lo tanto, puede considerarse una primera aproximación funcional, aunque aún hay espacio de mejora antes de tomarlo como modelo final.

---

## Implementación de modelo basado en un articulo del estado del arte

Para esta etapa del proyecto se seleccionó un modelo **SAINT-like** implementado en **Keras**, inspirado en la arquitectura **SAINT (Self-Attention and Intersample Attention Transformer)**, la cual fue propuesta para trabajar con datos tabulares. Se tomó como referencia este artículo [*Enhancing Employee Promotion Prediction with Hybrid Deep Learning and SAINT-based Transformers*](https://arxiv.org/pdf/2604.10337) , ya que el modelo que propone ,resultaba útil para entender cómo este tipo de arquitectura puede aplicarse a variables laborales y organizacionales sin depender únicamente de modelos clásicos de árboles.

Este modelo SAINT se eligió porque el problema planteado es de tipo **clasificación tabular** y contiene tanto variables numéricas como categóricas, por lo que resultaba interesante probar una arquitectura que pudiera capturar relaciones entre ambas de forma más flexible que un modelo tradicional.

Además, SAINT fue considerado como una opción relevante porque en la literatura reciente se presenta como una arquitectura enfocada precisamente en datos tabulares y en la generación de representaciones contextuales de las variables. Esto lo vuelve atractivo en escenarios donde no solo importa el valor individual de cada columna, sino también la relación entre varias de ellas.

Se utilizó un modelo **SAINT-like** y no una implementación completa de **SAINT** porque el objetivo del proyecto no era replicar exactamente la arquitectura original, sino adaptar sus ideas principales a un entorno más sencillo y manejable en **Keras**. SAINT original incluye componentes más especializados, como mecanismos de atención entre columnas y entre muestras, además de estrategias de entrenamiento más complejas. En cambio, el modelo implementado en este proyecto conserva la idea central de representar las variables tabulares como tokens y usar atención tipo Transformer para aprender relaciones entre ellas, pero simplifica la arquitectura para que fuera más fácil de entrenar, ajustar e integrar con el flujo de preprocesamiento del dataset.

La principal diferencia es que el modelo usado en el proyecto toma las variables categórica y numericas mediante **embeddings**, transforma las variables numéricas con capas densas y después combina ambas reprersentaciones para pasarlas por bloques de atención. Por eso se le llama **SAINT-like**: no es SAINT puro, pero está inspirado en su enfoque para datos tabulares. Esta decisión permitió experimentar con una arquitectura moderna basada en atención sin aumentar demasiado la complejidad del proyecto, manteniendo una implementación más clara, compatible con Keras y adecuada para comparar contra modelos clásicos como **Random Forest**.


Sin embargo, el artículo tomado como referencia no fue usado como una receta exacta para replicar, sino como una guía conceptual. De hecho, ese trabajo reporta que los modelos basados en árboles mantuvieron un mejor desempeño que SAINT y que los enfoques híbridos evaluados. Por esta razón, en este proyecto el uso de SAINT-like se entiende principalmente como una alternativa experimental y de comparación, útil para analizar otra familia de modelos sobre el mismo problema.

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

A continuación se mostraran los archivos utilizados en el proyecto y sus resposabilidades individuales:

`config.py`

Este archivo concentra la configuración general del proyecto. Aquí se definen rutas, nombres de archivos de salida, columna objetivo y los hiperparámetros principales del modelo, como `EMB_DIM`, `NUM_HEADS`, `NUM_LAYERS`, `DROPOUT`, `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE` y `PATIENCE`.

- definición de rutas de trabajo (`DATA_DIR`, `OUTPUT_DIR`)
- columna objetivo (`TARGET_COLUMN`)
- columnas que se eliminan (`COLUMNS_TO_DROP`)
- hiperparámetros del modelo
- rutas donde se guardan modelo, métricas, historial y gráficas

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


`model.py`

Aquí se define la arquitectura del modelo SAINT-like en Keras. Es el archivo donde se construye la red neuronal.

- `SliceColumn`: capa personalizada que extrae una columna específica de la entrada
- `transformer_block()`: define el bloque de atención tipo Transformer
- `build_saint_like_model()`: arma el modelo completo con entradas categóricas, numéricas, bloque Transformer y clasificador final
- `get_model_summary_text()`: guarda el resumen del modelo como texto

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


`inference.py`

Este archivo sirve para usar el modelo ya entrenado sobre nuevos datos. No entrena nada, solo carga el modelo y genera predicciones.

- `PromotionPredictor.__init__()`: carga el modelo y el preprocesador
- `predict_dataframe()`: predice varias filas y devuelve probabilidad y clase
- `predict_one()`: predice un solo registro a partir de un diccionario


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

El flujo general del proyecto empieza en train.py, donde se cargan los datos y se aplican las funciones definidas en preprocessing.py para prepararlos correctamente. Con esos datos ya transformados, model.py construye la arquitectura del modelo SAINT-like. Después, el mismo train.py se encarga de entrenarlo y utiliza funciones de utils.py para calcular métricas, evaluar el desempeño y guardar los resultados. Una vez terminado este proceso, tanto el modelo entrenado como el preprocesador se almacenan en la carpeta outputs/. Más adelante, inference.py carga esos archivos para poder hacer predicciones sobre datos nuevos, y finalmente app.py usa esa parte de inferencia para mostrar las predicciones dentro de la interfaz desarrollada en Gradio.

---

## Iteraciones del modelo SAINT-like

A continuación se presentan las distintas iteraciones evaluadas del modelo SAINT-like, incluyendo la configuración utilizada en cada caso, los resultados obtenidos y la razón por la cual se eligió probar esa combinación de hiperparámetros.

### Iteración 1

<img width="629" height="391" alt="image" src="https://github.com/user-attachments/assets/095608d1-c62a-424b-914b-5e8a9ab1da3d" />

#### Hiperparámetros
- `EMB_DIM = 8`
- `NUM_HEADS = 1`
- `NUM_LAYERS = 1`
- `DROPOUT = 0.10`
- `BATCH_SIZE = 1024`
- `EPOCHS = 15`
- `LEARNING_RATE = 0.001`
- `PATIENCE = 3`

**Entrenamiento detenido en la época 10**

#### Resultados

**Train**
- Accuracy: `0.8834`
- Precision: `0.4536`
- Recall: `0.8105`
- F1-score: `0.5816`
- ROC-AUC: `0.9384`
- PR-AUC: `0.6689`

**Validation**
- Accuracy: `0.8824`
- Precision: `0.4511`
- Recall: `0.8120`
- F1-score: `0.5800`
- ROC-AUC: `0.9382`
- PR-AUC: `0.6604`

**Test**
- Accuracy: `0.8821`
- Precision: `0.4502`
- Recall: `0.8090`
- F1-score: `0.5785`
- ROC-AUC: `0.9350`
- PR-AUC: `0.6508`

#### Justificación
Esta configuración se tomó como punto de partida porque representa una versión sencilla y ligera del modelo. Se eligió para establecer una línea base con la cual comparar el efecto de modificar el tamaño de embeddings, el número de heads y la profundidad del bloque Transformer. Al usar una estructura simple, esta iteración permite observar el comportamiento inicial del modelo sin añadir demasiada complejidad arquitectónica.

### Iteración 2

<img width="629" height="391" alt="image" src="https://github.com/user-attachments/assets/0155dc6a-47bb-45eb-ab97-b0437267e896" />


#### Hiperparámetros
- `EMB_DIM = 16`
- `NUM_HEADS = 1`
- `NUM_LAYERS = 1`
- `DROPOUT = 0.10`
- `BATCH_SIZE = 1024`
- `EPOCHS = 15`
- `LEARNING_RATE = 0.001`
- `PATIENCE = 3`

**Entrenamiento detenido en la época 6**

#### Resultados

**Train**
- Accuracy: `0.8941`
- Precision: `0.4815`
- Recall: `0.7668`
- F1-score: `0.5915`
- ROC-AUC: `0.9356`
- PR-AUC: `0.6608`

**Validation**
- Accuracy: `0.8914`
- Precision: `0.4733`
- Recall: `0.7630`
- F1-score: `0.5842`
- ROC-AUC: `0.9357`
- PR-AUC: `0.6492`

**Test**
- Accuracy: `0.8939`
- Precision: `0.4811`
- Recall: `0.7770`
- F1-score: `0.5943`
- ROC-AUC: `0.9343`
- PR-AUC: `0.6446`

#### Justificación
En esta iteración se incrementó `EMB_DIM` para darle al modelo una representación interna más rica de cada variable. La intención fue evaluar si un embedding más grande ayudaba a capturar mejor la información contenida en las columnas del dataset sin modificar todavía la estructura general del Transformer. Esta prueba permitió aislar el efecto del tamaño del embedding respecto a la configuración base.

### Iteración 3

<img width="629" height="391" alt="image" src="https://github.com/user-attachments/assets/2bfc0a64-bd6f-4ebd-bfe6-20848ebeac0d" />


#### Hiperparámetros
- `EMB_DIM = 16`
- `NUM_HEADS = 2`
- `NUM_LAYERS = 1`
- `DROPOUT = 0.10`
- `BATCH_SIZE = 1024`
- `EPOCHS = 15`
- `LEARNING_RATE = 0.001`
- `PATIENCE = 5`

**Entrenamiento detenido en la época 9**

#### Resultados

**Train**
- Accuracy: `0.8911`
- Precision: `0.4734`
- Recall: `0.7886`
- F1-score: `0.5916`
- ROC-AUC: `0.9383`
- PR-AUC: `0.6695`

**Validation**
- Accuracy: `0.8886`
- Precision: `0.4660`
- Recall: `0.7820`
- F1-score: `0.5840`
- ROC-AUC: `0.9380`
- PR-AUC: `0.6619`

**Test**
- Accuracy: `0.8881`
- Precision: `0.4649`
- Recall: `0.7870`
- F1-score: `0.5845`
- ROC-AUC: `0.9363`
- PR-AUC: `0.6533`

#### Justificación
Después de aumentar el tamaño del embedding, el siguiente paso fue incrementar `NUM_HEADS` para que el bloque de atención pudiera analizar relaciones entre variables desde más de una perspectiva. Con esta modificación se buscó evaluar si múltiples heads ayudaban a capturar interacciones más útiles entre columnas, manteniendo todavía una sola capa Transformer para no aumentar demasiado la complejidad del modelo.

### Iteración 4

<img width="629" height="391" alt="image" src="https://github.com/user-attachments/assets/d8734eee-561d-4341-99ce-39e69750c5b2" />


#### Hiperparámetros
- `EMB_DIM = 16`
- `NUM_HEADS = 2`
- `NUM_LAYERS = 2`
- `DROPOUT = 0.10`
- `BATCH_SIZE = 1024`
- `EPOCHS = 15`
- `LEARNING_RATE = 0.001`
- `PATIENCE = 5`

**Entrenamiento detenido en la época 12**

#### Resultados

**Train**
- Accuracy: `0.8879`
- Precision: `0.4654`
- Recall: `0.8159`
- F1-score: `0.5927`
- ROC-AUC: `0.9418`
- PR-AUC: `0.6772`

**Validation**
- Accuracy: `0.8838`
- Precision: `0.4544`
- Recall: `0.8070`
- F1-score: `0.5814`
- ROC-AUC: `0.9391`
- PR-AUC: `0.6628`

**Test**
- Accuracy: `0.8839`
- Precision: `0.4549`
- Recall: `0.8110`
- F1-score: `0.5828`
- ROC-AUC: `0.9380`
- PR-AUC: `0.6635`

#### Justificación
En esta iteración se añadió un segundo bloque Transformer mediante `NUM_LAYERS = 2`, con el objetivo de que el modelo pudiera aprender relaciones más profundas entre las variables. La hipótesis fue que una arquitectura más profunda permitiría refinar mejor las interacciones entre atributos laborales, personales y de desempeño. Esta prueba fue importante para observar si una mayor capacidad del modelo mejoraba la detección de la clase positiva.

### Iteración 5

<img width="629" height="391" alt="image" src="https://github.com/user-attachments/assets/f65a2b56-1973-416a-97df-994948a3bee7" />

#### Hiperparámetros
- `EMB_DIM = 16`
- `NUM_HEADS = 2`
- `NUM_LAYERS = 2`
- `DROPOUT = 0.20`
- `BATCH_SIZE = 512`
- `EPOCHS = 30`
- `LEARNING_RATE = 0.001`
- `PATIENCE = 5`

**Entrenamiento detenido en la época 18**

#### Resultados

**Train**
- Accuracy: `0.8943`
- Precision: `0.4827`
- Recall: `0.7985`
- F1-score: `0.6017`
- ROC-AUC: `0.9427`
- PR-AUC: `0.6812`

**Validation**
- Accuracy: `0.8884`
- Precision: `0.4653`
- Recall: `0.7780`
- F1-score: `0.5823`
- ROC-AUC: `0.9394`
- PR-AUC: `0.6667`

**Test**
- Accuracy: `0.8899`
- Precision: `0.4698`
- Recall: `0.7850`
- F1-score: `0.5878`
- ROC-AUC: `0.9358`
- PR-AUC: `0.6581`

#### Justificación
Esta configuración se eligió para introducir mayor regularización y un entrenamiento un poco más fino. El aumento de `DROPOUT` buscó reducir el sobreajuste, mientras que un `BATCH_SIZE` más pequeño permitió hacer más actualizaciones durante cada época. Además, al aumentar `EPOCHS` se dio al modelo más oportunidad de aprendizaje, dejando que `PATIENCE` controlara automáticamente el momento en que debía detenerse si ya no había mejora en validación.

---
## Comparación entre métricas de las iteraciones del modelo

A partir de los resultados obtenidos, puede observarse que no existió una sola iteración que dominara de forma absoluta en todas las métricas. Por esta razón, el análisis debe hacerse revisando cada métrica por separado y después interpretando el comportamiento general del modelo en función del objetivo del problema. Dado que se trata de una clasificación desbalanceada, no basta con observar únicamente el accuracy; también es necesario considerar precision, recall, F1-score, ROC-AUC y PR-AUC, especialmente en validación y prueba.

La métrica de **accuracy** mide la proporción total de predicciones correctas. En este caso, la mejor iteración en el conjunto de prueba fue la **Iteración 2**, con un valor de `0.8939`, seguida por la **Iteración 5** con `0.8899`. Las iteraciones 1 y 4 obtuvieron los valores más bajos, aunque la diferencia no fue extremadamente grande.

Esto indica que la Iteración 2 fue la que logró el mayor número de aciertos globales. Sin embargo, como la clase negativa es mayoritaria, este resultado debe interpretarse con cuidado, ya que un valor alto de accuracy no garantiza por sí solo que el modelo esté detectando bien a los empleados promovidos.

La **precision** indica qué proporción de los empleados predichos como promovidos realmente lo eran. En el conjunto de prueba, la mejor fue nuevamente la **Iteración 2**, con `0.4811`, seguida por la **Iteración 5** con `0.4698` y la **Iteración 3** con `0.4649`.

Esto sugiere que la Iteración 2 fue la más confiable al emitir predicciones positivas, es decir, fue la que produjo menos falsos positivos en comparación con las demás. En otras palabras, cuando este modelo predijo que un empleado sería promovido, tuvo una mayor probabilidad de acertar. Este comportamiento parece estar relacionado con el aumento de `EMB_DIM`, que probablemente permitió una representación interna más rica de las variables.

El **recall** mide qué proporción de los empleados realmente promovidos fue detectada por el modelo. En esta métrica, la mejor iteración fue la **Iteración 4**, con `0.8110` en prueba, seguida muy de cerca por la **Iteración 1** con `0.8090`. Las iteraciones 2 y 5 quedaron por debajo.

Este resultado es importante porque el recall es especialmente relevante cuando interesa detectar la mayor cantidad posible de promociones reales. La Iteración 4 destacó en este aspecto, lo que indica que su configuración más profunda, con `NUM_LAYERS = 2`, ayudó al modelo a encontrar mejor la clase positiva. Aunque no fue la más precisa, sí fue la más sensible para identificar empleados promovidos.

El **F1-score** resume el equilibrio entre precision y recall. En el conjunto de prueba, la mejor iteración fue la **Iteración 2**, con `0.5943`, seguida por la **Iteración 5** con `0.5878`, la **Iteración 3** con `0.5845`, la **Iteración 4** con `0.5828` y finalmente la **Iteración 1** con `0.5785`.

Esto indica que la Iteración 2 fue la que logró el mejor balance entre detectar promovidos y no equivocarse demasiado al marcarlos como positivos. Aunque la Iteración 4 tuvo mejor recall, su precision más baja redujo su F1-score. Por tanto, si el criterio principal fuera mantener equilibrio entre ambas métricas, la Iteración 2 sería la mejor opción.

La métrica **ROC-AUC** mide qué tan bien el modelo logra separar ambas clases de forma general. En prueba, la mejor iteración fue la **Iteración 4**, con `0.9380`, seguida por la **Iteración 3** con `0.9363`, la **Iteración 5** con `0.9358`, la **Iteración 1** con `0.9350` y la **Iteración 2** con `0.9343`.

Esto muestra que la Iteración 4 fue la que mejor logró distinguir entre empleados promovidos y no promovidos a nivel global, independientemente del umbral final de clasificación. En otras palabras, aunque no fue la mejor en accuracy o F1-score, sí fue una de las más sólidas en capacidad de separación entre clases.

La métrica **PR-AUC** es especialmente importante en este problema, porque se enfoca más en el comportamiento de la clase positiva dentro de un contexto desbalanceado. En prueba, la mejor fue nuevamente la **Iteración 4**, con `0.6635`, seguida por la **Iteración 5** con `0.6581`, la **Iteración 3** con `0.6533`, la **Iteración 1** con `0.6508` y la **Iteración 2** con `0.6446`.

Esto sugiere que la Iteración 4 fue la que mejor manejó el compromiso entre precision y recall a distintos umbrales, no solo al umbral fijo de 0.5. Por ello, desde la perspectiva de un problema desbalanceado, esta iteración resulta particularmente fuerte y relevante.

Añandido a esto comparar las métricas de entrenamiento, validació y prueba, no se observa un caso claro de overfitting en las iteraciones evaluadas. En general, los resultados se mantienen relativamente estables entre los tres conjuntos, lo que indica que el modelo no está memorizando únicamente los datos de entrenamiento, sino que conserva un desempeño similar ante datos no vistos.

La iteración que muestra más señales de posible sobreajuste es la Iteración 5, ya que obtiene el F1-score más alto en entrenamiento con 0.6017, pero baja a 0.5823 en validación y a 0.5878 en prueba. Esta diferencia sugiere que el modelo empezó a ajustarse un poco más al conjunto de entrenamiento. Sin embargo, la caída no es suficientemente grande como para afirmar que existe un overfitting severo. Por lo tanto, puede considerarse una ligera tendencia al sobreajuste, pero no un problema grave.

En conclusión, ninguna iteración está claramente overfiteada. Las iteraciones 1, 2 y 3 son las más estables, mientras que las iteraciones 4 y 5 muestran una separación ligeramente mayor entre train y validation. La Iteración 5 tiene el mejor F1-score en entrenamiento, pero no necesariamente generaliza mejor, por lo que debe evaluarse con cuidado frente a la Iteración 4 y la Iteración 2.


## Comparación individual entre iteraciones

### Iteración 1
La primera iteración funcionó como configuración base y ofreció un desempeño estable entre entrenamiento, validación y prueba. Su principal fortaleza fue un recall alto, lo que significa que desde el inicio el modelo ya era capaz de detectar una buena cantidad de empleados promovidos. Sin embargo, quedó por debajo de otras iteraciones en accuracy, precision y F1-score, por lo que se comportó mejor como punto de referencia inicial que como mejor versión final.

### Iteración 2
La Iteración 2 fue la mejor en accuracy, precision y F1-score dentro del conjunto de prueba. Esto sugiere que aumentar `EMB_DIM` de 8 a 16 mejoró la representación interna de las variables y permitió una clasificación más equilibrada al usar el umbral fijo. Su principal debilidad fue una reducción en recall, ROC-AUC y PR-AUC, lo cual indica que el modelo se volvió más conservador: acertó más cuando predijo la clase positiva, pero detectó menos promovidos reales en comparación con otras configuraciones.

### Iteración 3
La Iteración 3 introdujo dos cabezas de atención manteniendo una sola capa Transformer. Su comportamiento quedó en un punto intermedio: no fue la mejor en ninguna métrica principal, pero mejoró respecto a la Iteración 2 en recall, ROC-AUC y PR-AUC. Esto sugiere que agregar más heads permitió capturar mejor algunas relaciones entre variables, aunque el cambio no fue suficiente para superar claramente a las mejores configuraciones.

### Iteración 4
La Iteración 4 fue la mejor en recall, ROC-AUC y PR-AUC. Esto la vuelve especialmente fuerte en el contexto de un dataset desbalanceado, ya que fue la que mejor detectó empleados promovidos y la que mostró mejor capacidad de separar ambas clases. Aunque perdió algo de precision y F1-score respecto a la Iteración 2, tuvo un comportamiento muy competitivo y una ventaja clara en métricas más importantes para este problema.

### Iteración 5
La Iteración 5 no fue la mejor absoluta en ninguna métrica, pero sí mostró un desempeño muy equilibrado. Mejoró respecto a la Iteración 4 en accuracy, precision y F1-score, aunque bajó ligeramente en recall, ROC-AUC y PR-AUC. Esto indica que la mayor regularización y el ajuste en batch size ayudaron a estabilizar el modelo y a hacerlo competitivo en casi todos los indicadores, quedando como una alternativa intermedia bastante sólida.



---

## ¿Cuál fue la mejor iteración?

La mejor iteración depende del criterio principal con el que se evalúe el modelo. Como el problema está desbalanceado, no basta con observar únicamente el accuracy, ya que una métrica alta no siempre significa que el modelo esté detectando correctamente a los empleados promovidos.

Si se toma como referencia el desempeño de clasificación final usando el umbral actual, la **Iteración 2** puede considerarse la mejor. Esta iteración obtuvo el mejor accuracy, la mejor precision y el mejor F1-score, por lo que fue la opción más sólida cuando se buscó un equilibrio entre desempeño general y confiabilidad en las predicciones positivas. En otras palabras, fue la iteración que mejor funcionó bajo las condiciones actuales de decisión del modelo.

Sin embargo, si se considera con mayor importancia la detección de la clase positiva, la **Iteración 4** resulta más adecuada. Esta iteración obtuvo mejores resultados en recall, ROC-AUC y PR-AUC, métricas especialmente relevantes en un problema donde interesa identificar la mayor cantidad posible de empleados promovidos. Por esta razón, aunque la Iteración 2 tuvo un mejor balance con el umbral actual, la Iteración 4 puede considerarse la mejor opción global cuando el objetivo principal es mejorar la detección de promociones reales.

---

## Comparación entre SAINT-like y Random Forest

Al comparar el modelo **SAINT-like** con **Random Forest**, se observa que SAINT-like obtuvo un mejor desempeño tanto en validación como en prueba, especialmente en las métricas más importantes para este problema desbalanceado. Aunque Random Forest funcionó como una línea base útil por ser un modelo más simple y rápido de entrenar, sus resultados fueron inferiores al momento de detectar la clase positiva.

En validación, la mejor iteración global de SAINT-like alcanzó un Accuracy de 0.8838, Precision de 0.4544, Recall de 0.8070 y F1-score de 0.5814. En comparación, Random Forest obtuvo un Accuracy de 0.8626, Precision de 0.3741, Recall de 0.5382 y F1-score de 0.4414. Esta diferencia muestra que SAINT-like no solo tuvo un mejor desempeño general, sino que también fue más efectivo para identificar correctamente a los empleados promovidos.

La misma tendencia se mantuvo en prueba. SAINT-like obtuvo un Accuracy de 0.8839, Precision de 0.4549, Recall de 0.8110 y F1-score de 0.5828, mientras que Random Forest alcanzó un Accuracy de 0.8620, Precision de 0.3703, Recall de 0.5199 y F1-score de 0.4325. Esto confirma que el modelo SAINT-like generalizó mejor y mantuvo un mejor equilibrio entre precision y recall fuera del conjunto de entrenamiento.

En términos prácticos, SAINT-like fue claramente superior para detectar empleados promovidos. Su mayor recall indica que logró identificar una mayor proporción de casos positivos, mientras que su mejor F1-score muestra que esta mejora no ocurrió de forma aislada, sino manteniendo un balance más adecuado con la precision.

---

## Conclusión

En conclusión, los resultados muestran que los cambios realizados en la arquitectura del modelo SAINT-like sí tuvieron un impacto real en su comportamiento. Aumentar el valor de EMB_DIM ayudó a mejorar la calidad general de la clasificación con el umbral actual, mientras que incrementar NUM_HEADS y NUM_LAYERS permitió capturar relaciones más complejas entre las variables, favoreciendo métricas más orientadas a la detección de la clase positiva.

Por esta razón, no existe una única “mejor” iteración para todos los casos. Si el objetivo principal es obtener el mejor equilibrio de clasificación final con el umbral actual, la **Iteración 2** representa la mejor alternativa. En cambio, si el objetivo central del proyecto es detectar la mayor cantidad posible de empleados promovidos dentro de un problema desbalanceado, la **Iteración 4** puede considerarse la opción más adecuada.

Al comparar SAINT-like con Random Forest, también se puede concluir que Random Forest fue útil como modelo base por su simplicidad, menor costo computacional y facilidad de entrenamiento. Sin embargo, su desempeño fuera del entrenamiento fue claramente inferior, especialmente en la detección de la clase positiva. Por otro lado, SAINT-like requirió mayor poder computacional, más tiempo de entrenamiento y una arquitectura más compleja, pero ofreció mejores resultados de generalización y un desempeño más fuerte en las métricas críticas del problema.

En conjunto, SAINT-like representa la mejor alternativa entre los dos enfoques evaluados cuando la prioridad es identificar correctamente promociones reales. Random Forest puede mantenerse como una referencia inicial, pero el modelo SAINT-like demostró ser más adecuado para el objetivo principal del proyecto.

---
## Fuente del dataset

- Rohit Kumar. *Employee Promtion Prediction Dataset*. Kaggle.

