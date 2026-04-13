# Reporte del dataset y preprocesamiento realizado

## Dataset utilizado

Para este proyecto se utilizó el dataset **Employee Promtion Prediction Dataset**, publicado en Kaggle por **Rohit Kumar**. Este conjunto de datos está orientado al análisis de variables relacionadas con el desempeño, compromiso y características laborales de empleados, con el objetivo de predecir si un colaborador será promovido o no. Este contiene alrededor de **100,000 registros**,con **43 variables** y una variable objetivo llamada **promoted**.

Además, la tasa de promoción aproximada de **10%**, lo que lo convierte en un problema de clasificación binaria con desbalance de clases.
En términos generales, el dataset reúne información que puede representar distintos factores asociados a una promoción laboral, como desempeño, historial reciente, participación en proyectos, entrenamiento, asistencia y otros indicadores internos. Debido a esto, resulta adecuado para un problema de *machine learning* supervisado enfocado en clasificación, donde la meta es aprender patrones que permitan distinguir entre empleados promovidos y no promovidos.

---

## Objetivo del análisis

Hasta este punto, el trabajo realizado no se enfocó en entrenar un modelo, sino en preparar correctamente los datos para que posteriormente puedan ser usados en una etapa de modelado. El objetivo principal fue dejar listo un conjunto de entrenamiento limpio, transformado y balanceado, además de un conjunto de prueba separado correctamente para evaluación posterior.

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

## Estado actual del proyecto

En esta etapa todavía no se ha entrenado ningún modelo. Todo el trabajo realizado corresponde a la fase de preparación de datos, que es fundamental antes del modelado. Gracias a este procesamiento, el dataset ya quedó listo para la siguiente fase, donde se podrá entrenar y evaluar un modelo de clasificación con una base de datos más limpia, más consistente y mejor balanceada.

---

## Fuente del dataset

- Rohit Kumar. *Employee Promtion Prediction Dataset*. Kaggle.

