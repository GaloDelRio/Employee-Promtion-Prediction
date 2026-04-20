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

## Estado actual del proyecto

En este momento el proyecto ya cuenta con:

- un dataset explorado y depurado
- variables seleccionadas con base en correlación y redundancia
- división en entrenamiento, validación y prueba
- preprocesamiento para variables numéricas y categóricas
- balanceo del conjunto de entrenamiento con SMOTE
- una primera implementación de Random Forest
- una evaluación inicial con métricas adecuadas para clasificación desbalanceada

Con esto, la base experimental quedó lista para continuar con una siguiente fase de mejora del modelo, comparación con otras técnicas y documentación de resultados.

## Posibles mejoras a futuro

A partir de los resultados obtenidos, algunas acciones que podrían implementarse en etapas posteriores son:

- ajuste de hiperparámetros del Random Forest
- comparación contra otros modelos de clasificación
- prueba de modelos de ensamble más avanzados, como XGBoost
- análisis del umbral de decisión para mejorar el equilibrio entre precision y recall
- revisión del efecto de usar SMOTE frente a otras estrategias de balanceo
- análisis de importancia de variables para entender mejor cuáles atributos influyen más en la predicción

Estas mejoras permitirán determinar si es posible aumentar el desempeño del modelo, especialmente en la detección de la clase positiva.

---

## Fuente del dataset

- Rohit Kumar. *Employee Promtion Prediction Dataset*. Kaggle.

