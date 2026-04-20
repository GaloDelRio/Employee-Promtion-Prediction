# Employee-Promtion-Prediction

Este proyecto utiliza el dataset **Employee Promtion Prediction Dataset** de Kaggle, publicado por **Rohit Kumar**, para analizar la promoción de empleados a partir de distintas variables laborales.

El objetivo del proyecto es construir y evaluar un modelo de clasificación capaz de predecir si un empleado será promovido o no, usando información relacionada con desempeño, participación, historial laboral y otros indicadores internos.

## Trabajo realizado hasta esta etapa

En esta fase se desarrolló tanto la preparación de los datos como una primera implementación del modelo. El flujo realizado incluye:

- exploración inicial del dataset
- revisión de valores nulos
- análisis de correlación entre variables numéricas
- eliminación de variables poco útiles o redundantes
- separación en entrenamiento, validación y prueba
- preprocesamiento de variables numéricas y categóricas
- balanceo de clases con **SMOTE** sobre el conjunto de entrenamiento
- entrenamiento de un primer modelo de **Random Forest**
- evaluación inicial del modelo con métricas para clasificación desbalanceada

## Modelo implementado

En esta etapa se utilizó un modelo de **Random Forest** como primera aproximación para el problema de clasificación binaria. Este modelo fue entrenado con el conjunto de entrenamiento balanceado y evaluado posteriormente sobre validación y prueba.

## Estado actual

Actualmente el proyecto ya cuenta con una base experimental completa para seguir avanzando, incluyendo:

- dataset explorado y documentado
- variables seleccionadas
- pipeline de preprocesamiento
- estrategia de balanceo
- primer modelo entrenado
- evaluación inicial de resultados

## Documentación

La explicación completa del dataset, del preprocesamiento, del modelo implementado y de la interpretación de resultados se encuentra en `documentation.md`.

## Fuente del dataset

- Rohit Kumar. *Employee Promtion Prediction Dataset*. Kaggle.
