# Time Series Forecasting con LSTM

## 🔍 Descripción

Este proyecto desarrolla un modelo de predicción de series de tiempo utilizando redes neuronales recurrentes tipo LSTM (Long Short-Term Memory), con el objetivo de estimar valores futuros a partir de datos históricos.

El enfoque se centra en capturar dependencias temporales y patrones secuenciales, lo cual es clave en contextos como comportamiento financiero, análisis de riesgo y pronóstico de demanda.

---

## 🎯 Objetivo

Construir un modelo de deep learning capaz de aprender la dinámica de una serie temporal y generar predicciones futuras, evaluando su desempeño mediante el error absoluto medio (MAE).

---

## 🧠 Metodología

El desarrollo del modelo sigue las siguientes etapas:

- Preparación y limpieza de datos  
- Normalización de la serie  
- Generación de secuencias temporales  
- División en conjuntos de entrenamiento, validación y prueba  
- Entrenamiento del modelo LSTM  
- Evaluación del modelo con MAE  
- Generación de predicciones sobre datos no vistos  

---

## 🛠️ Tecnologías utilizadas

- Python  
- Pandas  
- NumPy  
- TensorFlow / Keras  
- Scikit-learn  
- Matplotlib  

---

## 📊 Resultados

El modelo logra capturar la tendencia general de la serie temporal, mostrando un desempeño adecuado en términos de error de predicción.

Se identifican limitaciones en la capacidad del modelo para adaptarse a cambios abruptos o alta volatilidad, lo que abre la oportunidad de mejorar el enfoque mediante ingeniería de variables o ajustes en la arquitectura.

---

## ⚠️ Retos

- Sensibilidad del modelo a la escala de los datos  
- Selección adecuada de la longitud de las secuencias  
- Riesgo de sobreajuste  
- Interpretabilidad limitada frente a modelos tradicionales  

---

## 🚀 Mejoras futuras

- Optimización de hiperparámetros  
- Incorporación de variables externas  
- Implementación de modelos híbridos (por ejemplo, ARIMA + LSTM)  
- Despliegue del modelo para predicción en tiempo real  

---

## 📌 Nota

El dataset utilizado es de carácter simulado, pero replica estructuras y comportamientos de series de tiempo reales, permitiendo aplicar técnicas de modelado con un enfoque práctico.
