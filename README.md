# Time Series Forecasting with LSTM

Este proyecto implementa un modelo de Deep Learning basado en redes neuronales recurrentes (LSTM) para el análisis y predicción de series temporales utilizando datos de Google Trends.

## Objetivo

Modelar el comportamiento temporal de una variable y generar predicciones a partir de patrones históricos mediante un enfoque basado en secuencias.

## Dataset

El dataset proviene de Google Trends (`multiTimeline.csv`) y contiene el interés de búsqueda en el tiempo para un término específico.

Características:
- Valores normalizados (0–100)
- Frecuencia temporal continua
- Encabezados no estándar que requieren limpieza

## Pipeline

1. Carga de datos desde archivo CSV  
2. Limpieza:
   - Eliminación de columnas vacías  
   - Eliminación de columnas duplicadas  
   - Conversión de tipos (fecha y numérico)  
3. Transformación:
   - Escalado con MinMaxScaler  
   - Creación de secuencias (windowing)  
4. Modelado:
   - Red neuronal LSTM (1 capa + Dense)  
5. Entrenamiento:
   - División train/test  
   - Optimización con Adam  
   - Función de pérdida MAE  
6. Evaluación:
   - Error absoluto medio (MAE)  
7. Visualización:
   - Comparación de valores reales vs predicción  

## Resultados

- MAE aproximado: 12.1  
- El modelo captura la tendencia general de la serie, con desviaciones en cambios abruptos.

## Instalación

pip install -r requirements.txt

## Ejecución

python src/lstm_forecasting.py

## Tecnologías

- Python  
- TensorFlow / Keras  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  

## Consideraciones

- Implementación base de LSTM sin optimización de hiperparámetros  
- No se utilizan variables externas  
- Enfoque en pipeline completo de modelado  

## Próximos pasos

- Comparar contra modelos estadísticos (ARIMA, Prophet)  
- Optimización de hiperparámetros  
- Incluir variables externas  
- Evaluar con métricas adicionales (RMSE, MAPE)  

## Autor

Adrián Cuatecatl  
Data Analyst | Data Engineer
