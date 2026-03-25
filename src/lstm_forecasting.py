# =========================================================
# LSTM Forecasting - Google Trends Time Series
# =========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# =========================
# CONFIG
# =========================
WINDOW_SIZE = 10
TRAIN_SPLIT = 0.8
EPOCHS = 20
BATCH_SIZE = 16

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "multiTimeline.csv")
PLOT_PATH = os.path.join(BASE_DIR, "images", "forecast.png")


# =========================
# VALIDACIONES
# =========================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"No se encontró el archivo en: {DATA_PATH}\n"
        "Asegúrate de tener multiTimeline.csv dentro de data/raw/"
    )


# =========================
# DATA
# =========================
def load_data(path: str) -> pd.DataFrame:
    """
    Carga y limpia datos de Google Trends.
    Devuelve un DataFrame con:
    - ds: fecha
    - y: valor objetivo
    """
    df = pd.read_csv(path, skiprows=1)

    # Quitar columnas completamente vacías
    df = df.dropna(axis=1, how="all")

    # Quitar columnas duplicadas
    df = df.loc[:, ~df.columns.duplicated()]

    print("Columnas detectadas:", df.columns.tolist())

    if df.shape[1] < 2:
        raise ValueError(
            "El archivo no tiene al menos dos columnas utilizables."
        )

    # Tomar solo las primeras dos columnas reales
    df = df.iloc[:, :2].copy()
    df.columns = ["ds", "y"]

    # Limpiar espacios
    df["ds"] = df["ds"].astype(str).str.strip()
    df["y"] = df["y"].astype(str).str.strip()

    # Convertir tipos
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # Eliminar filas inválidas
    df = df.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)

    if df.empty:
        raise ValueError("Después de limpiar el archivo, no quedaron datos válidos.")

    return df


def scale_series(series: pd.Series):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    return scaled, scaler


def create_sequences(data: np.ndarray, window_size: int):
    X, y = [], []

    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])

    X = np.array(X)
    y = np.array(y)

    return X, y


def split_data(X: np.ndarray, y: np.ndarray, split_ratio: float):
    split_index = int(len(X) * split_ratio)

    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test


# =========================
# MODEL
# =========================
def build_model(window_size: int):
    model = Sequential([
        LSTM(50, activation="relu", input_shape=(window_size, 1)),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mae")
    return model


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return mean_absolute_error(y_true, y_pred)


def plot_results(dates, y_true, y_pred):
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label="Real")
    plt.plot(dates, y_pred, label="Predicción")
    plt.title("LSTM Forecast vs Serie Real")
    plt.xlabel("Fecha")
    plt.ylabel("Interés de búsqueda")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.show()


# =========================
# PIPELINE
# =========================
def run():
    df = load_data(DATA_PATH)

    series = df["y"]

    scaled_series, scaler = scale_series(series)

    X, y = create_sequences(scaled_series, WINDOW_SIZE)

    if len(X) == 0:
        raise ValueError(
            "No se pudieron generar secuencias. "
            "Revisa que la serie tenga más registros que WINDOW_SIZE."
        )

    X_train, X_test, y_train, y_test = split_data(X, y, TRAIN_SPLIT)

    model = build_model(WINDOW_SIZE)

    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )

    y_pred = model.predict(X_test, verbose=0)

    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred).flatten()

    mae = evaluate_model(y_test_inv, y_pred_inv)
    print(f"MAE del modelo: {mae:.4f}")

    test_start = WINDOW_SIZE + len(y_train)
    test_dates = df["ds"].iloc[test_start:test_start + len(y_test_inv)]

    plot_results(test_dates, y_test_inv, y_pred_inv)


if __name__ == "__main__":
    run()