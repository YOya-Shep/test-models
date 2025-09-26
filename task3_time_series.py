import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing

N_SAMPLES = 1000


def data_generation():
    t = np.arange(N_SAMPLES)
    y = (
        10 * np.sin(2 * np.pi * t / 50)  # синусоида
        + 0.02 * t                       # тренд
        + np.random.normal(0, 1, size=N_SAMPLES)  # шум
    )
    return y

def split_data(y, train_size=0.8):
    split_idx = int(len(y) * train_size)
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    return y_train, y_test

def visualization_data(y, folder_name):
    plt.figure(figsize=(12, 6))
    plt.plot(y, alpha=0.8)
    plt.title("Синтетический временной ряд")
    plt.xlabel("Время")
    plt.ylabel("Значение")
    plt.grid(True)
    plot_path = os.path.join(folder_name, "time_series.png")
    plt.savefig(plot_path)
    plt.close()

def building_model_ar(y_train, y_test):
    model = AutoReg(y_train, lags=40)
    fitted_model = model.fit()
    n_forecast = len(y_test)
    forecast = fitted_model.forecast(steps=n_forecast)
    return {
        "model": fitted_model,
        "predictions": forecast
    }

def building_model_ets(y_train, y_test):
    model = ExponentialSmoothing(y_train, trend="add", damped_trend=True, seasonal="add", seasonal_periods=50)
    fitted_model = model.fit()
    n_forecast = len(y_test)
    forecast = fitted_model.forecast(steps=n_forecast)
    return {
        "model": fitted_model,
        "predictions": forecast
    }

def evaluation_model(y_test, predictions_ar, predictions_ets):
    
    # AR
    mae_ar = mean_absolute_error(y_test, predictions_ar)
    rmse_ar = np.sqrt(mean_squared_error(y_test, predictions_ar))

    # ETS
    mae_ets = mean_absolute_error(y_test, predictions_ets)
    rmse_ets = np.sqrt(mean_squared_error(y_test, predictions_ets))

    return {
        "AR": {"MAE": mae_ar, "RMSE": rmse_ar},
        "ETS": {"MAE": mae_ets, "RMSE": rmse_ets}
    }

def visualization_predictions(y_train, y_test, predictions_ar, predictions_ets, folder_name):
    n_train = len(y_train)
    n_test = len(y_test)
    total_len = n_train + n_test

    plt.figure(figsize=(14, 7))

    plt.plot(range(total_len), np.concatenate([y_train, y_test]), label="Истинные значения", alpha=0.7)
    plt.plot(range(n_train, total_len), predictions_ar, label="AR прогноз", alpha=0.7)
    plt.plot(range(n_train, total_len), predictions_ets, label="ETS прогноз", alpha=0.7)

    plt.title("Сравнение прогнозов AR и ETS")
    plt.xlabel("Время")
    plt.ylabel("Значение")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(folder_name, "AR_vs_ETS.png")
    plt.savefig(plot_path)
    plt.close()
    
def save_results_json(evaluation, folder_name):
    results_path = os.path.join(folder_name, "results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation, f, ensure_ascii=False, indent=4)

def main():
    y = data_generation()

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"task3_time_series_{now}"
    os.makedirs(folder_name, exist_ok=True)

    visualization_data(y, folder_name)

    y_train, y_test = split_data(y)
    
    ar_result = building_model_ar(y_train, y_test)
    ets_result = building_model_ets(y_train, y_test)
    predictions_ar = ar_result["predictions"]
    predictions_ets = ets_result["predictions"]

    evaluation = evaluation_model(y_test, predictions_ar, predictions_ets)
    visualization_predictions(y_train, y_test, predictions_ar, predictions_ets, folder_name)
    save_results_json(evaluation, folder_name)

if __name__ == "__main__":
    main()