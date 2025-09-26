import json
import os
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

N_SAMPLES = 1000
N_FEATURES = 10


def dataset_generation():
    
    X, y =  make_regression(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,      
        # random_state=42
        )
    
    coefficients = np.random.randn(N_FEATURES)
    y = X @ coefficients + np.random.normal(0, 0.5, size=N_SAMPLES)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
        )
    
    return X_train, X_test, y_train, y_test
   
def building_model(X_train, y_train, X_test, y_test):
    model = BaggingRegressor(estimator=LinearRegression())
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "predictions": y_pred.tolist(),
        "mse": mse,
        "r2": r2
    }, model

def analysis_model(model, X_test, y_test, y_pred, folder_name):
    df_features = pd.DataFrame(X_test)
    vif_data = pd.DataFrame()
    vif_data["features"] = df_features.columns
    vif_data["VIF"] = [variance_inflation_factor(df_features.values, i) for i in range(df_features.shape[1])]

    correlations = df_features.corr(method='pearson')
    excel_path = os.path.join(folder_name, "correlation_matrix.xlsx")
    correlations.to_excel(excel_path)

    residuals = y_test - y_pred
    avg_coef = np.mean([est.coef_ for est in model.estimators_], axis=0)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.hlines(0, y_pred.min(), y_pred.max(), colors='blue', linestyles='dashed')
    plt.title("График остатков (хорошо, если значения около 0)")
    plt.xlabel("Предсказанные значения")
    plt.ylabel("Остатки")
    plt.grid(True)
    plot_path = os.path.join(folder_name, "residuals_plot.png")
    plt.savefig(plot_path)
    plt.close()
    
    return {
        "vif": vif_data.to_dict(),
        "residuals": residuals.tolist(),
        "feature_importance": avg_coef.tolist()
    }

def results_json(building_results, analysis_results, model, folder_name):

    building_path = os.path.join(folder_name, "building_results.json")
    with open(building_path, 'w', encoding='utf-8') as f:
        json.dump(building_results, f, ensure_ascii=False, indent=4)
    
    analysis_path = os.path.join(folder_name, "analysis_results.json")
    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=4)

    model_path = os.path.join(folder_name, "model.joblib")
    joblib.dump(model, model_path)

def main():
    X_train, X_test, y_train, y_test = dataset_generation()
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"task2_regression_{now}"
    os.makedirs(folder_name, exist_ok=True)

    building_results, model  = building_model(X_train, y_train, X_test, y_test)
    y_pred =  np.array(building_results["predictions"])
    analysis_results = analysis_model(model, X_test, y_test, y_pred, folder_name)
    results_json(building_results, analysis_results, model, folder_name)

if __name__ == "__main__":
    main()