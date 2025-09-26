import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def scattering_graph(X, y, folder_name):
    os.makedirs(folder_name, exist_ok=True)
    custom_colors = ["#00528d", "#d36900"]
    plt.figure(figsize=(8, 6))
    for i in np.unique(y):
        idx = (y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=custom_colors[i], label=f'Класс {i}', alpha=0.7)
    plt.title("Датасет для задачи классификации")
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(folder_name, "scatter_plot.png")
    plt.savefig(plot_path)
    plt.close()

def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def gini_score(y_true, y_proba):
    auc = roc_auc_score(y_true, y_proba[:, 1])
    return 2 * auc - 1

def train_and_evaluate_models(X_train, X_test, y_train, y_test, random_state=42):

    model_configs = {
        "Logistic Regression": {
            "model": LogisticRegression(random_state=random_state),
            "params": {'C': [0.1, 1, 10]}
        },
        "SVM": {
            "model": SVC(probability=True, random_state=random_state),
            "params": {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=random_state),
            "params": {'max_depth': [5, 10, 20], 'min_samples_split': [2, 5, 10]}
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=random_state),
            "params": {'n_estimators': [10, 50, 100], 'max_depth': [5, 10, 20]}
        }
    }

    results = {}
    for name, config in model_configs.items():
        model = config["model"]
        params = config["params"]

        grid = GridSearchCV(model, params, cv=5, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)
        y_proba = grid.predict_proba(X_test)
        gini = gini_score(y_test, y_proba)

        results[name] = {
            "best_params": grid.best_params_,
            "gini": gini
        }

    return results

def save_results(results, folder_name):
    results_path = os.path.join(folder_name, "results_model.txt")
    with open(results_path, 'w', encoding='utf-8') as f:      
        for name, result in results.items():
            f.write(f"{name}:\n")
            f.write(f"  Лучшие параметры: {result['best_params']}\n")
            f.write(f"  Коэффициент Джини: {result['gini']:.4f}\n")
            f.write("\n")


if __name__ == "__main__":
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"task_of_classification_{now}"

    X, y =  make_blobs(
        n_samples=1000,
        # n_features=2, по умолчанию задается 2 признака, можно не писать
        centers=2,
        cluster_std=3.8,
        center_box=(-6.0, 6.0),
        random_state=42
    )
    
    scattering_graph(X, y, folder_name)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    save_results(results, folder_name)