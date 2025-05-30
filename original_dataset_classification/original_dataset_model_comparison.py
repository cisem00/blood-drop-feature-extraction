import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import ( 
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.base import clone

# Uyarıları bastır
for warning_type in [
    FutureWarning, ConvergenceWarning, UserWarning, RuntimeWarning, UndefinedMetricWarning]:
    warnings.filterwarnings("ignore", category=warning_type)

# CSV dosyaları ve etiket eşlemeleri
csv_configs = {
    "HDL vs CONTROL": {
        "path": "./CSV/features_for_binary_classification_HDL_CONTROL.csv",
        "label_map": {0: "Control", 1: "HDL"}
    },
    "LDL vs CONTROL": {
        "path": "./CSV/features_for_binary_classification_LDL_CONTROL.csv",
        "label_map": {0: "Control", 1: "LDL"}
    },
    "HDL vs LDL": {
        "path": "./CSV/features_for_binary_classification.csv",
        "label_map": {0: "LDL", 1: "HDL"}
    },
    "MULTICLASS HDL/CONTROL/LDL": {
        "path": "./CSV/features_for_multi_classification.csv",
        "label_map": {0: "HDL", 1: "CONTROL", 2: "LDL"}
    }
}

# Sınıflandırıcılar
classifiers = {
    "Random Forest": RandomForestClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "AdaBoost": AdaBoostClassifier(n_estimators=200),
    "Gradient Boosting": GradientBoostingClassifier(n_iter_no_change=5, validation_fraction=0.1),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=500),
    "MLP Classifier": MLPClassifier(max_iter=500),
    "Naive Bayes": GaussianNB(),
    "SVC": SVC(max_iter=500),
}

def evaluate_models(X, y, is_multiclass=False):
    """
    evaluate_models fonksiyonu.

    Args:
        X: Özellikler matrisi.
        y: Etiketler.
        is_multiclass: Çoklu sınıflandırma bayrağı.

    Returns:
        Model performans skorları.
    """

    results = {name: {m: [] for m in ["Accuracy", "Precision", "Recall", "F1-Score"]} for name in classifiers}
    average_type = 'weighted' if is_multiclass else 'binary'
    pos_label = None if is_multiclass else 1

    for seed in range(100):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            for name, clf in classifiers.items():
                try:
                    model = clone(clf)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    results[name]["Accuracy"].append(accuracy_score(y_test, y_pred))
                    results[name]["Precision"].append(precision_score(
                        y_test, y_pred, average=average_type, pos_label=pos_label, zero_division=0))
                    
                    results[name]["Recall"].append(recall_score(
                        y_test, y_pred, average=average_type, pos_label=pos_label, zero_division=0))
                    
                    results[name]["F1-Score"].append(f1_score(
                        y_test, y_pred, average=average_type,pos_label=pos_label, zero_division=0))
                except Exception as e:
                    print(f"{name} başarısız oldu: {e}")
                    for m in results[name]:
                        results[name][m].append(0.0)

    return results

def summarize_results(results_dict):
    """
    summarize_results fonksiyonu.

    Args:
        results_dict: Sonuç sözlüğü.

    Returns:
        Özet performans raporu.
    """

    return pd.DataFrame([
        {
            "Classifier": name,
            "Accuracy": round(np.mean(scores["Accuracy"]), 3),
            "Precision": round(np.mean(scores["Precision"]), 3),
            "Recall": round(np.mean(scores["Recall"]), 3),
            "F1-Score": round(np.mean(scores["F1-Score"]), 3)
        }
        for name, scores in results_dict.items()
    ]).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)

def plot_results(df, title):
    """
    plot_results fonksiyonu.

    Args:
        df: Sonuç dataframe.
        title: Grafik başlığı.

    Returns:
        Görselleştirme grafiği.
    """

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        df.set_index("Classifier"), annot=True, fmt=".3f", cmap="YlGnBu", cbar=False)
    
    plt.title(f"Classifier Performance – 100x5 CV ({title})", fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    safe_title = re.sub(r"[^a-zA-Z0-9_]", "_", title)
    filename = f"classifier_comparison_{safe_title}_100x5.png"
    plt.savefig(filename, dpi=300)
    plt.show()

    print(f"{title} tamamlandı. Görsel kaydedildi: {filename}")

def main():
    """
    Her CSV dosyası için sınıflandırıcıları değerlendirir ve görselleştirir.
    """
    for title, config in csv_configs.items():
        print(f"\nİşleniyor: {title}")
        df = pd.read_csv(config["path"])
        y = df["label"]
        X = df.drop(columns=["label"])
        is_multiclass = len(np.unique(y)) > 2

        results = evaluate_models(X, y, is_multiclass)
        summary_df = summarize_results(results)
        plot_results(summary_df, title)

if __name__ == "__main__":
    main()
