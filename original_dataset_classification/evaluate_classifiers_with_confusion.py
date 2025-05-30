
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
import warnings

# Uyarıları bastır
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Veri seti ve model konfigürasyonları
dataset_configs = {
    "HDL vs CONTROL": {
        "csv": "./CSV/features_for_binary_classification_HDL_CONTROL.csv",
        "label_map": {0: "Control", 1: "HDL"},
        "model": ExtraTreesClassifier()
    },
    "HDL vs LDL": {
        "csv": "./CSV/features_for_binary_classification.csv",
        "label_map": {0: "LDL", 1: "HDL"},
        "model": GaussianNB()
    },
    "LDL vs CONTROL": {
        "csv": "./CSV/features_for_binary_classification_LDL_CONTROL.csv",
        "label_map": {0: "Control", 1: "LDL"},
        "model": AdaBoostClassifier(n_estimators=200)
    },
    "MULTICLASS HDL/CONTROL/LDL": {
        "csv": "./CSV/features_for_multi_classification.csv",
        "label_map": {0: "HDL", 1: "CONTROL", 2: "LDL"},
        "model": LogisticRegression(max_iter=500)
    }
}

# Tüm veri setleri üzerinde işlem yap
for dataset_name, config in dataset_configs.items():
    print(f"Veri seti işleniyor: {dataset_name}")

    df = pd.read_csv(config["csv"])
    X = df[["radial_mean", "theta_mean", "radial_fft", "theta_fft"]]
    y = df["label"].map(config["label_map"])

    y_true_all, y_pred_all = [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    for seed in range(100):
        skf.random_state = seed
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = config["model"]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)

    # Confusion Matrisleri
    label_list = list(config["label_map"].values())
    cm = confusion_matrix(y_true_all, y_pred_all, labels=label_list)
    cm_norm = confusion_matrix(y_true_all, y_pred_all, labels=label_list, normalize="true")

    # Sınıflandırma raporu (temel)
    report = classification_report(
        y_true_all, y_pred_all, labels=label_list, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T[["precision", "recall", "f1-score", "support"]].round(3)
    report_df = report_df.loc[label_list]

    # Örnek sayısı bilgisi
    n_samples = len(df)
    n_features = X.shape[1]
    sample_info = f"Total Samples: {n_samples} × {n_features} features = {n_samples * n_features} feature-values"

    # Görselleştirme
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={'width_ratios': [1, 1, 1.2]})

    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_list, yticklabels=label_list, ax=axes[0])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Normalized Confusion Matrix
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=label_list, yticklabels=label_list, ax=axes[1])
    axes[1].set_title("Normalized Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    # Classification Report Table
    axes[2].axis('off')
    table_data = report_df.reset_index().values
    table = axes[2].table(cellText=table_data,
                          colLabels=report_df.reset_index().columns,
                          cellLoc='center',
                          loc='center')
    table.scale(1, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    axes[2].set_title("Classification Report", pad=20)

    # Alt köşeye örnek bilgisi ekle
    fig.text(0.95, 0.02, sample_info, ha='right', fontsize=9, color='gray')

    # Başlık ve görsel kaydetme
    plt.suptitle(f"{dataset_name} – Best Classifier (500 iterations)", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"confusion_report_with_table_{dataset_name.replace('/', '_').replace(' ', '_')}.png", dpi=300)
    plt.show()

    print(f"{dataset_name} için confusion matrix görseli ve tablo kaydedildi.")
