import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier,  RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
import warnings

#  Uyarıları bastır
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

#  Datasetler ve en iyi modeller
dataset_configs = {
    "HDL vs CONTROL": {
        "csv": "./CSV/60DegRotating/FinalClassificationCSVs/HDL_vs_CONTROL/Set_030deg.csv",
        "label_map": {0: "CONTROL", 1: "HDL"},
        "model": ExtraTreesClassifier()
    },
    "HDL vs LDL": {
        "csv": "./CSV/60DegRotating/FinalClassificationCSVs/HDL_vs_LDL/Set_016deg.csv",
        "label_map": {0: "HDL", 1: "LDL"},
        "model": LogisticRegression()
    },
    "LDL vs CONTROL": {
        "csv": "./CSV/45DegRotating/FinalClassificationCSVs/LDL_vs_CONTROL/Set_046deg.csv",
        "label_map": {0: "CONTROL", 1: "LDL"},
        "model": RandomForestClassifier()
    },
    "MULTICLASS HDL/CONTROL/LDL": {
        "csv": "./CSV/60DegRotating/FinalClassificationCSVs/HDL_vs_LDL_vs_CONTROL/Set_025deg.csv",
        "label_map": {0: "CONTROL", 1: "HDL", 2: "LDL"},
        "model": ExtraTreesClassifier()
    }
}

#  Tüm veri setleri için çalıştır
for dataset_name, config in dataset_configs.items():
    print(f"\n İşleniyor: {dataset_name}")
    
    df = pd.read_csv(config["csv"])
    if df["label"].dtype != "int":
        df["label"] = df["label"].astype("category").cat.codes

    X = df[["radial_mean", "theta_mean", "radial_fft", "theta_fft"]]
    y = df["label"].map(config["label_map"])

    y_true_all, y_pred_all = [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    for seed in range(100):  # 500 iterasyon = 5 fold × 100 tekrar
        skf.random_state = seed
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model = config["model"]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            y_true_all.extend(y_test)
            y_pred_all.extend(y_pred)

    #  Confusion Matrices
    label_list = list(config["label_map"].values())
    cm = confusion_matrix(y_true_all, y_pred_all, labels=label_list)
    #cm = (cm / 500).round(1) 
    cm_norm = confusion_matrix(y_true_all, y_pred_all, labels=label_list, normalize="true")

    #  Classification Report (sade)
    report = classification_report(y_true_all, y_pred_all, labels=label_list, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T[["precision", "recall", "f1-score", "support"]].round(3)
    report_df = report_df.loc[label_list]  # sadece sınıflar
    report_df["support"] = (report_df["support"] / 500).round(1)


    #  Görselleştir
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

    # Başlık ve kayıt
    plt.suptitle(f"{dataset_name} – Best Classifier (500 iterasyon)", fontsize=14)
    plt.tight_layout()
    save_name = f"confusion_report_with_table_{dataset_name.replace('/', '_').replace(' ', '_')}.png"
    plt.savefig(save_name, dpi=300)
    plt.show()

    # Raporu CSV olarak da kaydet
    report_df.to_csv(save_name.replace(".png", ".csv"))
    print(f" {dataset_name} için grafik + rapor CSV kaydedildi.")
