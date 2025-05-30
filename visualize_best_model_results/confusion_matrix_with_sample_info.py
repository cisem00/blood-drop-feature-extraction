import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from statistics import mode
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
#warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
#warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Dataset ve modeller
configs = {
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

for name, cfg in configs.items():
    df = pd.read_csv(cfg["csv"])
    if df["label"].dtype != int:
        df["label"] = df["label"].astype("category").cat.codes

    if "filename" in df.columns:
        ids = df["filename"]
    elif "source_image" in df.columns:
        ids = df["source_image"]
    else:
        raise ValueError("ID sütunu bulunamadı.")

    X = df[["radial_mean", "theta_mean", "radial_fft", "theta_fft"]]
    y_original = df["label"].map(cfg["label_map"])

    scenarios = [(cfg["label_map"], "original")]
    if "MULTICLASS" not in name:
        reversed_map = {k: v for k, v in zip(cfg["label_map"].keys(), list(cfg["label_map"].values())[::-1])}
        scenarios.append((reversed_map, "reversed"))

    for label_map, tag in scenarios:
        y = df["label"].map(label_map)
        label_list = list(label_map.values())

        true_labels = {}
        pred_votes = defaultdict(list)

        skf = StratifiedKFold(n_splits=5, shuffle=True)
        for seed in range(100):
            skf.random_state = seed
            for train_idx, test_idx in skf.split(X, y):
                model = cfg["model"]
                model.fit(X.iloc[train_idx], y.iloc[train_idx])
                preds = model.predict(X.iloc[test_idx])
                for idx, pred in zip(test_idx, preds):
                    sample_id = ids.iloc[idx]
                    pred_votes[sample_id].append(pred)
                    true_labels[sample_id] = y.iloc[idx]

        final_true = []
        final_pred = []
        for sid in pred_votes:
            try:
                final_true.append(true_labels[sid])
                final_pred.append(mode(pred_votes[sid]))
            except:
                continue

        cm = confusion_matrix(final_true, final_pred, labels=label_list)
        cm_norm = confusion_matrix(final_true, final_pred, labels=label_list, normalize="true")

        report = classification_report(final_true, final_pred, labels=label_list, target_names=label_list, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).T.loc[label_list, "precision":"support"].round(3)
        report_df["support"] = report_df["support"].astype(int)

        # Görselleştir
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), gridspec_kw={'width_ratios': [1, 1, 1.2]})

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_list, yticklabels=label_list, ax=axes[0])
        axes[0].set_title("Confusion Matrix")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")

        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", xticklabels=label_list, yticklabels=label_list, ax=axes[1])
        axes[1].set_title("Normalized Confusion Matrix")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Actual")

        axes[2].axis('off')
        tbl = axes[2].table(cellText=report_df.reset_index().values,
                            colLabels=report_df.reset_index().columns,
                            cellLoc='center', loc='center')
        tbl.scale(1, 2)
        tbl.set_fontsize(10)
        axes[2].set_title("Classification Report", pad=20)

        plt.suptitle(f"{name} ({tag}) – Majority Vote per Sample", fontsize=14)
        plt.tight_layout()
        if len(label_list) == 2:
            neg_class, pos_class = label_list
            annotation_text = f"*Negatif Sınıf: {neg_class}   |   Pozitif Sınıf: {pos_class}"
        else:
            annotation_text = "*Sınıflar: " + ", ".join(label_list)

        plt.gcf().text(0.95, 0.02, annotation_text, ha='right', va='bottom', fontsize=9, alpha=0.7)

        save_base = f"sample_confusion_{name.replace('/', '_').replace(' ', '_')}_{tag}"
        plt.savefig(f"{save_base}.png", dpi=300)
        report_df.to_csv(f"{save_base}.csv")
        plt.show()
        print(f" Kaydedildi: {save_base}.png ve .csv")
