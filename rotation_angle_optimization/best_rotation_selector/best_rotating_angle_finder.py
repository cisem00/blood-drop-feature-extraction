import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

#  Sınıflandırma bilgileri (en iyi modellerle)
classification_config = {
    "HDL_vs_CONTROL": {
        "path": "./CSV/60DegRotating/FinalClassificationCSVs/HDL_vs_CONTROL",
        "model": ExtraTreesClassifier()
    },
    "HDL_vs_LDL": {
        "path": "./CSV//60DegRotating/FinalClassificationCSVs/HDL_vs_LDL",
        "model": LogisticRegression()
    },
    "HDL_vs_LDL_vs_CONTROL": {
        "path": "./CSV//60DegRotating/FinalClassificationCSVs/HDL_vs_LDL_vs_CONTROL",
        "model": ExtraTreesClassifier()
    },
    "LDL_vs_CONTROL": {
        "path": "./CSV//45DegRotating/FinalClassificationCSVs/LDL_vs_CONTROL",
        "model": RandomForestClassifier()
    }
}

results = []

print("\n🚀 Rotate Açıları için Sınıflandırma Performansları Karşılaştırılıyor...\n")

for class_name, config in classification_config.items():
    path = config["path"]
    model = config["model"]
    print(f"{class_name} klasörü işleniyor: {path}")

    for filename in sorted(os.listdir(path)):
        if not filename.endswith(".csv"):
            continue

        try:
            angle = int(filename.split("_")[1][:3]) 
            if angle < 10 or angle > 50:
                 continue
        except ValueError:
            continue

        csv_path = os.path.join(path, filename)
        df = pd.read_csv(csv_path)
        if df.empty or "label" not in df.columns:
            continue

        X = df[["radial_mean", "theta_mean", "radial_fft", "theta_fft"]]
        y = df["label"]

        acc_scores = []
        f1_scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for _ in range(20):  # 5 x 20 = 100 iterasyon
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc_scores.append(accuracy_score(y_test, y_pred))
                f1_scores.append(f1_score(y_test, y_pred, average="macro"))

        results.append({
            "classification": class_name,
            "rotate_start_angle": angle,
            "accuracy": np.mean(acc_scores),
            "f1_score": np.mean(f1_scores)
        })

#  Sonuçları DataFrame'e çevir
results_df = pd.DataFrame(results)
results_df.to_csv("rotation_classification_results.csv", index=False)
print("\n📅 Sonuçlar CSV olarak kaydedildi: rotation_classification_results.csv")

#  Accuracy grafiği
plt.figure(figsize=(14, 6))
sns.lineplot(
    data=results_df,
    x="rotate_start_angle",
    y="accuracy",
    hue="classification",
    marker="o"
)
plt.title("Rotate Açısına Göre Accuracy (En İyi Model ile)")
plt.xlabel("Rotate Başlangıç Açısı (Derece)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig("rotation_accuracy_plot.png")
print("Accuracy grafiği kaydedildi: rotation_accuracy_plot.png")
