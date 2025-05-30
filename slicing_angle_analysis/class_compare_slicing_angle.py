import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

#  Feature klasörü
base_dir = "./CSV/10-90FindingBestAngle/FeatureCSVs"
classification_sets = {
    "HDL_vs_LDL": os.path.join(base_dir, "HDL_vs_LDL"),
    "HDL_vs_CONTROL": os.path.join(base_dir, "HDL_vs_CONTROL"),
    "LDL_vs_CONTROL": os.path.join(base_dir, "LDL_vs_CONTROL"),
    "HDL_vs_LDL_vs_CONTROL": os.path.join(base_dir, "HDL_vs_LDL_vs_CONTROL")
}

# 🤖 Modeller
models = {
    "RandomForest": RandomForestClassifier(n_jobs=-1),
    "ExtraTrees": ExtraTreesClassifier(n_jobs=-1),
    "AdaBoost": AdaBoostClassifier(),
    "GradientBoost": GradientBoostingClassifier(),
    "LogisticRegression": LogisticRegression(),
    "NaiveBayes": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "SVC": SVC(),
    "DecisionTree": DecisionTreeClassifier(),
    "XGBoost": XGBClassifier(),
    "MLP": MLPClassifier(max_iter=1000)
}

results = []

print("\n🚀 Sınıflandırma karşılaştırması başlıyor...\n")

#  Her classification seti ve dosyası için
for class_name, folder in classification_sets.items():
    print(f"\n📂 Sınıflandırma seti: {class_name}")
    for file in sorted(os.listdir(folder)):
        if not file.endswith(".csv"):
            continue

        angle = int(file.split("angle")[1].replace(".csv", "").replace("_", ""))
        print(f"  🔎 İşleniyor: {file} ➤ Açı: {angle}°")

        df = pd.read_csv(os.path.join(folder, file))
        if df.empty:
            print("   Dosya boş, atlanıyor.")
            continue

        X = df[["radial_mean", "theta_mean", "radial_fft", "theta_fft"]]
        y = df["label"]

        #  LabelEncoder ile tüm modellerle uyumlu hale getir
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        for model_name, model in models.items():
            acc_scores = []
            f1_scores = []
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            for _ in range(20): 
                for train_idx, test_idx in skf.split(X, y_encoded):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc_scores.append(accuracy_score(y_test, y_pred))
                    f1_scores.append(f1_score(y_test, y_pred, average="macro"))

            results.append({
                "classification": class_name,
                "angle": angle,
                "model": model_name,
                "mean_accuracy": np.mean(acc_scores),
                "mean_f1": np.mean(f1_scores)
            })
            print(f"     {model_name} tamamlandı | Mean F1: {np.mean(f1_scores):.4f}")

#  Sonuçları kaydet
results_df = pd.DataFrame(results)
results_csv_path = os.path.join(base_dir, "classification_compare_results.csv")
results_df.to_csv(results_csv_path, index=False)
print(f"\n💾 Tüm sonuçlar kaydedildi: {results_csv_path}")

#  Her sınıflandırma için en iyi model
best_per_class = results_df.sort_values("mean_f1", ascending=False).groupby("classification").first()
best_path = os.path.join(base_dir, "best_model_per_classification.csv")
best_per_class.to_csv(best_path)
print(f"🏅 En iyi modeller klasör bazında kaydedildi: {best_path}")

# 🥇 En iyi genel model
overall_best = results_df.loc[results_df["mean_f1"].idxmax()]
print("\n🏆 En iyi genel sonuç:")
print(overall_best)

#  Heatmap oluştur
plt.figure(figsize=(14, 6))
pivot = results_df.pivot_table(index="model", columns="angle", values="mean_f1", aggfunc="mean")
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
plt.title("Model F1-Score Karşılaştırması (Açı Bazlı)")
plt.tight_layout()

heatmap_path = os.path.join(base_dir, "classification_f1_heatmap.png")
plt.savefig(heatmap_path)
print(f"🖼 Heatmap kaydedildi: {heatmap_path}")
