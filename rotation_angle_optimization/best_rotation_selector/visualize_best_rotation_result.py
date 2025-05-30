import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 📥 En iyi sonuçları içeren CSV'yi yükle
csv_path = "./CSV/BestRotatingAngle/rotation_classification_results.csv"
df = pd.read_csv(csv_path)

# 🔎 Her sınıflandırma için en yüksek accuracy'ye sahip satırları al
best = df.loc[df.groupby("classification")["accuracy"].idxmax()].reset_index(drop=True)

# 📝 Her sınıflandırma için manuel model isimleri (CSV'deki sıraya göre)
best["model"] = [
    "ExtraTrees",            # HDL_vs_CONTROL
    "LogisticRegression",    # HDL_vs_LDL
    "ExtraTrees",            # HDL_vs_LDL_vs_CONTROL
    "RandomForest"           # LDL_vs_CONTROL
]

best["angle"] = [
    "60",
    "60",
    "60",
    "45"
]

#  Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=best, x="classification", y="accuracy", palette="viridis")

# 🏷️ Her bar üzerine model, açı ve accuracy yaz
for i, row in best.iterrows():
    plt.text(i,
             row["accuracy"] + 0.005,
             f"{row['model']}  |  {row['angle']}°\n{row['rotate_start_angle']}° | {row['accuracy']:.2f}",
             ha='center', va='bottom', fontsize=9)

plt.title("Her Sınıflandırma için En İyi Accuracy, Açı, Model ve Rotate Açısı")
plt.ylabel("Accuracy")
plt.xlabel("Classification")
plt.ylim(0.7, 1.05)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("BestResult.png", dpi=300)
plt.show()
