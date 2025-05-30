
import os
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft
import matplotlib.pyplot as plt
import random

# Özellik vektörleri ve etiket listeleri
X, y, y_str, filenames, all_images = [], [], [], [], []

# Grup ve etiket eşlemeleri
group_map = {
    "2A": "CONTROL", 
    "2B": "HDL", "2D": "HDL", "2E": "HDL", "2F": "HDL",
    "3A": "CONTROL",
    "3B": "LDL", "3D": "LDL", "3E": "LDL", "3F": "LDL",
}
label_map = {"CONTROL": 0, "HDL": 1, "LDL": 2}


def polar_transform(binary_image, center=None, output_shape=(300, 720), flip_vertical=True):
    """
    polar_transform fonksiyonu.

    Args:
        binary_image: Giriş binary görüntü (tek kanal).
        center: Dönüşüm merkezi (x, y). Varsayılan None.
        output_shape: Çıktı boyutu (yükseklik, genişlik).
        flip_vertical: Dikey eksende ters çevirme bayrağı.

    Returns:
        Polar dönüşmüş görüntü (tek kanal).
    """
    if center is None:
        center = (binary_image.shape[1] // 2, binary_image.shape[0] // 2)
    max_radius = min(center[0], center[1])
    polar_img = np.zeros(output_shape, dtype=np.uint8)

    r_vals = np.linspace(0, max_radius, output_shape[0])
    theta_vals = np.linspace(0, 2 * np.pi, output_shape[1])

    for i, r in enumerate(r_vals):
        for j, theta in enumerate(theta_vals):
            x = int(center[0] + r * np.cos(theta))
            y = int(center[1] + r * np.sin(theta))
            if 0 <= x < binary_image.shape[1] and 0 <= y < binary_image.shape[0]:
                polar_img[i, j] = binary_image[y, x]

    if flip_vertical:
        polar_img = np.flipud(polar_img)

    return polar_img


# Görsellerin işlendiği ana klasör
root_folder = "./data/DataBase"
img_counter = 0

# Görselleri işleme
for dirpath, _, filenames_in_folder in os.walk(root_folder):
    for file in filenames_in_folder:
        if file.lower().endswith(".jpg"):
            full_path = os.path.join(dirpath, file)
            parts = full_path.split(os.sep)
            try:
                subgroup = parts[-3]
                label_str = group_map.get(subgroup)
                if not label_str:
                    continue
            except IndexError:
                continue

            gray = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue

            img_counter += 1
            print(f"[{img_counter}] İşleniyor: {file}")

            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cleaned = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

            polar_img = polar_transform(cleaned, flip_vertical=True)

            theta_profile = np.mean(polar_img, axis=0) / 255.0
            theta_profile = gaussian_filter1d(theta_profile, sigma=2)

            radial_profile = np.mean(polar_img, axis=1) / 255.0
            radial_profile = gaussian_filter1d(radial_profile, sigma=2)

            theta_mean = np.mean(theta_profile)
            radial_mean = np.mean(radial_profile)

            theta_fft = np.abs(fft(theta_profile - np.mean(theta_profile)))[:10]
            radial_fft = np.abs(fft(radial_profile - np.mean(radial_profile)))[:10]

            theta_fft_feature = np.max(theta_fft)
            radial_fft_feature = np.max(radial_fft)

            X.append([radial_mean, theta_mean, radial_fft_feature, theta_fft_feature])
            y.append(label_map[label_str])
            y_str.append(label_str)
            filenames.append(file)
            all_images.append((full_path, file, subgroup, label_str))


# Özellikleri CSV dosyasına kaydet
df = pd.DataFrame(X, columns=["radial_mean", "theta_mean", "radial_fft", "theta_fft"])
df["label_str"] = y_str  
df["label"] = y           
df["filename"] = filenames
df.to_csv("features_for_multi_classification.csv", index=False)

print("Özellikler başarıyla kaydedildi (CONTROL=0, HDL=1, LDL=2).")


# Rastgele bir görsel seçilerek örnek görselleştirme yapılır
random_img_path, random_filename, random_subgroup, random_label_str = random.choice(all_images)

original = cv2.imread(random_img_path)
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
_, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)
otsu_cleaned = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)

polar_img = polar_transform(otsu_cleaned, flip_vertical=True)

contours, _ = cv2.findContours(otsu_cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(f"Kontur sayısı: {len(contours)}")

overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
cv2.drawContours(overlay, contours, -1, (0, 0, 255), thickness=3)

full_title = f"{random_filename} ({random_subgroup}) → {random_label_str}"

fig, axs = plt.subplots(1, 5, figsize=(25, 5))
axs[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
axs[0].set_title("Original")

axs[1].imshow(gray, cmap='gray')
axs[1].set_title("Grayscale")

axs[2].imshow(otsu, cmap='gray')
axs[2].set_title("OTSU Thresholding")

axs[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
axs[3].set_title("Red Contours")

axs[4].imshow(polar_img, cmap='gray')
axs[4].set_title("Polar Image")

for ax in axs:
    ax.axis('off')

plt.suptitle(full_title, fontsize=14)
plt.tight_layout()
plt.show()
