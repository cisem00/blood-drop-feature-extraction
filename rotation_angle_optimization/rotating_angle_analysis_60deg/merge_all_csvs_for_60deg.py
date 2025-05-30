import os
import cv2
import numpy as np
import pandas as pd
from scipy.fft import fft

input_root = "./data/45DegRotating/Cropped_Sets_45deg"
output_root = "./CSV/FinalClassificationCSVs"
os.makedirs(output_root, exist_ok=True)

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

#  Feature çıkarımı
def extract_polar_features(image):
    """
    extract_polar_features fonksiyonu.

    Args:
        image: Giriş görüntüsü.

    Returns:
        Görüntüden çıkarılan polar özellikler.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    center = (gray.shape[1] // 2, gray.shape[0] // 2)
    max_radius = min(center[0], center[1], gray.shape[1] - center[0], gray.shape[0] - center[1])
    polar_img = polar_transform(binary, center=center, output_shape=(300, 720), flip_vertical=True)
    polar_norm = polar_img / 255.0
    radial_profile = np.mean(polar_norm, axis=1)
    theta_profile = np.mean(polar_norm, axis=0)
    return np.mean(radial_profile), np.mean(theta_profile), np.mean(np.abs(fft(radial_profile))[1:10]), np.mean(np.abs(fft(theta_profile))[1:10])

#  Label eşlemeleri
map_hdl_ldl = {
    "2A": "HDL", "2B": "HDL", "2D": "HDL", "2E": "HDL", "2F": "HDL",
    "3A": "LDL", "3B": "LDL", "3D": "LDL", "3E": "LDL", "3F": "LDL"
}
map_hdl_ctrl = {
    "2A": "CONTROL", "2B": "HDL", "2D": "HDL", "2E": "HDL", "2F": "HDL"
}
map_ldl_ctrl = {
    "3A": "CONTROL", "3B": "LDL", "3D": "LDL", "3E": "LDL", "3F": "LDL"
}
map_all = {
    "2A": "CONTROL", "3A": "CONTROL",
    "2B": "HDL", "2D": "HDL", "2E": "HDL", "2F": "HDL",
    "3B": "LDL", "3D": "LDL", "3E": "LDL", "3F": "LDL"
}

classification_configs = [
    ("HDL_vs_LDL", map_hdl_ldl),
    ("HDL_vs_CONTROL", map_hdl_ctrl),
    ("LDL_vs_CONTROL", map_ldl_ctrl),
    ("HDL_vs_LDL_vs_CONTROL", map_all)
]

#  Her açıyı gez (Set_000deg ... Set_059deg)
for set_folder in sorted(os.listdir(input_root)):
    set_path = os.path.join(input_root, set_folder)
    if not os.path.isdir(set_path) or not set_folder.startswith("Set_"):
        continue

    print(f"\n {set_folder} işleniyor...")
    feature_rows = []

    for root, _, files in os.walk(set_path):
        for file in files:
            if not file.endswith(".png"):
                continue

            full_path = os.path.join(root, file)
            img = cv2.imread(full_path)
            if img is None:
                continue

            parts = full_path.split(os.sep)
            try:
                group = parts[-3]  # örn. 2A, 3B
            except:
                continue

            if group not in map_all:
                continue

            r_mean, t_mean, r_fft, t_fft = extract_polar_features(img)

            feature_rows.append({
                "filename": file,
                "source_image": os.path.join(parts[-4], parts[-3], parts[-2], parts[-1]),
                "group": group,
                "radial_mean": r_mean,
                "theta_mean": t_mean,
                "radial_fft": r_fft,
                "theta_fft": t_fft
            })

    df = pd.DataFrame(feature_rows)

    for class_name, class_map in classification_configs:
        df_sub = df[df["group"].isin(class_map)].copy()
        df_sub["label"] = df_sub["group"].map(class_map)

        class_dir = os.path.join(output_root, class_name)
        os.makedirs(class_dir, exist_ok=True)

        csv_path = os.path.join(class_dir, f"{set_folder}.csv")
        df_sub.to_csv(csv_path, index=False)

        print(f" {class_name} klasörü → {csv_path} kaydedildi.")

print("\n🎉 Tüm açılar ve tüm sınıflandırmalar için CSV'ler oluşturuldu.")
