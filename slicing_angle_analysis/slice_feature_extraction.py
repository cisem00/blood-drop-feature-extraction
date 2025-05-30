
import os
import cv2
import numpy as np
import math
import pandas as pd
from scipy.fft import fft

input_dir = "./data/DataBase"
output_base_dir = "./data/Cropped_Drops_Sliced"
csv_output_dir = "./CSV/FeatureCSVs"
os.makedirs(csv_output_dir, exist_ok=True)

angle_list = [10, 15, 30, 45, 60, 90]

def create_sector_mask(shape, center, radius, start_angle, end_angle):
    """
    create_sector_mask fonksiyonu.

    Args:
        shape: Görüntü şekli.
        center: Maske merkezi.
        radius: Yarıçap.
        start_angle: Başlangıç açısı (derece).
        end_angle: Bitiş açısı (derece).

    Returns:
        Belirli açıda maske matrisi.
    """

    mask = np.zeros(shape[:2], dtype=np.uint8)
    pts = [center]
    for angle in np.linspace(start_angle, end_angle, num=30):
        x = int(center[0] + radius * math.cos(math.radians(angle)))
        y = int(center[1] + radius * math.sin(math.radians(angle)))
        pts.append((x, y))
    pts = np.array([pts], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask

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

    return np.flipud(polar_img) if flip_vertical else polar_img

def extract_radial_theta_features(image, center):
    """
    extract_radial_theta_features fonksiyonu.

    Args:
        image: Giriş görüntüsü.
        center: Görüntü merkezi.

    Returns:
        Görüntüden çıkarılan açısal ve radyal özellikler.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    max_radius = int(min(center[0], center[1], image.shape[1] - center[0], image.shape[0] - center[1]))
    polar_img = polar_transform(binary, center=center, output_shape=(300, 720), flip_vertical=True)


    polar_norm = polar_img / 255.0
    radial_profile = np.mean(polar_norm, axis=1)
    theta_profile = np.mean(polar_norm, axis=0)

    radial_mean = np.mean(radial_profile)
    theta_mean = np.mean(theta_profile)
    radial_fft = np.mean(np.abs(fft(radial_profile))[1:10])
    theta_fft = np.mean(np.abs(fft(theta_profile))[1:10])

    return radial_mean, theta_mean, radial_fft, theta_fft

def process_images_with_angle(angle_step):
    """
    process_images_with_angle fonksiyonu.

    Args:
        angle_step: Açısal adım.

    Returns:
        İşlenen görüntülerin bilgileri.
    """

    output_dir = f"{output_base_dir}{angle_step}"
    os.makedirs(output_dir, exist_ok=True)
    kernel = np.ones((3, 3), np.uint8)
    feature_data = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            if file.lower().endswith('.cr2') or '.ds_store' in file.lower():
                continue

            path = os.path.join(root, file)
            img = cv2.imread(path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cleaned = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            large_enough = [c for c in contours if cv2.contourArea(c) > 500]
            if not large_enough:
                continue

            largest_contour = max(large_enough, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            if w == 0 or h == 0:
                continue

            cropped = img[y:y+h, x:x+w]
            gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray_crop, (5, 5), 0)
            _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cleaned = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            center = (cx, cy)

            h_c, w_c = cropped.shape[:2]
            radius = min(cx, cy, w_c - cx, h_c - cy)

            for i in range(360 // angle_step):
                start_angle = i * angle_step
                end_angle = (i + 1) * angle_step
                mask = create_sector_mask(cropped.shape, center, radius, start_angle, end_angle)
                sector = cv2.bitwise_and(cropped, cropped, mask=mask)

                gray_sector = cv2.cvtColor(sector, cv2.COLOR_BGR2GRAY)
                coords = cv2.findNonZero(gray_sector)
                if coords is None:
                    continue
                x_s, y_s, w_s, h_s = cv2.boundingRect(coords)
                final_crop = sector[y_s:y_s+h_s, x_s:x_s+w_s]

                base_name = os.path.splitext(os.path.basename(file))[0]
                rel_dir = os.path.relpath(path, input_dir)
                output_subdir = os.path.join(output_dir, os.path.dirname(rel_dir))
                os.makedirs(output_subdir, exist_ok=True)

                slice_name = f"{base_name}_slice_{i+1}.png"
                save_path = os.path.join(output_subdir, slice_name)
                cv2.imwrite(save_path, final_crop)

                r_mean, t_mean, r_fft, t_fft = extract_radial_theta_features(final_crop, (final_crop.shape[1]//2, final_crop.shape[0]//2))
                label = "HDL" if any(x in rel_dir for x in ["2A", "2B", "2D", "2E", "2F"]) else "LDL"
                feature_data.append({
                    "filename": slice_name,
                    "source_image": os.path.join(os.path.dirname(rel_dir), file),
                    "angle": angle_step,
                    "radial_mean": r_mean,
                    "theta_mean": t_mean,
                    "radial_fft": r_fft,
                    "theta_fft": t_fft,
                    "label": label
                })

        print(f" {angle_step}° ➤ {file} ➤ slice ve feature tamamlandı.")

    df = pd.DataFrame(feature_data)
    csv_path = os.path.join(csv_output_dir, f"features_angle_{angle_step}.csv")
    df.to_csv(csv_path, index=False)
    print(f" CSV kaydedildi: {csv_path}")

#  Tüm açılar için çalıştır
for angle in angle_list:
    print(f"🚀 {angle}° ile tüm işlemler başlatılıyor...")
    process_images_with_angle(angle)

print("🎉 Tüm açılar için slice + feature çıkarımı tamamlandı.")

#  Ek sınıflandırma CSV'leri oluştur
def generate_additional_csvs():
    """
    generate_additional_csvs fonksiyonu.

    Args:
        Yok.

    Returns:
        Ek CSV dosyaları üretir.
    """

    hc_dir = os.path.join(csv_output_dir, "HDL_vs_CONTROL")
    lc_dir = os.path.join(csv_output_dir, "LDL_vs_CONTROL")
    all_dir = os.path.join(csv_output_dir, "HDL_vs_LDL_vs_CONTROL")
    os.makedirs(hc_dir, exist_ok=True)
    os.makedirs(lc_dir, exist_ok=True)
    os.makedirs(all_dir, exist_ok=True)

    for angle in angle_list:
        path = os.path.join(csv_output_dir, f"features_angle_{angle}.csv")
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)

        df_hc = df[df["source_image"].str.contains("2A|2B|2D|2E|2F", na=False)].copy()
        df_hc["label"] = df_hc["source_image"].apply(lambda x: "CONTROL" if "2A" in x else "HDL")
        df_hc.to_csv(os.path.join(hc_dir, f"features_HDL_vs_CONTROL_angle{angle}.csv"), index=False)

        df_lc = df[df["source_image"].str.contains("3A|3B|3D|3E|3F", na=False)].copy()
        df_lc["label"] = df_lc["source_image"].apply(lambda x: "CONTROL" if "3A" in x else "LDL")
        df_lc.to_csv(os.path.join(lc_dir, f"features_LDL_vs_CONTROL_angle{angle}.csv"), index=False)

        df_all = df[df["source_image"].str.contains("2A|2B|2D|2E|2F|3A|3B|3D|3E|3F", na=False)].copy()
        df_all["label"] = df_all["source_image"].apply(
            lambda x: "CONTROL" if "2A" in x or "3A" in x else
                      "HDL" if "2B" in x or "2D" in x or "2E" in x or "2F" in x else "LDL"
        )
        df_all.to_csv(os.path.join(all_dir, f"features_HDL_vs_LDL_vs_CONTROL_angle{angle}.csv"), index=False)

        print(f" {angle}° ➤ Ek sınıflandırmalar oluşturuldu")

generate_additional_csvs()
