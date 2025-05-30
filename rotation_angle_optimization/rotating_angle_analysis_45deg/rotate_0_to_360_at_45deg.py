import os
import cv2
import numpy as np
import math
import pandas as pd
from scipy.fft import fft

input_dir = "./data/DataBase"
output_base_dir = "./data/Cropped_Sets_45deg"
csv_output_dir = "./CSV/FeatureCSVs_Sets45"
os.makedirs(csv_output_dir, exist_ok=True)

fixed_angle = 45
total_slices_per_set = 8
set_start_angles = range(0, 60)  # 0–59

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

    # wrap around: örneğin 300–0 slice'ı → 300–360 + 0–0 gibi düşünülmeli
    if end_angle < start_angle:
        angles = list(np.linspace(start_angle, 360, num=50)) + list(np.linspace(0, end_angle, num=50))
    else:
        angles = np.linspace(start_angle, end_angle, num=100)

    for angle in angles:
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
    polar_img = polar_transform(binary, center=center, output_shape=(300, 720), flip_vertical=True)

    polar_norm = polar_img / 255.0
    radial_profile = np.mean(polar_norm, axis=1)
    theta_profile = np.mean(polar_norm, axis=0)

    radial_mean = np.mean(radial_profile)
    theta_mean = np.mean(theta_profile)
    radial_fft = np.mean(np.abs(fft(radial_profile))[1:10])
    theta_fft = np.mean(np.abs(fft(theta_profile))[1:10])

    return radial_mean, theta_mean, radial_fft, theta_fft

def generate_angle_pairs(start_angle, count, delta):
    """
    generate_angle_pairs fonksiyonu.

    Args:
        start_angle: Başlangıç açısı.
        count: Üretilecek çift sayısı.
        delta: Açı aralığı.

    Returns:
        Açı çiftleri listesi.
    """

    return [( (start_angle + i * delta) % 360, (start_angle + i * delta + delta) % 360 ) for i in range(count)]

def process_set(start_angle):
    """
    process_set fonksiyonu.

    Args:
        start_angle: Başlangıç açısı.

    Returns:
        İşlenen veri seti bilgileri.
    """

    angle_pairs = generate_angle_pairs(start_angle, total_slices_per_set, fixed_angle)
    set_label = f"{start_angle:03d}"
    set_folder = os.path.join(output_base_dir, f"Set_{set_label}deg")
    os.makedirs(set_folder, exist_ok=True)
    kernel = np.ones((3, 3), np.uint8)
    feature_data = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')) or '.ds_store' in file.lower():
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

            for i, (start, end) in enumerate(angle_pairs):
                mask = create_sector_mask(cropped.shape, center, radius, start, end)
                sector = cv2.bitwise_and(cropped, cropped, mask=mask)
                gray_sector = cv2.cvtColor(sector, cv2.COLOR_BGR2GRAY)
                coords = cv2.findNonZero(gray_sector)
                if coords is None:
                    continue
                x_s, y_s, w_s, h_s = cv2.boundingRect(coords)
                final_crop = sector[y_s:y_s+h_s, x_s:x_s+w_s]

                base_name = os.path.splitext(os.path.basename(file))[0]
                rel_dir = os.path.relpath(path, input_dir)
                slice_name = f"{base_name}_slice_{start:03d}_{end:03d}.png"
                out_subdir = os.path.join(set_folder, os.path.dirname(rel_dir))
                os.makedirs(out_subdir, exist_ok=True)
                save_path = os.path.join(out_subdir, slice_name)
                cv2.imwrite(save_path, final_crop)

                r_mean, t_mean, r_fft, t_fft = extract_radial_theta_features(final_crop, (final_crop.shape[1]//2, final_crop.shape[0]//2))
                label = "HDL" if any(x in rel_dir for x in ["2A", "2B", "2D", "2E", "2F"]) else "LDL"
                feature_data.append({
                    "filename": slice_name,
                    "source_image": file,
                    "start_angle": start,
                    "end_angle": end,
                    "radial_mean": r_mean,
                    "theta_mean": t_mean,
                    "radial_fft": r_fft,
                    "theta_fft": t_fft,
                    "label": label
                })

    df = pd.DataFrame(feature_data)
    csv_path = os.path.join(csv_output_dir, f"features_set_{set_label}deg.csv")
    df.to_csv(csv_path, index=False)
    print(f" Set {set_label}° → {csv_path} yazıldı")

#  Tüm setleri üret
for set_start in set_start_angles:
    process_set(set_start)

print("🎉 Tüm slice setleri başarıyla tamamlandı.")
