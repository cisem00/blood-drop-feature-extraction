
# Rotasyon ve Slicing Analizi ile Görüntü İşleme

Bu proje, kan damlası görüntüleri üzerinde polar koordinat dönüşümü, açısal slicing ve sınıflandırma analizi gibi görüntü işleme tekniklerini kullanarak biyolojik parametrelerin tahmin edilmesine yönelik bir yaklaşım sunar. Python ve OpenCV kütüphanesiyle geliştirilmiş olan bu sistem, kan damlası görüntülerinden çıkarılan özelliklerin biyolojik analizlerde kullanılabilmesini hedefler.

---

## Projenin Amacı
- Kan damlası görüntülerinin farklı açılardan incelenmesi: Görüntüler, hem rotasyon hem de slicing teknikleriyle farklı açılardan analiz edilerek detaylı bilgi elde edilmesi sağlanır.
- Polar koordinat dönüşümü ve açısal slicing ile özellik çıkarımı yapılması: Görüntülerde belirlenen merkez etrafında polar koordinat sistemine dönüştürme ve belirli açılarda dilimleme (slicing) işlemleri uygulanarak, her açı için ayırt edici özellikler (ör. şekil, kontur, renk, yoğunluk) çıkarılır.
- Çıkarılan özellikler ile biyolojik parametre tahmini ve sınıflandırma yapılması: Elde edilen özellikler kullanılarak, HDL (yüksek yoğunluklu lipoprotein), LDL (düşük yoğunluklu lipoprotein) ve kontrol grupları arasında sınıflandırma algoritmaları uygulanır. Böylece görüntülerden biyolojik parametre tahmini gerçekleştirilir.

---

## Proje Yapısı
```
KOD/
├── rotation_angle_optimization/
│   ├── rotating_angle_analysis_45deg/
│   │   ├── merge_all_csvs_for_45deg.py
│   │   ├── rotate_0_to_360_at_45deg.py
│   ├── rotating_angle_analysis_60deg/
│   │   ├── merge_all_csvs_for_60deg.py
│   │   ├── rotate_0_to_360_at_60deg.py
│   ├── best_rotation_selector/
│   │   ├── visualize_best_rotation_result.py
│   │   ├── best_rotating_angle_finder.py
├── original_dataset_classification/
│   ├── feature_extraction_and_visualization.py
│   ├── evaluate_classifiers_with_confusion.py
│   ├── original_dataset_model_comparison.py
├── slicing_angle_analysis/
│   ├── class_compare_slicing_angle.py
│   ├── slice_feature_extraction.py
├── visualize_best_model_results/
│   ├── compare_best_model_confusions.py
│   ├── confusion_matrix_with_sample_info.py
├── data/ (kullanıcı tarafından oluşturulmalı)
│   ├── DataBase/
│   ├── Cropped_Sets_45deg/
│   ├── Cropped_Sets_60deg/
├── CSV/ (kullanıcı tarafından oluşturulmalı)
│   ├── FeatureCSVs/
│   ├── FinalClassificationCSVs/
│   ├── BestRotatingAngle/
│   │   ├── rotation_classification_results.csv
│   ├── features_for_binary_classification_HDL_CONTROL.csv
│   ├── features_for_binary_classification.csv
│   ├── features_for_binary_classification_LDL_CONTROL.csv
│   ├── features_for_multi_classification.csv
│   ├── 60DegRotating/
│   │   ├── FinalClassificationCSVs/
│   │   │   ├── HDL_vs_CONTROL/Set_030deg.csv
│   │   │   ├── HDL_vs_LDL/Set_016deg.csv
│   │   │   ├── HDL_vs_LDL_vs_CONTROL/Set_025deg.csv
│   ├── 45DegRotating/
│   │   ├── FinalClassificationCSVs/
│   │   │   ├── LDL_vs_CONTROL/Set_046deg.csv
│   ├── classification_compare_results.csv
│   ├── best_model_per_classification.csv

```

---

## Gerekli Kurulumlar
Projenin çalışabilmesi için aşağıdaki Python kütüphaneleri gereklidir:
```bash
pip install numpy pandas opencv-python matplotlib scikit-learn scipy seaborn tqdm
```

---

## Kullanım Talimatları
1️⃣ **Veri klasörlerini oluşturun**:  
- Projenin kök dizininde `data/` klasörünü oluşturun ve altına aşağıdaki dizinleri yerleştirin:  
  - `DataBase/`: Ham kan damlası görüntüleri.  
  - `Cropped_Sets_45deg/`: 45° dilimleme (slicing) yapılmış görüntüler.  
  - `Cropped_Sets_60deg/`: 60° dilimleme (slicing) yapılmış görüntüler.  
- Projenin kök dizininde `CSV/` klasörünü oluşturun ve altına sonuçların ve çıkarılan özelliklerin kaydedileceği klasörleri oluşturun.  

2️⃣ **Kod dosyalarını çalıştırın**:  
Her analiz dosyası kendi alt klasöründe yer alır. Aşağıda örnek kullanım komutları verilmiştir:  
```bash
python KOD/rotation_angle_optimization/rotating_angle_analysis_45deg/merge_all_csvs_for_45deg.py
python KOD/rotation_angle_optimization/rotating_angle_analysis_60deg/merge_all_csvs_for_60deg.py
python KOD/slicing_angle_analysis/slice_feature_extraction.py
python KOD/original_dataset_classification/original_dataset_model_comparison.py


3️⃣ **Sonuçlar**:  
- Çıktı dosyaları ./CSV/ dizininde oluşturulur.
- Özellik CSV'leri FeatureCSVs/, sınıflandırma sonuçları ise FinalClassificationCSVs/ ve ilgili alt klasörlerde yer alacaktır.
- BestRotatingAngle/, classification_compare_results.csv, best_model_per_classification.csv gibi dosyalar da bu dizin içinde bulunur.

---

## Açıklamalar
- 📌 **Veri Yolları**: Tüm kodlar **göreli yollar** ile çalışır. Proje kök dizininden (`KOD/`) başlatılmalıdır.  
- 📌 **Fonksiyonlar**: Kod içinde fonksiyonlar `docstring` açıklamaları ile belgelenmiştir.  
- 📌 **Analiz Dosyaları**: Her alt klasörde slicing, rotasyon, ve model karşılaştırma için ayrı dosyalar bulunur.  

---

##  Katkıda Bulunma
Projenin gelişimine katkıda bulunmak için fork edip PR gönderebilirsiniz.  

---

## İletişim
Projeyle ilgili sorularınız için cisemkaplan07@gmail.com üzerinden iletişime geçebilirsiniz.

---
