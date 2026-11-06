# Blood Drop Feature Extraction

Image processing and classification pipeline for blood drop images, built for my BSc thesis in Computer Engineering at Galatasaray University (supervised by Prof. Mouloud Adel, May 2025).

The goal is to classify dried blood drop images into **HDL**, **LDL** and **control** groups using only the shape and texture of the drop, as a possible low-cost and non-invasive alternative to lab tests.

## How it works

1. **Preprocessing:** grayscale conversion, Otsu thresholding and morphological opening to separate the drop from the background. The drop's center is found with image moments.
2. **Polar transform:** each drop is converted to polar coordinates around its center with a custom function, so it can be analyzed both radially (center to edge) and angularly (0° to 360°).
3. **Angular slicing and rotation:** each drop is cut into angular slices (10°, 15°, 30°, 45°, 60°, 90°), and the starting angle is shifted from 0° to 59°. This produces many versions of each image and works as data augmentation.
4. **Feature extraction:** four features per slice: mean radial intensity, mean angular intensity, and the average of the first 10 Fourier components of each profile.
5. **Classification:** 10 scikit-learn classifiers (Random Forest, Extra Trees, Logistic Regression, SVC, MLP and others) are compared on four tasks, using repeated stratified 5-fold cross-validation.

## Results

The dataset contains 46 images. Accuracy before and after slicing and rotation optimization:

| Task | Original | Optimized | Best setup |
| --- | --- | --- | --- |
| HDL vs CONTROL | 0.76 | 0.98 | Extra Trees, 60° slices, 30° rotation |
| HDL vs LDL | 0.75 | 0.82 | Logistic Regression, 60° slices, 16° rotation |
| LDL vs CONTROL | 0.97 | 1.00 | Random Forest, 45° slices, 46° rotation |
| HDL vs LDL vs CONTROL | 0.67 | 0.82 | Extra Trees, 60° slices, 25° rotation |

Because the dataset is small, these results should be confirmed on a larger dataset.

## Project structure

```
rotation_angle_optimization/
├── rotating_angle_analysis_45deg/
│   ├── rotate_0_to_360_at_45deg.py        # 45° slices for each starting angle
│   └── merge_all_csvs_for_45deg.py
├── rotating_angle_analysis_60deg/
│   ├── rotate_0_to_360_at_60deg.py        # 60° slices for each starting angle
│   └── merge_all_csvs_for_60deg.py
└── best_rotation_selector/
    ├── best_rotating_angle_finder.py      # finds the best starting angle per task
    └── visualize_best_rotation_result.py
original_dataset_classification/
├── feature_extraction_and_visualization.py   # features from the original images
├── original_dataset_model_comparison.py      # compares classifiers (baseline)
└── evaluate_classifiers_with_confusion.py
slicing_angle_analysis/
├── slice_feature_extraction.py        # features for each slice angle
└── class_compare_slicing_angle.py
visualize_best_model_results/
├── compare_best_model_confusions.py
└── confusion_matrix_with_sample_info.py
```

## Setup

```bash
pip install numpy pandas opencv-python matplotlib scikit-learn scipy seaborn tqdm
```

The dataset is not included in this repository. To run the code, create these folders in the project root:

- `data/DataBase/`: raw blood drop images
- `CSV/`: extracted features and classification results are saved here

## Usage

Run the scripts from the project root, for example:

```bash
python original_dataset_classification/feature_extraction_and_visualization.py
python original_dataset_classification/original_dataset_model_comparison.py
python slicing_angle_analysis/slice_feature_extraction.py
python rotation_angle_optimization/rotating_angle_analysis_45deg/rotate_0_to_360_at_45deg.py
python rotation_angle_optimization/rotating_angle_analysis_45deg/merge_all_csvs_for_45deg.py
python rotation_angle_optimization/best_rotation_selector/best_rotating_angle_finder.py
```

Output CSVs and plots are written under `CSV/`.

## Contact

Habibe Çisem Kaplan · cisemkaplan07@gmail.com
