import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import albumentations as A
from ultralytics import YOLO

CONDITIONS = {
    "Orijinal": None,
    "Sisli": A.Compose([
        A.RandomFog(fog_coef_range=(0.3, 0.6), alpha_coef=0.1, p=1.0)
    ]),
    "Karanlik": A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.5, -0.3), contrast_limit=(-0.2, 0.0), p=1.0)
    ]),
    "Yagmurlu": A.Compose([
        A.RandomRain(drop_length=15, drop_width=1, drop_color=(200, 200, 200), blur_value=3, p=1.0)
    ]),
    "Gurultulu": A.Compose([
        A.GaussNoise(std_range=(0.2, 0.44), p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=1.0)
    ])
}

def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"Model dosyasi bulunamadi: {model_path}")
        return YOLO("yolov8n-cls.pt")
    return YOLO(model_path)

def evaluate_yolo_validation(model, data_path, output_dir="runs/evaluation"):
    os.makedirs(output_dir, exist_ok=True)
    metrics = model.val(data=data_path, project=output_dir, name="test_results")
    return metrics

def evaluate_under_conditions(model, test_images_dir, output_dir="runs/evaluation"):
    os.makedirs(output_dir, exist_ok=True)
    condition_results = {}

    valid_exts = ('.png', '.jpg', '.jpeg', '.ppm')
    image_paths = []
    for root, _, files in os.walk(test_images_dir):
        for f in files:
            if f.lower().endswith(valid_exts):
                image_paths.append(os.path.join(root, f))

    if not image_paths:
        print(f"Test klasorunde resim bulunamadi: {test_images_dir}")
        return {}

    for cond_name, transform in CONDITIONS.items():
        total_count = 0
        for img_path in image_paths[:200]:
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            if transform is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                augmented = transform(image=img_rgb)
                eval_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
            else:
                eval_img = img

            results = model.predict(eval_img, verbose=False)
            total_count += 1

        condition_results[cond_name] = total_count

    return condition_results

def plot_condition_comparison(results_dict, output_path="runs/evaluation/kosul_karsilastirma.png"):
    plt.figure(figsize=(9, 5))
    conditions = list(results_dict.keys())
    scores = list(results_dict.values())

    bars = plt.bar(conditions, scores, color=['#2ecc71', '#95a5a6', '#34495e', '#3498db', '#e67e22'])
    plt.title("Cevre Kosullarinda Model Karsilastirmasi")
    plt.xlabel("Kosul")
    plt.ylabel("Ornek Sayisi / Basari")
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{int(yval)}", ha='center', va='bottom')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

def plot_confusion_matrix_custom(y_true, y_pred, class_names, output_path="runs/evaluation/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Tahmin")
    plt.ylabel("Gercek")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/yolo_best.pt")
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default="runs/evaluation")
    args = parser.parse_args()

    model = load_model(args.model)
