import os
import shutil
import glob
import cv2
import pandas as pd
import albumentations as A
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 1. تعريف التحويلات البيئية مع مراعاة تحديث إحداثيات Bounding Boxes
augment_transform = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.35, 0.25), contrast_limit=(-0.3, 0.3), p=0.7),
    A.RandomFog(fog_coef_lower=0.15, fog_coef_upper=0.45, alpha_coef=0.08, p=0.35),
    A.RandomRain(drop_length=12, drop_width=1, drop_color=(200, 200, 200), blur_value=3, brightness_coefficient=0.8, p=0.35),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.Rotate(limit=12, p=0.5, border_mode=cv2.BORDER_CONSTANT),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def convert_bbox_to_yolo(img_w, img_h, x1, y1, x2, y2):
    """تحويل الإحداثيات إلى نسب بين 0 و 1 لصيغة YOLO"""
    x_center = max(0.0, min(1.0, (((x1 + x2) / 2.0) / img_w)))
    y_center = max(0.0, min(1.0, (((y1 + y2) / 2.0) / img_h)))
    width = max(0.0, min(1.0, ((x2 - x1) / img_w)))
    height = max(0.0, min(1.0, ((y2 - y1) / img_h)))
    return [x_center, y_center, width, height]

def generate_yolo_augmented_dataset(csv_file, base_dir, output_dir, num_augmented=2, test_size=0.2):
    df = pd.read_csv(csv_file)

    # إنشاء المجلدات الخاصة بـ YOLO
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # تقسيم البيانات إلى train و val
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42, stratify=df['ClassId'])
    splits = {'train': train_df, 'val': val_df}

    for split_name, split_df in splits.items():
        print(f"🔄 جاري تحويل وزيادة بيانات مجموعة: {split_name}...")

        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            img_rel_path = row['Path']
            img_full_path = os.path.join(base_dir, img_rel_path)

            if not os.path.exists(img_full_path):
                continue

            img = cv2.imread(img_full_path)
            if img is None:
                continue
            
            img_h, img_w, _ = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # تحويل الإحداثيات للأصلية
            bbox = convert_bbox_to_yolo(img_w, img_h, row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2'])
            class_id = int(row['ClassId'])

            file_base_name = os.path.splitext(os.path.basename(img_rel_path))[0]

            # 1. حفظ الصورة والملف النصي الأصلي بأسلوب YOLO
            orig_img_out = os.path.join(output_dir, 'images', split_name, f"{file_base_name}.jpg")
            orig_lbl_out = os.path.join(output_dir, 'labels', split_name, f"{file_base_name}.txt")
            
            cv2.imwrite(orig_img_out, img)
            with open(orig_lbl_out, 'w') as f:
                f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

            # 2. تطبيق الزيادة فقط على مجموعة الـ train لتجنب التعديل على بيانات الاختيار (val)
            if split_name == 'train':
                for i in range(num_augmented):
                    try:
                        augmented = augment_transform(image=img_rgb, bboxes=[bbox], class_labels=[class_id])
                        aug_img = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                        aug_bboxes = augmented['bboxes']

                        if len(aug_bboxes) > 0:
                            aug_file_name = f"{file_base_name}_aug_{i+1}"
                            aug_img_out = os.path.join(output_dir, 'images', split_name, f"{aug_file_name}.jpg")
                            aug_lbl_out = os.path.join(output_dir, 'labels', split_name, f"{aug_file_name}.txt")

                            cv2.imwrite(aug_img_out, aug_img)
                            b = aug_bboxes[0]
                            with open(aug_lbl_out, 'w') as f:
                                f.write(f"{class_id} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")
                    except Exception:
                        continue

    print("\n✅ تم تجهيز مجموعة بيانات YOLO المعدلة بنجاح في مجلد: data/yolo_dataset")

if __name__ == "__main__":
    CSV_FILE = os.path.join("data", "Train.csv")
    BASE_DIR = "data"
    OUTPUT_DIR = os.path.join("data", "yolo_dataset")

    # num_augmented=2 تعني إضافة نسختين معدلتين لكل صورة تدريب
    generate_yolo_augmented_dataset(CSV_FILE, BASE_DIR, OUTPUT_DIR, num_augmented=2)