import os
import cv2
import pandas as pd
from tqdm import tqdm

def convert_bbox_to_yolo(img_w, img_h, x1, y1, x2, y2):
    x_center = max(0.0, min(1.0, (((x1 + x2) / 2.0) / img_w)))
    y_center = max(0.0, min(1.0, (((y1 + y2) / 2.0) / img_h)))
    width = max(0.0, min(1.0, ((x2 - x1) / img_w)))
    height = max(0.0, min(1.0, ((y2 - y1) / img_h)))
    return [x_center, y_center, width, height]

def process_test_dataset(csv_file, base_dir, output_dir):
    df = pd.read_csv(csv_file)
    
    os.makedirs(os.path.join(output_dir, 'images', 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'test'), exist_ok=True)

    print("🔄 جاري تجهيز مجموعة الاختبار (Test Set) لصيغة YOLO...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_rel_path = row['Path']
        img_full_path = os.path.join(base_dir, img_rel_path)

        if not os.path.exists(img_full_path):
            continue

        img = cv2.imread(img_full_path)
        if img is None:
            continue
        
        img_h, img_w, _ = img.shape
        bbox = convert_bbox_to_yolo(img_w, img_h, row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2'])
        class_id = int(row['ClassId'])

        file_base_name = os.path.splitext(os.path.basename(img_rel_path))[0]

        img_out_path = os.path.join(output_dir, 'images', 'test', f"{file_base_name}.jpg")
        lbl_out_path = os.path.join(output_dir, 'labels', 'test', f"{file_base_name}.txt")

        cv2.imwrite(img_out_path, img)
        with open(lbl_out_path, 'w') as f:
            f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

    print("✅ تم تحويل مجموعة الاختيار (Test) بنجاح دون أي تعديل على الصور!")

if __name__ == "__main__":
    CSV_FILE = os.path.join("data", "Test.csv")
    BASE_DIR = "data"
    OUTPUT_DIR = os.path.join("data", "yolo_dataset")

    process_test_dataset(CSV_FILE, BASE_DIR, OUTPUT_DIR)