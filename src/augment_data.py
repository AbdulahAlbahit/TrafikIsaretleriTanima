import os
import cv2
import glob
from tqdm import tqdm
import albumentations as A

# 1. تعريف التحويلات لمحاكاة الظروف البيئية الصعبة (ليل، مطر، ضباب، تشويش)
augment_transform = A.Compose([
    # محاكاة تغيير الإضاءة والظلام (الليل والسيارات المتقابلة)
    A.RandomBrightnessContrast(brightness_limit=(-0.35, 0.25), contrast_limit=(-0.3, 0.3), p=0.7),
    
    # محاكاة الضباب في الشارع
    A.RandomFog(fog_coef_lower=0.15, fog_coef_upper=0.45, alpha_coef=0.08, p=0.35),
    
    # محاكاة تساقط المطر
    A.RandomRain(drop_length=12, drop_width=1, drop_color=(200, 200, 200), blur_value=3, brightness_coefficient=0.8, p=0.35),
    
    # محاكاة التشويش والرؤية الخافته/الضبابية (Motion/Gaussian Blur & Noise)
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    
    # دوران وتغيير زاوية الرؤية للإشارة
    A.Rotate(limit=12, p=0.5, border_mode=cv2.BORDER_CONSTANT),
])

def augment_train_dataset(train_dir, num_augmented_per_image=2):
    """
    يمر التابع على كل مجلد فئة (0, 1, 2...) داخل مجلد Train ويولد صوراً إضافية لكل صورة أصلية
    """
    if not os.path.exists(train_dir):
        print(f"❌ المجلد {train_dir} غير موجود!")
        return

    # الحصول على قائمة مجلدات الفئات (0, 1, 2...)
    class_folders = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]
    print(f"📁 تم العثور على {len(class_folders)} فئة داخل مجلد التدريب.")

    total_generated = 0

    # التكرار على كل مجلد فئة مع إظهار شريط التقدم (Progress Bar)
    for class_folder in tqdm(class_folders, desc="معالجة الفئات"):
        class_path = os.path.join(train_dir, class_folder)
        
        # البحث عن كافة الصور داخل الفئة الحالية
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.ppm']:
            image_paths.extend(glob.glob(os.path.join(class_path, ext)))

        for img_path in image_paths:
            # تتخطى الصور التي تم توليدها سابقاً لعدم تكرار العملية
            if "_aug_" in os.path.basename(img_path):
                continue

            image = cv2.imread(img_path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            base_name, ext = os.path.splitext(os.path.basename(img_path))

            # توليد الصور المضافة
            for i in range(num_augmented_per_image):
                augmented = augment_transform(image=image_rgb)
                aug_image = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)

                # تسمية الصورة بأسلوب يمنع الالتباس
                output_filename = f"{base_name}_aug_{i+1}{ext}"
                output_filepath = os.path.join(class_path, output_filename)

                cv2.imwrite(output_filepath, aug_image)
                total_generated += 1

    print(f"\n✅ اكتملت العملية بنجاح! تم إضافة {total_generated} صورة جديدة.")

if __name__ == "__main__":
    # المسار المباشر لمجلد Train بناءً على هيكلية مشروعك
    TRAIN_DIR = os.path.join("data", "Train")
    
    # يمكنك تغيير num_augmented_per_image لجعل كل صورة تعطي (1 أو 2 أو 3 صور جديدة)
    augment_train_dataset(TRAIN_DIR, num_augmented_per_image=2)