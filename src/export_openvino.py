from pathlib import Path
from ultralytics import YOLO

# Dosyanın konumuna göre ana proje dizinini ve models klasörünü bul
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

det_pt_path = MODELS_DIR / "detector_best.pt"
cls_pt_path = MODELS_DIR / "classifier_best.pt"

print(f"Modeller okunuyor: {MODELS_DIR}")

# 1. Dedektör modelini dönüştür (hem 640px video hem 1024px fotoğraf için dynamic=True)
print("Dedektör OpenVINO formatına dönüştürülüyor...")
detector = YOLO(str(det_pt_path))
detector.export(format="openvino", dynamic=True)

# 2. Sınıflandırıcı modelini dönüştür (sabit 96x96 kırpılmış görseller için)
print("Sınıflandırıcı OpenVINO formatına dönüştürülüyor...")
classifier = YOLO(str(cls_pt_path))
classifier.export(format="openvino", imgsz=96)

print(f"Tamamlandı! OpenVINO klasörleri '{MODELS_DIR}' içine oluşturuldu.")