# 🚦 Farklı Görüntü Koşullarında Trafik İşareti Tanıma Sistemi

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo)](https://docs.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Proje Hakkında

Bu projenin amacı; **otonom araçlar** ve **gelişmiş sürüş destek sistemleri (ADAS)** için, ideal olmayan çevre koşullarında (karanlık, sis, hareket bulanıklığı, sensör gürültüsü vb.) trafik işaretlerini yüksek kararlılıkla tespit edip sınıflandırabilen **derin öğrenme tabanlı** bir sistem geliştirmektir.

### 🎯 Temel Hedefler
- Farklı hava ve ışık koşullarında güvenilir trafik işareti tanıma
- YOLOv8 tabanlı nesne tespiti ve CNN tabanlı sınıflandırma
- Sentetik veri artırma ile model dayanıklılığının artırılması
- Gerçek zamanlı çalışabilecek düşük gecikme süreli bir sistem

## 🗂️ Proje Yapısı

```
📁 Trafik-Isareti-Tanima/
│
├── 📁 data/                  # Veri setleri (GitHub'a yüklenmez)
│   ├── 📁 raw/               # Orijinal GTSRB verileri
│   ├── 📁 processed/         # YOLO formatına dönüştürülmüş veriler
│   └── 📁 augmented/         # Bozulma efektleri uygulanmış veriler
│
├── 📁 notebooks/             # Jupyter/Colab Notebook dosyaları
│   ├── 01_veri_kesfi_ve_artirma.ipynb
│   ├── 02_model_egitimi.ipynb
│   └── 03_test_ve_analiz.ipynb
│
├── 📁 src/                   # Python kaynak kodları
│   ├── data_utils.py         # Görüntü bozma ve veri hazırlama
│   ├── train.py              # Model eğitim scripti
│   └── evaluate.py           # Başarı metrikleri hesaplama
│
├── 📁 models/                # Eğitilmiş model ağırlıkları
├── 📁 docs/                  # Proje raporları ve dokümanlar
│
├── .gitignore
├── requirements.txt
└── README.md
```

## 🛠️ Kullanılan Teknolojiler

| Kategori | Teknoloji | Açıklama |
|----------|-----------|----------|
| **Derin Öğrenme** | PyTorch, Ultralytics YOLOv8 | Model eğitimi ve nesne tespiti |
| **Görüntü İşleme** | OpenCV, Albumentations | Bozulma efektleri ve ön işleme |
| **Veri Analizi** | NumPy, Pandas, Matplotlib | Veri keşfi ve görselleştirme |
| **Metrikler** | Scikit-learn | Performans değerlendirme |
| **Ortam** | Google Colab, VS Code | Geliştirme ve eğitim |

## 🚀 Kurulum

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/KULLANICI_ADINIZ/Trafik-Isareti-Tanima.git
cd Trafik-Isareti-Tanima
```

### 2. Sanal Ortam Oluşturun (Önerilen)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 4. Veri Setini İndirin
```bash
# Kaggle API ile GTSRB veri setini indirin
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign -p data/raw/
```

> **Not:** Kaggle API kullanmak için `~/.kaggle/kaggle.json` dosyanızın yapılandırılmış olması gerekir.

## 📊 Veri Seti

Bu projede **GTSRB (German Traffic Sign Recognition Benchmark)** veri seti kullanılmaktadır:
- **43 farklı** trafik işareti sınıfı
- **50.000+** eğitim görüntüsü
- **12.000+** test görüntüsü

### Sentetik Bozulma Türleri
| Bozulma Türü | Açıklama |
|---------------|----------|
| 🌫️ Sis Efekti | Gaussian bulanıklaştırma + parlaklık artışı |
| 🌑 Karanlık | Parlaklık ve kontrast düşürme |
| 📷 Hareket Bulanıklığı | Yönlü bulanıklaştırma filtresi |
| 📡 Sensör Gürültüsü | Gaussian ve tuz-biber gürültüsü |
| 🌧️ Yağmur Efekti | Çizgisel yağmur simülasyonu |

## 🧪 Kullanım

### Veri Artırma (Augmentation)
```python
from src.data_utils import apply_augmentations

# Tek bir görüntüye bozulma uygula
augmented = apply_augmentations("data/raw/image.png", effects=["fog", "dark", "noise"])
```

### Model Eğitimi
```python
from src.train import train_yolo

# YOLOv8 modelini eğit
train_yolo(data_yaml="data/processed/dataset.yaml", epochs=50, imgsz=640)
```

### Model Değerlendirme
```python
from src.evaluate import evaluate_model

# Test seti üzerinde metrikleri hesapla
results = evaluate_model(model_path="models/yolo_best.pt", test_data="data/processed/test/")
```

## 📈 Başarı Metrikleri

| Metrik | Hedef |
|--------|-------|
| mAP@0.5 | ≥ %90 |
| Precision | ≥ %85 |
| Recall | ≥ %85 |
| FPS (Inference) | ≥ 30 |

## 👥 Ekip ve Görev Dağılımı

| Görev | Dosya | Açıklama |
|-------|-------|----------|
| Veri Keşfi & Artırma | `01_veri_kesfi_ve_artirma.ipynb` | Veri seti analizi ve sentetik bozulmalar |
| Model Eğitimi | `02_model_egitimi.ipynb` | YOLOv8/CNN model eğitimi |
| Test & Analiz | `03_test_ve_analiz.ipynb` | Performans testi ve sonuç analizi |

## 📄 Lisans

Bu proje eğitim amaçlıdır.

---

> **⚡ İlk Adım:** Google Colab'de `01_veri_kesfi_ve_artirma.ipynb` dosyasını açarak GTSRB veri setini indirin ve ilk görüntüleri ekrana bastırın!
