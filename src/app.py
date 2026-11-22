import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile

st.set_page_config(page_title="Trafik İşareti Tespit ve Tanıma Sistemi", layout="wide")

st.title("🚦 İki Aşamalı Trafik İşareti Tespit ve Tanıma Sistemi")
st.markdown("1. Aşama: **YOLOv8s Dedektör (1024px)** | 2. Aşama: **YOLOv8s-cls Sınıflandırıcı (43 Sınıf)**")

# Modelleri önbelleğe alarak yükle (her işlemde tekrar yüklenmesini önler)
@st.cache_resource
def modelleri_yukle():
    detector = YOLO("../models/detector_best.pt")
    classifier = YOLO("../models/classifier_best.pt")
    return detector, classifier

try:
    detector, classifier = modelleri_yukle()
except Exception as e:
    st.error(f"Model dosyaları yüklenemedi. 'detector_best.pt' ve 'classifier_best.pt' dosyalarının app.py ile aynı klasörde olduğundan emin olun. Hata: {e}")
    st.stop()

# Yan Menü Ayarları
st.sidebar.header("⚙️ Model Parametreleri")
det_conf = st.sidebar.slider("1. Aşama: Tespit (Dedektör) Eşiği", 0.15, 0.90, 0.35, 0.05)
cls_conf = st.sidebar.slider("2. Aşama: Sınıflandırma Eşiği", 0.20, 0.99, 0.50, 0.05)
padding_orani = st.sidebar.slider("Kırpma Kenar Payı (Padding %)", 0, 25, 10, 5) / 100.0

def kare_isle(bgr_image):
    h, w, _ = bgr_image.shape
    cizimli_resim = bgr_image.copy()
    tespit_listesi = []

    # 1. AŞAMA: Geniş açıdan tabela yerlerini bul
    det_results = detector.predict(source=bgr_image, imgsz=1024, conf=det_conf, verbose=False)[0]

    for box in det_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        det_score = float(box.conf[0])

        # GTSRB formatına uyum için kutuya %10 kenar payı (padding) ekle
        box_w = x2 - x1
        box_h = y2 - y1
        pad_x = int(box_w * padding_orani)
        pad_y = int(box_h * padding_orani)

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        crop = bgr_image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        # 2. AŞAMA: Kırpılan tabelayı 43 sınıftan birine ata
        cls_results = classifier.predict(source=crop, imgsz=96, verbose=False)[0]
        top1_idx = int(cls_results.probs.top1)
        top1_score = float(cls_results.probs.top1conf)

        # Klasör isimlerindeki 'km-s' ifadesini ekranda 'km/s' olarak düzelt
        sinif_adi = classifier.names[top1_idx].replace("km-s", "km/s")

        # Sınıflandırma güven eşiğini geçiyorsa çiz ve kaydet
        if top1_score >= cls_conf:
            etiket = f"{sinif_adi} (%{top1_score*100:.1f})"
            renk = (0, 200, 0)  # Yeşil kutu

            cv2.rectangle(cizimli_resim, (x1, y1), (x2, y2), renk, 2)
            (tw, th), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(cizimli_resim, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), renk, -1)
            cv2.putText(cizimli_resim, etiket, (x1 + 3, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            tespit_listesi.append({
                "Trafik İşareti": sinif_adi,
                "Sınıflandırma Güveni": f"%{top1_score*100:.1f}",
                "Dedektör Güveni": f"%{det_score*100:.1f}"
            })

    return cizimli_resim, tespit_listesi

# Dosya Yükleme Alanı
dosya_turu = st.radio("Test Türü Seçin:", ["Fotoğraf Yükle", "Video Yükle"], horizontal=True)

if dosya_turu == "Fotoğraf Yükle":
    yuklenen_foto = st.file_uploader("Bir yol fotoğrafı seçin (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if yuklenen_foto is not None:
        pil_img = Image.open(yuklenen_foto).convert("RGB")
        bgr_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        with st.spinner("İki aşamalı analiz yapılıyor..."):
            sonuc_bgr, tespitler = kare_isle(bgr_img)
            sonuc_rgb = cv2.cvtColor(sonuc_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(sonuc_rgb, caption="Tespit ve Sınıflandırma Sonucu", use_container_width=True)
        with col2:
            st.subheader("Tespit Edilen İşaretler")
            if tespitler:
                st.dataframe(tespitler, use_container_width=True)
            else:
                st.info("Belirlenen eşik değerlerinde trafik işareti bulunamadı.")

else:
    yuklenen_video = st.file_uploader("Bir sürüş videosu seçin (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
    if yuklenen_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(yuklenen_video.read())
        cap = cv2.VideoCapture(tfile.name)

        st_frame = st.empty()
        stop_btn = st.button("Videoyu Durdur")

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break
            islenmis_kare, _ = kare_isle(frame)
            st_frame.image(cv2.cvtColor(islenmis_kare, cv2.COLOR_BGR2RGB), use_container_width=True)

        cap.release()