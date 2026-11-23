import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import tempfile

st.set_page_config(page_title="Trafik İşareti Tespit ve Tanıma Sistemi", layout="wide")

st.title("🚦 İki Aşamalı Trafik İşareti Tespit ve Tanıma Sistemi (OpenVINO)")
st.markdown("1. Aşama: **YOLOv8s Dedektör (OpenVINO)** | 2. Aşama: **YOLOv8s-cls Sınıflandırıcı (43 Sınıf)**")

# Proje kök dizini ve models klasörünü otomatik tanımla
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

DET_MODEL_PATH = MODELS_DIR / "detector_best_openvino_model"
CLS_MODEL_PATH = MODELS_DIR / "classifier_best_openvino_model"

@st.cache_resource
def modelleri_yukle():
    detector = YOLO(str(DET_MODEL_PATH), task="detect")
    classifier = YOLO(str(CLS_MODEL_PATH), task="classify")
    return detector, classifier

try:
    detector, classifier = modelleri_yukle()
except Exception as e:
    st.error(
        f"OpenVINO model klasörleri yüklenemedi. Önce 'src/export_openvino.py' dosyasını çalıştırıp "
        f"'{MODELS_DIR}' içinde OpenVINO klasörlerinin oluştuğundan emin olun.\n\nHata: {e}"
    )
    st.stop()

# Yan Menü Ayarları
st.sidebar.header("⚙️ Model Parametreleri")
det_conf = st.sidebar.slider("1. Aşama: Tespit (Dedektör) Eşiği", 0.15, 0.90, 0.35, 0.05)
cls_conf = st.sidebar.slider("2. Aşama: Sınıflandırma Eşiği", 0.20, 0.99, 0.50, 0.05)
padding_orani = st.sidebar.slider("Kırpma Kenar Payı (Padding %)", 0, 25, 10, 5) / 100.0

st.sidebar.header("⚡ Video Performans Ayarları")
kare_atlama = st.sidebar.slider("Kare Atlama (Her X karede bir analiz)", 1, 8, 2, 1)
video_imgsz = st.sidebar.selectbox("Video Analiz Çözünürlüğü", [640, 800, 1024], index=0)

def tespit_yap(bgr_image, analiz_boyutu):
    """Koordinatları ve sınıf etiketlerini OpenVINO modelleriyle hesaplar."""
    h, w, _ = bgr_image.shape
    sonuc_kutulari = []

    det_results = detector.predict(source=bgr_image, imgsz=analiz_boyutu, conf=det_conf, verbose=False)[0]

    for box in det_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        det_score = float(box.conf[0])

        box_w, box_h = x2 - x1, y2 - y1
        pad_x = int(box_w * padding_orani)
        pad_y = int(box_h * padding_orani)

        cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

        crop = bgr_image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        cls_results = classifier.predict(source=crop, imgsz=96, verbose=False)[0]
        top1_idx = int(cls_results.probs.top1)
        top1_score = float(cls_results.probs.top1conf)
        sinif_adi = classifier.names[top1_idx].replace("km-s", "km/s")

        if top1_score >= cls_conf:
            sonuc_kutulari.append({
                "coords": (x1, y1, x2, y2),
                "etiket": f"{sinif_adi} (%{top1_score*100:.1f})",
                "sinif": sinif_adi,
                "cls_guven": f"%{top1_score*100:.1f}",
                "det_guven": f"%{det_score*100:.1f}"
            })

    return sonuc_kutulari

def kutulari_ciz(bgr_image, kutular):
    """Hesaplanan kutuları kare üzerine çizer."""
    cizimli = bgr_image.copy()
    renk = (0, 200, 0)
    for item in kutular:
        x1, y1, x2, y2 = item["coords"]
        etiket = item["etiket"]
        cv2.rectangle(cizimli, (x1, y1), (x2, y2), renk, 2)
        (tw, th), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(cizimli, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, y1), renk, -1)
        cv2.putText(cizimli, etiket, (x1 + 3, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return cizimli

# Dosya Yükleme Alanı
dosya_turu = st.radio("Test Türü Seçin:", ["Fotoğraf Yükle", "Video Yükle"], horizontal=True)

if dosya_turu == "Fotoğraf Yükle":
    yuklenen_foto = st.file_uploader("Bir yol fotoğrafı seçin (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if yuklenen_foto is not None:
        pil_img = Image.open(yuklenen_foto).convert("RGB")
        bgr_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        with st.spinner("OpenVINO ile iki aşamalı analiz yapılıyor..."):
            kutular = tespit_yap(bgr_img, analiz_boyutu=1024)
            sonuc_bgr = kutulari_ciz(bgr_img, kutular)
            sonuc_rgb = cv2.cvtColor(sonuc_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(sonuc_rgb, caption="Tespit ve Sınıflandırma Sonucu", use_container_width=True)
        with col2:
            st.subheader("Tespit Edilen İşaretler")
            if kutular:
                tablo = [{"Trafik İşareti": k["sinif"], "Sınıf Güveni": k["cls_guven"], "Tespit Güveni": k["det_guven"]} for k in kutular]
                st.dataframe(tablo, use_container_width=True)
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

        kare_sayaci = 0
        son_kutular = []

        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break

            if kare_sayaci % kare_atlama == 0:
                son_kutular = tespit_yap(frame, analiz_boyutu=video_imgsz)

            islenmis_kare = kutulari_ciz(frame, son_kutular)

            h, w, _ = islenmis_kare.shape
            if w > 850:
                oran = 850 / w
                islenmis_kare = cv2.resize(islenmis_kare, (850, int(h * oran)))

            st_frame.image(cv2.cvtColor(islenmis_kare, cv2.COLOR_BGR2RGB))
            kare_sayaci += 1

        cap.release()