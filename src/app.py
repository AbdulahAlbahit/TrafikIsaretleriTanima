import tempfile
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"

st.set_page_config(
    page_title="Trafik İşareti Tanıma Sistemi",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Farklı Görüntü Koşullarında Trafik İşareti Tanıma Sistemi")
st.write(
    "Eğitilmiş YOLOv8 modelini kullanarak resim ve video üzerinden "
    "gerçek zamanlı nesne tespiti yapabilirsiniz."
)


@st.cache_resource
def load_yolo_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: {MODEL_PATH}. "
            "Eğitilmiş ağırlıkları models/best.pt konumuna ekleyin."
        )
    return YOLO(str(MODEL_PATH))


try:
    model = load_yolo_model()
    st.sidebar.success("Model (best.pt) başarıyla yüklendi!")
except Exception as error:
    st.sidebar.error(f"Model yüklenirken hata oluştu: {error}")
    st.stop()

st.sidebar.header("Model Ayarları")
conf_threshold = st.sidebar.slider(
    "Güven Eşiği (Confidence Threshold)", 0.1, 1.0, 0.25, 0.05
)
test_type = st.radio("Test Yöntemini Seçin:", ["Resim Analizi", "Video Analizi"])

if test_type == "Resim Analizi":
    uploaded_image = st.file_uploader(
        "Bir resim yükleyin...", type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Orijinal Resim")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Model Tespiti")
            with st.spinner("Model tahmin yapıyor..."):
                results = model.predict(image, conf=conf_threshold, imgsz=640)
                res_plotted = results[0].plot()[:, :, ::-1]
                st.image(res_plotted, use_container_width=True)

elif test_type == "Video Analizi":
    uploaded_video = st.file_uploader(
        "Bir video yükleyin (MP4)...", type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(uploaded_video.name).suffix
        ) as video_file:
            video_file.write(uploaded_video.read())

        st.subheader("Canlı Video Tespiti")
        cap = cv2.VideoCapture(video_file.name)
        st_frame = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, conf=conf_threshold)
            res_plotted = results[0].plot()
            st_frame.image(res_plotted, channels="BGR", use_container_width=True)

        cap.release()
        st.success("Video işleme tamamlandı!")
