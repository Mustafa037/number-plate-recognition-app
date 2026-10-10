from PIL import Image
import streamlit as st
from ultralytics import YOLO
from helper import detect_plate, read_plate

st.set_page_config(page_title="Plaka Tanıma Sistemi", page_icon="🚗", layout="wide")

# Modelleri cache'e alıyoruz (her sayfa yenilendiğinde diskten tekrar yüklenmez)
@st.cache_resource
def load_models():
    detection_model = YOLO("models/plate_detection.pt")
    reading_model = YOLO("models/plate_reading.pt")
    return detection_model, reading_model

try:
    detector, reader = load_models()
except Exception as e:
    st.error(f"Modeller yüklenirken hata oluştu: {e}")
    st.info("Lütfen 'models/' klasöründe 'plate_detection.pt' ve 'plate_reading.pt' dosyalarının bulunduğundan emin olun.")
    st.stop()

st.title("🚗 Otomatik Araç Plaka Tanıma Sistemi")
st.write("Görsel yükleyin; sistem aracı, plakayı ve plaka üzerindeki karakterleri otomatik analiz etsin.")

file = st.file_uploader("Bir araç fotoğrafı seçin", type=["png", "jpg", "jpeg"])

if file is not None:
    image = Image.open(file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Orijinal Görsel")
        st.image(image, use_container_width=True)

    with st.spinner("Plaka tespit ediliyor ve karakterler çözümleniyor..."):
        annotated_image, crops = detect_plate(image, detector)

    with col2:
        st.subheader("Tespit Sonucu")
        st.image(annotated_image, use_container_width=True)

    st.divider()

    if crops:
        st.subheader("🔍 Okunan Plaka Detayları")
        for idx, crop in enumerate(crops, start=1):
            plate_text, annotated_crop = read_plate(crop, reader)

            crop_col, text_col = st.columns([1, 2])
            with crop_col:
                st.image(annotated_crop, caption=f"Tespit Edilen Bölge #{idx}", use_container_width=True)
            with text_col:
                st.markdown(f"### Plaka #{idx}")
                st.success(f"# **{plate_text}**")
    else:
        st.warning("Görsel üzerinde plaka tespit edilemedi. Farklı bir açı veya daha net bir görsel deneyebilirsiniz.")