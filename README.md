# 🚗 Number Plate Recognition 

YOLOv8 ve Streamlit tabanlı iki aşamalı plaka tanıma uygulaması:
1. **Plaka Tespiti (`plate_detection.pt`):** Araç üzerinden plaka bölgesini tespit edip kırpar.
2. **Karakter Okuma (`plate_reading.pt`):** Kırpılan plakadaki karakterleri tespit edip $x$ eksenine göre soldan sağa dizerek plaka metnini oluşturur.

---

## 🚀 Canlı Demo
🔗 [Streamlit Uygulamasına Git](https://number-plate-recognition-app-6sv7vxpkfkpk6w3pkt692l.streamlit.app/)

---

## 🛠️ Yerel Kurulum

```bash
git clone [https://github.com/Mustafa037/number-plate-recognition-app.git](https://github.com/Mustafa037/number-plate-recognition-app.git)
cd number-plate-recognition-app
pip install -r requirements.txt
streamlit run main.py
