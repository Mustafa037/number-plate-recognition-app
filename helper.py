import cv2
import numpy as np

def detect_plate(image, model):
    """
    Görsel üzerindeki araç plakalarını tespit eder ve bölgeleri kırpar.
    """
    image_np = np.array(image)
    results = model(image_np, conf=0.4)[0]

    crops = []
    annotated_image = image_np.copy()

    if len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0].item())

            # Görsel sınırları dışına taşmayı engelle
            h, w, _ = image_np.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Plaka alanını kes
            crop = image_np[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)

            # Orijinal görsel üzerine plaka kutusu çiz
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"Plate: {conf:.2f}"
            cv2.putText(
                annotated_image, label, (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

    return annotated_image, crops


def read_plate(crop_image, model):
    """
    Kırpılmış plaka görselindeki karakterleri tespit eder,
    soldan sağa (x koordinatına göre) sıralayıp metin oluşturur.
    """
    crop_np = np.array(crop_image)
    results = model(crop_np, conf=0.3)[0]

    detected_chars = []
    annotated_crop = crop_np.copy()

    if len(results.boxes) > 0:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0].item())
            char_name = model.names[cls_id]

            detected_chars.append({
                "char": char_name,
                "x1": x1
            })

            # Karakter etrafına ince kutu çiz
            cv2.rectangle(annotated_crop, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # Karakterleri soldan sağa x1 koordinatına göre sırala
        detected_chars.sort(key=lambda item: item["x1"])
        plate_text = "".join([item["char"] for item in detected_chars])
    else:
        plate_text = "Okunamadı"

    return plate_text, annotated_crop