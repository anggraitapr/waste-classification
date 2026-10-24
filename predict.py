"""Prediksi sampah dari gambar atau webcam.

    python predict.py --image contoh.jpg
    python predict.py --webcam          # tekan 'q' keluar, 's' simpan frame

Tampilan hasil: panel di samping kamera berisi nama kategori, warna tong,
dan petunjuk membuang. Jika ada file ikon di folder icons/<nama_kelas>.png
(misal icons/plastik_kaleng.png), ikon itu otomatis ditampilkan di panel.
Kalau belum ada, panel tetap tampil hanya dengan teks + warna tong.
"""
import argparse
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import numpy as np

from features import extract_features

ICON_DIR = Path("icons")
UNSURE_THRESHOLD = 0.45  # di bawah ini dianggap "tidak yakin"
PANEL_W = 340

# Info per kelas. Warna dalam BGR. Sesuaikan key dengan nama folder dataset.
CATEGORY_INFO = {
    "kertas_dus": {
        "title": "KERTAS & DUS", "bin": "Tong BIRU",
        "tip": "Kering dan bersih", "color": (190, 110, 30)},
    "plastik_kaleng": {
        "title": "PLASTIK & KALENG", "bin": "Tong KUNING",
        "tip": "Kosongkan & bilas dulu", "color": (0, 190, 240)},
    "sisa_makanan": {
        "title": "SISA MAKANAN", "bin": "Tong HIJAU",
        "tip": "Tanpa kemasan", "color": (110, 170, 60)},
    "residu": {
        "title": "RESIDU", "bin": "Tong RESIDU",
        "tip": "Sampah non-daur ulang", "color": (120, 120, 120)},
}
UNSURE_INFO = {"title": "TIDAK YAKIN", "bin": "Arahkan ulang objek",
               "tip": "Dekatkan & perjelas objek", "color": (60, 60, 60)}


def get_info(label):
    if label is None:
        return UNSURE_INFO
    return CATEGORY_INFO.get(label, {
        "title": label.upper().replace("_", " "), "bin": "", "tip": "",
        "color": (90, 90, 90)})


def classify(bundle, img):
    feat = extract_features(img).reshape(1, -1)
    proba = bundle["model"].predict_proba(feat)[0]
    idx = int(proba.argmax())
    return bundle["classes"][idx], float(proba[idx])


def load_icon(label, size):
    path = ICON_DIR / f"{label}.png"
    if not path.exists():
        path = ICON_DIR / f"{label}.jpg"
    if not path.exists():
        return None
    icon = cv2.imread(str(path))
    return None if icon is None else cv2.resize(icon, (size, size))


def _center_text(img, text, y, scale, color, thick=2):
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    # kecilkan otomatis kalau terlalu lebar
    while tw > img.shape[1] - 20 and scale > 0.4:
        scale -= 0.05
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = (img.shape[1] - tw) // 2
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def make_panel(label, conf, height, icons):
    """Panel hasil: warna tong + ikon (jika ada) + nama kategori + petunjuk."""
    info = get_info(label)
    panel = np.full((height, PANEL_W, 3), info["color"], dtype=np.uint8)
    white = (255, 255, 255)

    icon_size = 170
    top = 30
    icon = icons.get(label) if label else None
    if icon is not None:
        x = (PANEL_W - icon_size) // 2
        panel[top:top + icon_size, x:x + icon_size] = icon
    else:
        # placeholder lingkaran putih sampai foto kategori tersedia
        cv2.circle(panel, (PANEL_W // 2, top + icon_size // 2), icon_size // 2 - 10, white, 4)
        _center_text(panel[top:top + icon_size], "?" if label is None else label[0].upper(),
                     icon_size // 2 + 25, 2.5, white, 4)

    y = top + icon_size + 50
    _center_text(panel, info["title"], y, 0.95, white)
    _center_text(panel, info["bin"], y + 40, 0.75, white, 2)
    _center_text(panel, info["tip"], y + 75, 0.55, white, 1)
    if label is not None:
        _center_text(panel, f"Keyakinan: {conf * 100:.0f}%", height - 25, 0.6, white, 1)
    return panel


def preload_icons(classes, size=170):
    return {c: load_icon(c, size) for c in classes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model.joblib")
    ap.add_argument("--image")
    ap.add_argument("--webcam", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    icons = preload_icons(bundle["classes"])

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit("Gambar tidak bisa dibaca.")
        label, conf = classify(bundle, img)
        print(f"Prediksi: {label} ({conf * 100:.1f}%)")
        shown = label if conf >= UNSURE_THRESHOLD else None
        h = 480
        view = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
        cv2.imshow("Hasil", np.hstack([view, make_panel(shown, conf, h, icons)]))
        cv2.waitKey(0)

    elif args.webcam:
        cap = cv2.VideoCapture(args.camera)
        history = deque(maxlen=7)  # voting beberapa frame agar tidak berkedip
        n = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            s = min(h, w) // 2
            x1, y1 = w // 2 - s // 2, h // 2 - s // 2
            roi = frame[y1:y1 + s, x1:x1 + s]  # taruh sampah di dalam kotak

            label, conf = classify(bundle, roi)
            history.append(label if conf >= UNSURE_THRESHOLD else None)
            stable = Counter(history).most_common(1)[0][0]

            cv2.rectangle(frame, (x1, y1), (x1 + s, y1 + s), get_info(stable)["color"], 3)
            cv2.putText(frame, "Taruh sampah di dalam kotak", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            panel = make_panel(stable, conf, h, icons)
            cv2.imshow("Klasifikasi Sampah", np.hstack([frame, panel]))

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("s"):
                cv2.imwrite(f"capture_{n}.jpg", np.hstack([frame, panel])); n += 1
        cap.release()
    else:
        ap.error("Pilih --image atau --webcam")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
