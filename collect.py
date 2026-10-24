"""Pengumpul dataset lewat webcam.

    python collect.py
    python collect.py --camera 1 --out dataset

Tombol:
    1-4   pilih kelas (sesuai urutan CLASSES di bawah)
    SPASI ambil 1 foto
    b     nyalakan/matikan mode burst (otomatis ambil foto tiap ~0.4 detik)
    q     keluar

Foto diambil dari kotak tengah (area yang sama dengan predict.py) dan disimpan
ke dataset/<kelas>/. Setelah selesai mengumpulkan, jalankan train.py sendiri.
"""
import argparse
import time
from pathlib import Path

import cv2

CLASSES = ["kertas_dus", "plastik_kaleng", "sisa_makanan", "residu"]
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BURST_INTERVAL = 0.4


def count_images(folder):
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in EXT) if folder.exists() else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    for c in CLASSES:
        (out / c).mkdir(parents=True, exist_ok=True)
    counts = {c: count_images(out / c) for c in CLASSES}

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Kamera tidak bisa dibuka. Coba --camera 1.")

    current = 0
    burst = False
    last_shot = 0.0
    flash_until = 0.0
    session = time.strftime("%Y%m%d_%H%M%S")

    def save(roi):
        cls = CLASSES[current]
        counts[cls] += 1
        name = f"cam_{session}_{counts[cls]:04d}.jpg"
        cv2.imwrite(str(out / cls / name), roi)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        s = min(h, w) // 2
        x1, y1 = w // 2 - s // 2, h // 2 - s // 2
        roi = frame[y1:y1 + s, x1:x1 + s].copy()  # ambil sebelum digambar

        now = time.time()
        if burst and now - last_shot >= BURST_INTERVAL:
            save(roi)
            last_shot = now
            flash_until = now + 0.1

        view = frame.copy()
        color = (0, 0, 255) if now < flash_until else (0, 255, 255)
        cv2.rectangle(view, (x1, y1), (x1 + s, y1 + s), color, 3)

        # Panel info
        cv2.rectangle(view, (0, 0), (w, 34 + 24 * len(CLASSES)), (0, 0, 0), -1)
        mode = "BURST ON" if burst else "manual"
        cv2.putText(view, f"Kelas aktif: {CLASSES[current]}  [{mode}]", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        for i, c in enumerate(CLASSES):
            mark = ">" if i == current else " "
            col = (0, 255, 255) if i == current else (255, 255, 255)
            cv2.putText(view, f"{mark} {i + 1}. {c}: {counts[c]} foto", (10, 50 + 24 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
        cv2.putText(view, "1-4 kelas | SPASI foto | b burst | q keluar", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Kumpulkan Dataset", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord(" "):
            save(roi)
            flash_until = now + 0.15
        elif k == ord("b"):
            burst = not burst
        elif ord("1") <= k < ord("1") + len(CLASSES):
            current = k - ord("1")

    cap.release()
    cv2.destroyAllWindows()
    print("Jumlah foto per kelas:")
    for c in CLASSES:
        print(f"  {c}: {counts[c]}")
    print("Selanjutnya jalankan: python train.py --data dataset --augment")


if __name__ == "__main__":
    main()
