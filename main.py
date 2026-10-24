"""Titik masuk (entry point) proyek Klasifikasi Sampah dengan OpenCV.

Pemakaian:
    python main.py collect                            # kumpulkan dataset lewat webcam
    python main.py train [--data dataset] [--augment] # latih model + evaluasi
    python main.py predict --webcam                   # prediksi real-time
    python main.py predict --image contoh.jpg         # prediksi satu gambar

Argumen setelah nama perintah diteruskan apa adanya ke script terkait.
"""
import sys

COMMANDS = {
    "collect": "collect",
    "train": "train",
    "predict": "predict",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        raise SystemExit(0 if len(sys.argv) < 2 else 1)

    cmd = sys.argv[1]
    sys.argv = [f"{cmd}.py"] + sys.argv[2:]  # teruskan argumen ke script tujuan
    module = __import__(COMMANDS[cmd])
    module.main()


if __name__ == "__main__":
    main()
