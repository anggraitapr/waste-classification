"""Training klasifikasi sampah.

Struktur dataset (satu folder per kelas):
    dataset/
        kertas_dus/   *.jpg
        plastik_kaleng/
        sisa_makanan/
        residu/

Jalankan:
    python train.py --data dataset --augment
"""
import argparse
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import joblib
import matplotlib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import extract_features

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def augment(img):
    """Variasi sederhana: flip horizontal dan perubahan kecerahan."""
    yield cv2.flip(img, 1)
    yield cv2.convertScaleAbs(img, alpha=1.0, beta=30)
    yield cv2.convertScaleAbs(img, alpha=1.0, beta=-30)


def load_paths(data_dir):
    classes = sorted(p.name for p in Path(data_dir).iterdir() if p.is_dir())
    items = []
    for ci, name in enumerate(classes):
        for f in sorted((Path(data_dir) / name).rglob("*")):
            if f.suffix.lower() in EXT:
                items.append((f, ci))
    return classes, items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--out", default="model.joblib")
    ap.add_argument("--augment", action="store_true", help="augmentasi hanya pada data latih")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    classes, items = load_paths(args.data)
    print(f"Kelas: {classes} | total gambar: {len(items)}")
    if len(items) < 10:
        raise SystemExit("Dataset terlalu sedikit / struktur folder salah.")

    # Split di level file dulu agar augmentasi tidak bocor ke data uji
    paths = [p for p, _ in items]
    labels = [y for _, y in items]
    p_tr, p_te, y_tr, y_te = train_test_split(
        paths, labels, test_size=args.test_size, stratify=labels, random_state=42)

    def build(plist, ylist, aug):
        X, Y = [], []
        for p, y in zip(plist, ylist):
            img = cv2.imread(str(p))
            if img is None:
                print("Gagal baca:", p)
                continue
            X.append(extract_features(img)); Y.append(y)
            if aug:
                for a in augment(img):
                    X.append(extract_features(a)); Y.append(y)
        return np.array(X), np.array(Y)

    X_tr, Y_tr = build(p_tr, y_tr, args.augment)
    X_te, Y_te = build(p_te, y_te, False)
    print("Fitur latih:", X_tr.shape, "| uji:", X_te.shape)

    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True,
                                               class_weight="balanced"))
    grid = GridSearchCV(pipe, {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.01, 0.001]},
                        cv=3, n_jobs=-1)
    grid.fit(X_tr, Y_tr)
    print("Parameter terbaik:", grid.best_params_)

    pred = grid.predict(X_te)
    report = classification_report(Y_te, pred, target_names=classes, zero_division=0)
    print(report)
    cm = confusion_matrix(Y_te, pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual"); ax.set_title("Confusion Matrix")
    fig.tight_layout()

    # Semua hasil satu kali training disimpan di runs/<tanggal_jam>/
    run_dir = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(run_dir / "confusion_matrix.png", dpi=150)
    joblib.dump({"model": grid.best_estimator_, "classes": classes}, run_dir / "model.joblib")

    per_class = {c: sum(1 for _, y in items if y == i) for i, c in enumerate(classes)}
    summary = (
        f"Waktu training : {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"Dataset        : {args.data}\n"
        f"Gambar per kelas: {per_class}\n"
        f"Augmentasi     : {args.augment}\n"
        f"Data latih/uji : {len(p_tr)} / {len(p_te)} gambar asli\n"
        f"Parameter terbaik: {grid.best_params_}\n\n"
        f"{report}\nConfusion matrix:\n{cm}\n"
    )
    (run_dir / "report.txt").write_text(summary, encoding="utf-8")

    # Salinan model terbaru di lokasi tetap, supaya predict.py tidak perlu diubah
    shutil.copy(run_dir / "model.joblib", args.out)
    print(f"Hasil training disimpan di {run_dir}/ (model, confusion_matrix.png, report.txt)")
    print(f"Model terbaru juga disalin ke {args.out}")


if __name__ == "__main__":
    main()
