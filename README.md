# Klasifikasi Sampah Menggunakan OpenCV

## 1. Deskripsi Proyek

Pemilahan sampah yang tidak tepat mengurangi efektivitas daur ulang. Proyek ini adalah prototipe
sistem computer vision yang mengambil gambar sampah dari kamera (webcam) atau file gambar,
menganalisis karakteristik visualnya, lalu mengklasifikasikannya ke kategori yang sesuai dengan
tempat sampah di kampus, sehingga pengguna tahu harus membuangnya ke tong yang mana.

Kategori (kelas):

| Kelas            | Tong                      |
| ---------------- | ------------------------- |
| `kertas_dus`     | Biru (kertas & dus)       |
| `plastik_kaleng` | Kuning (plastik & kaleng) |
| `sisa_makanan`   | Hijau (sisa makanan)      |
| `residu`         | Residu (non-daur ulang)   |

Metode: computer vision klasik, tanpa deep learning.

- Pra-pemrosesan (OpenCV): resize 224x224 dan Gaussian blur.
- Ekstraksi fitur (OpenCV): warna (histogram HSV, statistik Lab), tekstur (LBP, kepadatan tepi
  Canny, variansi Laplacian), bentuk (Hu moments dan kontur terbesar).
- Klasifikasi: SVM kernel RBF (scikit-learn) dengan pencarian parameter GridSearchCV.

## 2. Kebutuhan Perangkat Lunak dan Dependensi

- Python 3.9 sampai 3.12
- Webcam (untuk `collect` dan `predict --webcam`)
- Library (lihat `requirements.txt`): `opencv-python`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`

## 3. Petunjuk Instalasi

Pilih salah satu cara berikut.

**Opsi A: langsung (tanpa virtual environment)**

```
pip install -r requirements.txt
```

**Opsi B: dengan virtual environment (disarankan, agar tidak bentrok dengan proyek lain)**

```
python -m venv venv
venv\Scripts\activate          # Windows (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

Cek instalasi:

```
python -c "import cv2; print(cv2.__version__)"
```

## 4. Cara Menjalankan Aplikasi

Urutan penggunaan:

1. Kumpulkan dataset lewat webcam (opsional jika dataset sudah ada):
   ```
   python main.py collect
   ```
2. Latih model:
   ```
   python main.py train --data dataset --augment
   ```
3. Jalankan prediksi:
   ```
   python main.py predict --webcam
   python main.py predict --image contoh.jpg
   ```
   Script juga bisa dijalankan langsung (`python train.py ...`, `python predict.py ...`).

### Dataset

Susun gambar dengan satu folder per kelas (nama folder harus persis sama dengan nama kelas):

```
dataset/
  kertas_dus/
  plastik_kaleng/
  sisa_makanan/
  residu/
```

Sumber data: (isi: foto sendiri lewat `collect.py` berawalan `cam_`,
dan dataset eksternal beserta tautan dan lisensinya jika ada).

## 5. Cara Memberikan Input

- **Webcam:** `python main.py predict --webcam`. Letakkan sampah di dalam kotak di tengah layar.
  Tombol: `q` keluar, `s` simpan tangkapan layar.
- **File gambar:** `python main.py predict --image path/ke/gambar.jpg`.
- **Pengumpulan data:** `python main.py collect`. Tombol: `1`-`4` pilih kelas, `SPASI` ambil foto,
  `b` mode burst (otomatis tiap sekitar 0,4 detik), `q` keluar.

Kamera lain dapat dipilih dengan `--camera 1`.

## 6. Output yang Diharapkan

- **`train`**: classification report (precision, recall, F1 per kelas) di terminal,
  `confusion_matrix.png`, dan model terlatih `model.joblib`.
- **`predict --webcam`**: video langsung dengan kotak penanda dan panel di samping yang
  menampilkan nama kategori, warna dan nama tong, petunjuk membuang, serta persentase keyakinan.
  Jika keyakinan di bawah 45%, panel menampilkan "TIDAK YAKIN".
- **`predict --image`**: hasil prediksi dicetak di terminal dan ditampilkan dengan panel yang sama.
- **`collect`**: foto tersimpan di `dataset/<kelas>/` dan jumlah per kelas ditampilkan di layar.

## 7. Model dan Konfigurasi Tambahan

- `model.joblib` harus ada di folder yang sama saat menjalankan `predict`. File ini dibuat oleh `train`.
  Gunakan `--model path/model.joblib` untuk lokasi lain.
- (Opsional) Taruh gambar kategori di `icons/<nama_kelas>.png` (misalnya `icons/plastik_kaleng.png`)
  agar ditampilkan di panel hasil. Jika tidak ada, panel memakai placeholder.
- Nama kelas, warna, dan teks petunjuk dapat diubah di `CATEGORY_INFO` pada `predict.py`.
  Daftar kelas untuk pengumpulan data ada di `CLASSES` pada `collect.py`.
- Ambang "tidak yakin" diatur lewat `UNSURE_THRESHOLD` di `predict.py`.

## 8. Keterbatasan yang Diketahui

- Fitur bentuk mengandalkan segmentasi sederhana (Otsu), sehingga paling baik dengan satu objek
  pada latar yang relatif polos. Latar ramai atau banyak objek menurunkan akurasi.
- Fitur dibuat manual (warna, tekstur, bentuk), sehingga kurang tangguh dibanding model deep learning
  untuk objek yang mirip secara visual (misalnya plastik bening dan kaca, atau kertas berwarna
  dan plastik).
- Sensitif terhadap pencahayaan dan kualitas kamera.
- Akurasi bergantung pada jumlah dan variasi dataset. Kondisi saat demo sebaiknya mirip dengan
  kondisi pengambilan data.
- Hanya mengenali 4 kelas yang telah ditentukan. Objek di luar itu tetap dipaksa masuk ke salah
  satu kelas, kecuali keyakinannya rendah.
- Sampah campuran (misalnya botol berisi sisa minuman) belum ditangani.
- Hasil evaluasi: (isi dengan akurasi dan ringkasan hasil dari `train.py` setelah training).

## 9. Struktur Berkas

- `main.py`: titik masuk (`collect`, `train`, `predict`)
- `collect.py`: pengumpul dataset lewat webcam
- `features.py`: ekstraksi fitur OpenCV
- `train.py`: training, evaluasi, penyimpanan model
- `predict.py`: inferensi gambar atau webcam
- `requirements.txt`: dependensi

## 10. Referensi

- OpenCV, https://opencv.org
- scikit-learn, https://scikit-learn.org
- NumPy, joblib, matplotlib
- Canny (1986), Hu (1962), Ojala dkk. (2002, LBP), Cortes & Vapnik (1995, SVM)
- Dataset: (isi sesuai sumber yang dipakai)
- Repository GitHub: (isi jika dipakai)
