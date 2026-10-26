import cv2

kamera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not kamera.isOpened():
    kamera = cv2.VideoCapture(0)

if not kamera.isOpened():
    print("ERROR: Kamera tidak dapat dibuka.")
    exit()

print("Kamera berhasil dibuka! Tekan Q untuk keluar.")

while True:
    berhasil, frame = kamera.read()

    if not berhasil:
        print("ERROR: Gagal membaca frame kamera.")
        break

    cv2.imshow("Tes Webcam - Waste Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

kamera.release()
cv2.destroyAllWindows()
