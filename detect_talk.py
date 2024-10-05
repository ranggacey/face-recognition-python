import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import time

# Fungsi untuk menghitung rasio pergerakan bibir
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[3], mouth[9])  # Jarak vertikal
    B = distance.euclidean(mouth[2], mouth[10])  # Jarak vertikal
    C = distance.euclidean(mouth[0], mouth[6])  # Jarak horizontal
    return (A + B) / (2.0 * C)

# Set ambang batas berbicara
MAR_THRESHOLD = 0.6
MOUTH_AR_CONSEC_FRAMES = 15  # Jika bibir terbuka selama 15 frame, dianggap berbicara

# Inisialisasi detektor wajah dan pelacak landmark wajah
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Inisialisasi capture kamera
cap = cv2.VideoCapture(0)

# Ambil indeks titik bibir dari dlib (dari 68 titik landmark wajah)
(mStart, mEnd) = (49, 68)

frame_count = 0
talking_frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ubah frame ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi wajah
    faces = detector(gray)

    for face in faces:
        # Prediksi landmark wajah
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])

        # Ekstrak koordinat bibir
        mouth = shape[mStart:mEnd]

        # Hitung MAR (mouth aspect ratio)
        mar = mouth_aspect_ratio(mouth)

        # Gambar kontur bibir
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)

        if mar > MAR_THRESHOLD:
            talking_frame_count += 1
            # Jika jumlah frame bicara sudah melewati threshold, ambil foto
            if talking_frame_count >= MOUTH_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Talking", (face.left(), face.top() - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.circle(frame, (face.left() + face.width() // 2, face.top() + face.height() // 2), face.width() // 2, (0, 255, 0), 2)

                # Simpan foto
                filename = f"detected_{time.time()}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Foto disimpan: {filename}")
        else:
            talking_frame_count = 0

    # Tampilkan frame
    cv2.imshow("Video", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepas kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()
