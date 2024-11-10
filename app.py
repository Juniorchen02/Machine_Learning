# Mengimpor modul yang diperlukan
from flask import Flask, request, render_template, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

# Membuat instance Flask untuk aplikasi web
app = Flask(__name__)

# Memuat model terlatih dari file 'animal_detector.h5'
model = load_model('model/animal_detector.h5')

# Memuat informasi label untuk klasifikasi (cats dan dogs)
class_indices = {0: 'cats', 1: 'dogs'}  # Mendefinisikan kelas langsung jika model hanya memiliki 2 kelas

# Route utama yang menangani permintaan GET dan POST
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST': # Jika metode request adalah POST (form dikirimkan)
        # Mengecek apakah file gambar ada dalam form
        if 'file' not in request.files:
            return redirect(request.url) # Jika tidak ada file, kembali ke halaman utama
        file = request.files['file']
        if file.filename == '': # Jika tidak ada file yang dipilih
            return redirect(request.url)
        if file: # Jika file ada
            filepath = os.path.join('static', file.filename) # Menentukan path penyimpanan file di folder 'static'
            file.save(filepath) # Menyimpan file gambar ke dalam folder 'static'

            # Proses preprocessing gambar agar siap untuk prediksi
            image = load_img(filepath, target_size=(128, 128))   # Mengubah ukuran gambar menjadi (128, 128)
            image = img_to_array(image) # Mengubah gambar menjadi array NumPy
            image = np.expand_dims(image, axis=0) / 255.0 # Menambahkan dimensi baru dan menormalisasi nilai gambar

            # Melakukan prediksi menggunakan model yang telah diload
            preds = model.predict(image)  # Menggunakan model untuk memprediksi gambar
            pred_class = np.argmax(preds, axis=1)[0]  # Mendapatkan kelas dengan probabilitas tertinggi
            pred_label = class_indices[pred_class]  # Menentukan label kelas berdasarkan index yang diprediksi

            # Menghitung probabilitas untuk kedua kelas (kucing dan anjing)
            cat_prob = preds[0][0] * 100  # Probabilitas untuk kelas kucing
            dog_prob = preds[0][1] * 100  # Probabilitas untuk kelas anjing

            # Menyimpan hasil deteksi dan akurasi prediksi
            result = pred_label  # Menyimpan hasil prediksi ('cats' atau 'dogs')
            accuracy = cat_prob if result == 'cats' else dog_prob  # Akurasi berdasarkan kelas yang terdeteksi

            # Mengirimkan hasil prediksi ke halaman hasil
            return render_template('result.html', label=pred_label, 
                                   image=file.filename, cat_prob=cat_prob, dog_prob=dog_prob,
                                   pred_label=pred_label, accuracy=accuracy)  # Mengirimkan hasil ke halaman result.html

    return render_template('index.html') # Mengembalikan halaman utama untuk GET request

# Route untuk melihat riwayat (history) hasil prediksi
@app.route('/history', methods=['GET'])
def history():
    # Mengambil riwayat dari penyimpanan lokal (ini dapat dimodifikasi untuk menggunakan database)
    history = []  # Gantilah dengan logika untuk mengambil riwayat sebenarnya
    return render_template('history.html', history=history)  # Mengirim riwayat ke halaman history.html

# Menjalankan aplikasi Flask jika file ini dijalankan langsung
if __name__ == '__main__':
    app.run(debug=True)  # Menjalankan aplikasi dalam mode debug