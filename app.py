# Mengimpor modul yang diperlukan
from flask import Flask, request, render_template, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import base64
from io import BytesIO
from PIL import Image

# Membuat instance Flask untuk aplikasi web
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Memuat model terlatih dari file 'animal_detector.h5'
model = load_model('model/animal_detector.h5')

# Memuat informasi label untuk klasifikasi (cats dan dogs)
class_indices = {0: 'cats', 1: 'dogs'}  # Mendefinisikan kelas langsung jika model hanya memiliki 2 kelas

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            filepath = os.path.join('static', file.filename)
            file.save(filepath)
            image = load_img(filepath, target_size=(128, 128))
            image = image.convert('RGB')
            image_path = file.filename

        elif 'imageData' in request.form:
            # Decode the image data from the camera
            image_data = request.form['imageData']
            image_data = base64.b64decode(image_data.split(',')[1])
            image = Image.open(BytesIO(image_data)).resize((128, 128))
            image = image.convert('RGB')
            
            # Save the image from camera to 'static' directory
            image_path = "camera_capture.png"
            image.save(os.path.join('static', image_path))

        else:
            return redirect(request.url)

        image = img_to_array(image)
        image = np.expand_dims(image, axis=0) / 255.0

        preds = model.predict(image)
        pred_class = np.argmax(preds, axis=1)[0]
        pred_label = class_indices[pred_class]

        cat_prob = preds[0][0] * 100
        dog_prob = preds[0][1] * 100

        return render_template(
            'result.html',
            label=pred_label,
            image=image_path,
            cat_prob=cat_prob,
            dog_prob=dog_prob
        )

    return render_template('index.html')


# Route untuk melihat riwayat (history) hasil prediksi
@app.route('/history', methods=['GET'])
def history():
    # Mengambil riwayat dari penyimpanan lokal (ini dapat dimodifikasi untuk menggunakan database)
    history = []  # Gantilah dengan logika untuk mengambil riwayat sebenarnya
    return render_template('history.html', history=history)  # Mengirim riwayat ke halaman history.html

# Menjalankan aplikasi Flask jika file ini dijalankan langsung
if __name__ == '__main__':
    app.run(debug=True)  # Menjalankan aplikasi dalam mode debug