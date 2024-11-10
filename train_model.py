# Mengimpor library yang diperlukan
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Menentukan seed untuk memastikan reproducibility
seed = 123

# Menentukan direktori untuk dataset pelatihan dan validasi
train_dir = 'test_set/test_set'
validation_dir = 'training_set/training_set'

# Menggunakan fungsi image_dataset_from_directory untuk membuat dataset pelatihan
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=train_dir, # Menentukan directory yang akan dijalankan
    labels="inferred", # Menentukan label otomatis berdasarkan subfolder
    label_mode="categorical",  # Menggunakan label categorical, untuk mengkategori class
    class_names=None, # Menyertakan semua kelas yang ditemukan di direktori
    color_mode="rgb", # Menggunakan format warna RGB
    batch_size=32, # Ukuran batch untuk setiap iterasi
    image_size=(128, 128), # Ukuran gambar yang akan diproses
    seed=seed # Menetapkan seed untuk memastikan hasil yang konsisten
)

# Menggunakan fungsi image_dataset_from_directory untuk membuat dataset validasi
validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory=validation_dir,
    labels="inferred",  # Menentukan label otomatis berdasarkan subfolder
    label_mode="categorical",  # Menggunakan label kategorikal
    class_names=None,  # Menyertakan semua kelas yang ditemukan di direktori
    color_mode="rgb",  # Menggunakan format warna RGB
    batch_size=32,  # Ukuran batch untuk setiap iterasi
    image_size=(128, 128),  # Ukuran gambar yang akan diproses
    seed=seed  # Menetapkan seed untuk memastikan hasil yang konsisten
)

# Menampilkan ukuran dataset pelatihan dan validasi
train_size = len(train_ds)
validation_size = len(validation_ds)
print("Total ukuran dataset pelatihan : ", train_size)
print("Total ukuran dataset validasi : ", validation_size)

# Menyimpan nama kelas dari dataset pelatihan
class_names = train_ds.class_names
print("Class names:", class_names)

# Membuat dictionary yang mengasosiasikan index dengan nama kelas
class_indices = {i: class_name for i, class_name in enumerate(class_names)}

# Menyimpan dictionary tersebut sebagai file .npy
np.save('model/class.npy', class_indices)

# Fungsi untuk menormalkan data (mengubah nilai piksel ke rentang [0, 1])
def process(image,label):
    image = tf.cast(image/255,tf.float32)
    return image,label

# Menerapkan normalisasi ke dataset pelatihan dan validasi
train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

# Menentukan direktori dataset validasi untuk menghitung jumlah kelas
dataset_path = validation_dir

# Menghitung jumlah subdirektori (kelas) dalam dataset validasi
num_classes = len([name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))])
print("Number of classes:", num_classes)

# Menggunakan MobileNetV2 sebagai base model, dengan bobot yang telah dilatih pada ImageNet
conv_base = MobileNetV2(
    weights='imagenet',  # Menggunakan bobot yang dilatih pada ImageNet
    include_top=False,  # Tidak termasuk layer atas (fully connected layers)
    input_shape=(128, 128, 3),  # Ukuran input gambar
    pooling='avg'  # Menggunakan global average pooling sebagai lapisan akhir
)

# Menonaktifkan pelatihan pada base model MobileNetV2
conv_base.trainable = False

# Membuat model sequential baru
model = Sequential()
model.add(conv_base)  # Menambahkan base model MobileNetV2
model.add(BatchNormalization())  # Menambahkan batch normalization
model.add(Dense(32, activation='relu'))  # Menambahkan lapisan Dense dengan 32 unit
model.add(Dropout(0.35))  # Menambahkan Dropout untuk mencegah overfitting
model.add(BatchNormalization())  # Menambahkan batch normalization
model.add(Dense(64, activation='relu'))  # Menambahkan lapisan Dense dengan 64 unit
model.add(Dropout(0.45))  # Menambahkan Dropout
model.add(BatchNormalization())  # Menambahkan batch normalization
model.add(Dense(128, activation='relu'))  # Menambahkan lapisan Dense dengan 128 unit
model.add(Dropout(0.55))  # Menambahkan Dropout
model.add(Dense(num_classes, activation='sigmoid'))  # Lapisan output dengan jumlah kelas dan fungsi aktivasi sigmoid

# Menyiapkan callback untuk pengurangan laju pembelajaran jika validasi loss tidak membaik
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Mengkompilasi model dengan optimizer Adam dan loss function binary crossentropy
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Melatih model menggunakan data pelatihan dan validasi
history = model.fit(train_ds, epochs=5, validation_data=validation_ds, callbacks=[EarlyStopping(patience=5), reduce_lr]) 

# Menyimpan model terlatih ke dalam file
model.save('model/animal_detector.h5')