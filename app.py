from flask import Flask, render_template, request
import cv2
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt

# Inisialisasi Flask
app = Flask(__name__)

# Path folder uploads
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Pastikan folder uploads ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    file = request.files['image']
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Baca gambar & konversi ke grayscale
    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 3)

    # Simpan hasil citra
    gray_path = os.path.join(app.config['UPLOAD_FOLDER'], 'gray_' + filename)
    median_path = os.path.join(app.config['UPLOAD_FOLDER'], 'median_' + filename)
    cv2.imwrite(gray_path, gray)
    cv2.imwrite(median_path, median)

    # Ekstraksi fitur GLCM
    glcm = graycomatrix(median, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    # Simpan histogram
    plt.figure(figsize=(4, 3))
    plt.hist(median.ravel(), bins=256, range=(0, 256))
    plt.title('Histogram Grayscale')
    plt.xlabel('Intensitas')
    plt.ylabel('Jumlah Piksel')
    hist_path = os.path.join(app.config['UPLOAD_FOLDER'], 'hist_' + filename)
    plt.savefig(hist_path)
    plt.close()

    # Kirim hasil ke halaman result
    result = {
        'contrast': contrast,
        'energy': energy,
        'homogeneity': homogeneity,
        'filename': filename
    }

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
