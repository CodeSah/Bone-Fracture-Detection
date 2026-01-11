from flask import Flask, request, jsonify, render_template, send_from_directory
import pickle
import numpy as np
import joblib
import cv2
from skimage.feature import hog
from werkzeug.utils import secure_filename
import os

app = Flask(__name__, static_folder='static')

model_path = os.path.join(os.path.dirname(__file__), "bone_fracture_model.pkl")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = joblib.load(model_path)

OPTIMAL_THRESHOLD = 0.3725

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_hog(image):
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features.reshape(1, -1)

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = image.reshape(1, -1)
    return image



@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        print("Received file:", file.filename)

        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return jsonify({"error": "Invalid image format"}), 400
        IMG_HEIGHT, IMG_WIDTH = 224, 224
        img=img/255.0
        img_features=extract_hog(img)

        prob_fracture=model.predict_proba(img_features)[0][1]
        prediction = 1 if prob_fracture > OPTIMAL_THRESHOLD else 0
        result = "Fracture Detected: Further examination recommended" if prediction == 1 else "No Fracture Detected :X-ray appears normal"

        print("Prediction result:", result)

        return jsonify({"prediction": result })

    except Exception as e:
        print("Error processing image:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)