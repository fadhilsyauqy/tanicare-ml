import os
import json
import numpy as np
from tensorflow import keras
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load model
model = keras.models.load_model("./model_v4.h5", compile=False)
labels = [
    "Potato__early_blight", "Potato__healthy", "Potato__late_blight",
    "Rice__brown_spot", "Rice__healthy", "Rice__leaf_blast", "Rice__neck_blast",
    "Soybean__caterpillar", "Soybean__diabrotica_speciosa", "Soybean__healthy",
    "Wheat__brown_rust", "Wheat__healthy", "Wheat__yellow_rust"
] 

# Membaca data JSON
with open("./tani-description.json", "r") as file:
    crop_data = json.load(file)

# Fungsi untuk mendapatkan informasi penyakit berdasarkan label
def get_disease_info(label, crop_data):
    crop_name = label.split("__")[0].lower()  # Mendapatkan nama tanaman dari label
    for disease in crop_data["crops"][crop_name]["diseases"]:
        if disease["disease_id"] == label:
            return disease
    return None

def predict_label(img):
    # Preprocess image
    img_array = np.asarray(img) / 255.0
    img_array = img_array.reshape(1, 224, 224, 3)  # Adjust input for MobileNetV2
    
    # Predict using the model
    predictions = model.predict(img_array)
    pred_label = labels[np.argmax(predictions)]
    pred_accuracy = np.max(predictions) * 100  # Convert to percentage
    
    # Mendapatkan informasi penyakit
    disease_info = get_disease_info(pred_label, crop_data)
    if disease_info:
        name_info = disease_info["name"]
        symptoms = ", ".join(disease_info["detection"]["visual_symptoms"])
        application = disease_info["treatment"]["application_method"]
        immediate = disease_info["treatment"]["immediate_actions"]
        medicines = disease_info["treatment"]["medicines"]
        
        
    else:
        name_info = "Penyakit tidak terdeteksi"
        symptoms = "Tanaman Sehat"
        immediate = "Tidak diperlukan tindakan"
        application = "Tidak diperlukan tindakan"
        medicines = "Tidak diperlukan obat"


    return pred_label, pred_accuracy, name_info, symptoms ,medicines, application, immediate

