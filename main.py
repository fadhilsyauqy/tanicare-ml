from flask import Flask, request, jsonify
from PIL import Image
import io
from detect import predict_label
from storedata import save_prediction_to_firestore, get_prediction_history_from_firestore

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get('file')
    if file is None or file.filename == "":
        return jsonify({"error": "No file provided"}), 400
    
    try:
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))  # Resize for MobileNetV2
        
        # Melakukan prediksi
        pred_label, pred_accuracy, name_info, symptoms, medicines, application, immediate = predict_label(img)
        
        # Menyimpan hasil ke Firestore
        save_prediction_to_firestore(pred_label, pred_accuracy, name_info, symptoms, medicines)

        return jsonify({
            "label": pred_label,
            "accuracy": f"{pred_accuracy:.2f}%",
            "disease_name": name_info,
            "symptoms": symptoms,
            "application": application,
            "immediate": immediate,
            "medicines": medicines
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/history", methods=["GET"])
def history():
    try:
        # Mengambil riwayat prediksi dari Firestore
        history_data = get_prediction_history_from_firestore()
        
        return jsonify({
            "history": history_data
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
