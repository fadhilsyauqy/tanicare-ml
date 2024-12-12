#Main.py Murni 

```
from flask import Flask, request, jsonify
from PIL import Image
import io
from tanicare_detect import predict_label

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
        
        pred_label, pred_accuracy, name_info, symptoms, medicines = predict_label(img)
        
        return jsonify({
            "label": pred_label,
            "accuracy": f"{pred_accuracy:.2f}%",
            "disease_name": name_info,
            "symptoms": symptoms,
            "medicines": medicines
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
```