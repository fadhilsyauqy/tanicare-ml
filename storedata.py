import firebase_admin
from firebase_admin import firestore

# Inisialisasi aplikasi Firebase Admin menggunakan kredensial default
firebase_admin.initialize_app()

# Mendapatkan klien Firestore
db = firestore.client()

# Fungsi untuk menyimpan hasil prediksi ke Firestore
def save_prediction_to_firestore(pred_label, pred_accuracy, name_info, symptoms, medicines):
    prediction_data = {
        "label": pred_label,
        "accuracy": f"{pred_accuracy:.2f}%",
        "disease_name": name_info,
        "symptoms": symptoms,
        "medicines": medicines,
        "timestamp": firestore.SERVER_TIMESTAMP  # Menambahkan waktu prediksi
    }
    
    # Menyimpan data ke koleksi 'predictions' di Firestore
    db.collection('predictions').add(prediction_data)

# Fungsi untuk mengambil semua riwayat prediksi dari Firestore
def get_prediction_history_from_firestore():
    predictions_ref = db.collection('predictions')
    docs = predictions_ref.stream()
    
    history = []
    for doc in docs:
        history.append(doc.to_dict())
    return history
