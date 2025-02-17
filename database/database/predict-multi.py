from flask import Flask, request, jsonify, render_template
import torch
import pandas as pd
from transformers import BertTokenizer
import os
import torch.nn as nn
from transformers import BertModel

app = Flask(__name__)

# ========================
# 🔹 LOAD TOKENIZER & MODEL
# ========================
MODEL_PATH = "../model/sentiment_model.pth"
tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Definisikan Model
class BertLSTMSentiment(nn.Module):
    def __init__(self, bert_model_name="indobenchmark/indobert-base-p1", hidden_dim=256, num_labels=4):
        super(BertLSTMSentiment, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(input_size=self.bert.config.hidden_size, hidden_size=hidden_dim,
                            num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_labels * 3)  # 3 kelas per label
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(bert_output.last_hidden_state)
        lstm_out = lstm_out[:, 0, :]
        output = self.fc(self.dropout(lstm_out))
        return output.view(-1, 4, 3)  # (batch_size, 4 aspek, 3 kelas)

# 🔹 Load Model dengan Error Handling
model = BertLSTMSentiment().to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
else:
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# ========================
# 🔹 FUNGSI PREDIKSI SENTIMEN
# ========================
def predict_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return {"error": "Input harus berupa teks (string) dan tidak boleh kosong"}

    inputs = tokenizer([text], padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        predictions = torch.argmax(outputs, dim=-1).cpu().numpy()

    labels = ["Negatif", "Netral", "Positif"]

    return {
        "price": labels[predictions[0, 0]],
        "packaging": labels[predictions[0, 1]],
        "product": labels[predictions[0, 2]],
        "aroma": labels[predictions[0, 3]],
    }

# ========================
# 🔹 ROUTE HALAMAN UTAMA
# ========================
@app.route("/")
def home():
    return render_template("index.html")

# ========================
# 🔹 ROUTE PREDIKSI TEKS
# ========================
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    prediction = predict_sentiment(text)

    return jsonify(prediction)

# ========================
# 🔹 ROUTE PREDIKSI CSV
# ========================
@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    try:
        df = pd.read_csv(file)

        # 🔹 Validasi apakah ada kolom 'comment'
        if "comment" not in df.columns:
            return jsonify({"error": "CSV file must contain 'comment' column"}), 400

        # 🔹 Validasi apakah ada nilai kosong di kolom "comment"
        df = df.dropna(subset=["comment"])

        predictions = []
        for text in df["comment"]:
            pred = predict_sentiment(text)
            pred["comment"] = text  # Tambahkan teks asli ke hasil prediksi
            predictions.append(pred)

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========================
# 🔹 JALANKAN API
# ========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
