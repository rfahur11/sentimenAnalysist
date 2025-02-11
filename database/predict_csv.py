from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import pandas as pd
from io import StringIO
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

app = Flask(__name__)
CORS(app)

# === Definisi Model (sama seperti saat training) ===
class IndoBERT_BiLSTM(nn.Module):
    def __init__(self, bert_model="indobenchmark/indobert-base-p1", lstm_hidden=128, num_classes=3):
        super(IndoBERT_BiLSTM, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        self.bert.requires_grad_(False)
        self.lstm = nn.LSTM(
            input_size=768, 
            hidden_size=lstm_hidden, 
            num_layers=2,
            batch_first=True, 
            bidirectional=True, 
            dropout=0.3
        )
        self.batch_norm = nn.BatchNorm1d(lstm_hidden * 2)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)
        
    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embedding = bert_output.last_hidden_state[:, 0, :]  # [batch_size, 768]
        lstm_out, _ = self.lstm(bert_embedding.unsqueeze(1))  # [batch_size, seq_len=1, hidden_size*2]
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.batch_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output

# === Inisialisasi Model, Tokenizer, Device, dan Label Mapping ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IndoBERT_BiLSTM().to(device)
model.load_state_dict(torch.load("../model/best_model.pth", map_location=device))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
label_mapping = {0: "negatif", 1: "netral", 2: "positif"}

# === Fungsi Prediksi untuk Input Teks ===
def predict(text, model, tokenizer, device, label_mapping):
    model.eval()
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    pred_idx = torch.argmax(outputs, dim=1).item()
    return label_mapping[pred_idx]

# === Endpoint untuk Menerima File CSV dan Mengembalikan File CSV dengan Prediksi ===
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    # Pastikan file dengan key 'file' diupload
    if 'file' not in request.files:
        return jsonify({"error": "Tidak ada file yang diupload."}), 400
    
    file = request.files['file']
    
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": "File CSV tidak valid."}), 400

    # Pastikan kolom 'comment' ada pada CSV
    if "comment" not in df.columns:
        return jsonify({"error": "CSV harus mengandung kolom 'comment'."}), 400

    # Lakukan prediksi untuk setiap baris komentar
    predictions = []
    for comment in df["comment"]:
        pred_label = predict(str(comment), model, tokenizer, device, label_mapping)
        predictions.append(pred_label)
    df["predicted_label"] = predictions

    # Simpan DataFrame ke string CSV
    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    # Kembalikan file CSV sebagai response
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=predictions.csv"}
    )

if __name__ == "__main__":
    app.run(debug=True)
