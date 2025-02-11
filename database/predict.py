from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

app = Flask(__name__)

# 1. Definisikan arsitektur model seperti saat training.
class IndoBERT_BiLSTM(nn.Module):
    def __init__(self, bert_model="indobenchmark/indobert-base-p1", lstm_hidden=128, num_classes=3):
        super(IndoBERT_BiLSTM, self).__init__()
        
        # Load IndoBERT sebagai feature extractor dan freeze bobotnya.
        self.bert = AutoModel.from_pretrained(bert_model)
        self.bert.requires_grad_(False)
        
        # BiLSTM untuk memproses representasi vektor dari IndoBERT.
        self.lstm = nn.LSTM(
            input_size=768, 
            hidden_size=lstm_hidden, 
            num_layers=2,
            batch_first=True, 
            bidirectional=True, 
            dropout=0.3
        )
        
        # Batch Normalization dan Dropout.
        self.batch_norm = nn.BatchNorm1d(lstm_hidden * 2)
        self.dropout = nn.Dropout(0.3)
        
        # Fully Connected layer untuk menghasilkan prediksi kelas.
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # Ekstraksi fitur dengan IndoBERT.
        with torch.no_grad():
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Ambil representasi token [CLS] sebagai representasi kalimat.
        bert_embedding = bert_output.last_hidden_state[:, 0, :]  # [batch_size, 768]
        
        # Proses dengan BiLSTM (tambahkan dimensi sekuens).
        lstm_out, _ = self.lstm(bert_embedding.unsqueeze(1))  # [batch_size, seq_len=1, hidden_size*2]
        lstm_out = lstm_out[:, -1, :]  # Ambil output terakhir
        
        # Normalisasi dan dropout.
        lstm_out = self.batch_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        # Prediksi akhir.
        output = self.fc(lstm_out)
        return output

# 2. Inisialisasi device, model, tokenizer, dan mapping label.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IndoBERT_BiLSTM().to(device)
model.load_state_dict(torch.load("../model/best_model.pth", map_location=device))
model.eval()  # Set model ke mode evaluasi.

tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
label_mapping = {0: "negatif", 1: "netral", 2: "positif"}

# 3. Definisikan fungsi prediksi.
def predict(text, model, tokenizer, device, label_mapping):
    """
    Melakukan prediksi sentimen pada input teks.
    """
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

# 4. Buat endpoint API untuk prediksi.
@app.route("/predict", methods=["POST"])
def predict_endpoint():
    # Ambil data input berupa JSON.
    data = request.get_json(force=True)
    text = data.get("text")
    
    if not text or not isinstance(text, str):
        return jsonify({"error": "Input 'text' tidak valid."}), 400
    
    # Lakukan prediksi.
    prediction = predict(text, model, tokenizer, device, label_mapping)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
