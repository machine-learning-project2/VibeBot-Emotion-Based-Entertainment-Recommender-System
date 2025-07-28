import streamlit as st
import torch
import torch.nn as nn
import pickle
import re

# ========== CONFIG ==========
MAX_LEN = 100
EMBED_DIM = 128
HIDDEN_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== MODEL ==========
class BiLSTMWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(BiLSTMWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, _ = self.lstm(embedded)
        attn_weights = torch.softmax(self.attention(outputs).squeeze(-1), dim=1)
        context = torch.sum(outputs * attn_weights.unsqueeze(-1), dim=1)
        return self.fc(context)

# ========== LOAD ASSETS ==========
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

model = BiLSTMWithAttention(len(vocab), EMBED_DIM, HIDDEN_DIM, len(label_encoder.classes_)).to(DEVICE)
model.load_state_dict(torch.load("bilstm_model.pt", map_location=DEVICE))
model.eval()

# ========== HELPER FUNCTIONS ==========
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def text_to_sequence(text):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab.get("<unk>", 0)) for token in tokens[:MAX_LEN]]
    padded = ids + [vocab.get("<pad>", 0)] * (MAX_LEN - len(ids))
    return torch.tensor([padded], dtype=torch.long).to(DEVICE)

def predict_emotion(text):
    seq = text_to_sequence(text)
    with torch.no_grad():
        output = model(seq)
        pred = torch.argmax(output, dim=1).item()
    return label_encoder.inverse_transform([pred])[0]

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="Vibe Bot", page_icon="🎭", layout="centered")
st.markdown("""
    <style>
    .title {text-align: center; font-size: 2.5em; color: #4B8BBE; margin-bottom: 20px;}
    .category-title {font-size: 1.3em; font-weight: bold; color: #333; margin-top: 20px;}
    .recommendation-list a {text-decoration: none; color: #1f77b4;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🎭 Vibe Bot</div>", unsafe_allow_html=True)

user_input = st.text_input("💬 Describe how you're feeling:")

# ========== RECOMMENDATIONS ==========
from recommendations import recommendations

if user_input:
    emotion = predict_emotion(user_input).lower()
    st.success(f"🎯 Detected Emotion: **{emotion.capitalize()}**")

    if emotion in recommendations:
        st.markdown(f"<div class='category-title'>🧠 Recommendations for <span style='color:#1f77b4'>{emotion.capitalize()}</span></div>", unsafe_allow_html=True)
        for category, items in recommendations[emotion].items():
            if items:
                st.markdown(f"<div class='category-title'>{'🎵' if category == 'songs' else '📚'} {category.capitalize()}</div>", unsafe_allow_html=True)
                for item in items:
                    st.markdown(f'<div class="recommendation-list">• <a href="{item["link"]}" target="_blank">{item["title"]}</a></div>', unsafe_allow_html=True)
    else:
        st.warning("😕 Sorry, we don't have recommendations for that emotion yet.")
