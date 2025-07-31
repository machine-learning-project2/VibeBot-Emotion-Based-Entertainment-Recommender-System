import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
import pandas as pd

# ✅ MUST be the first Streamlit command
st.set_page_config(page_title="Vibe Bot", page_icon="🎭", layout="centered")

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

model = BiLSTMWithAttention(
    vocab_size=len(vocab),
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=len(label_encoder.classes_)
).to(DEVICE)

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

# ========== LOAD CSV FILES ==========
@st.cache_data
def load_csv_data():
    songs = pd.read_csv("emotion_labeled_tracks.csv")
    movies = pd.read_csv("emotion_based_movies_with_links.csv")
    books = pd.read_csv("emotion_labeled_books.csv")
    return songs, movies, books

songs_df, movies_df, books_df = load_csv_data()

# ========== STREAMLIT UI ==========
st.markdown("""
    <style>
    .title {text-align: center; font-size: 2.5em; color: #4B8BBE; margin-bottom: 20px;}
    .category-title {font-size: 1.3em; font-weight: bold; color: #333; margin-top: 20px;}
    .recommendation-list a {text-decoration: none; color: #1f77b4;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🎭 Vibe Bot</div>", unsafe_allow_html=True)

user_input = st.text_input("💬 Describe how you're feeling:")

if user_input:
    emotion = predict_emotion(user_input).lower()
    st.success(f"🎯 Detected Emotion: **{emotion.capitalize()}**")

    # Filter each CSV by emotion
    song_recs = songs_df[songs_df['emotion'].str.lower() == emotion]
    movie_recs = movies_df[movies_df['emotion'].str.lower() == emotion]
    book_recs = books_df[books_df['emotion'].str.lower() == emotion]

    has_any = False

    if not song_recs.empty:
        has_any = True
        st.markdown("<div class='category-title'>🎵 Top 5 Songs</div>", unsafe_allow_html=True)
        for _, row in song_recs.sample(n=min(5, len(song_recs))).iterrows():
            title = row["track_name"]
            artist = row["artist_name"]
            link = row["track_url"]
            st.markdown(
                f'<div class="recommendation-list">• <a href="{link}" target="_blank">{title}</a> by <i>{artist}</i></div>',
                unsafe_allow_html=True
            )

    if not movie_recs.empty:
        has_any = True
        st.markdown("<div class='category-title'>🎥 Top 5 Movies</div>", unsafe_allow_html=True)
        for _, row in movie_recs.sample(n=min(5, len(movie_recs))).iterrows():
            title = row["title"]
            overview = row["overview"]
            link = row["link"]
            st.markdown(
                f'<div class="recommendation-list">• <a href="{link}" target="_blank">{title}</a><br><small>{overview}</small></div>',
                unsafe_allow_html=True
            )

    if not book_recs.empty:
        has_any = True
        st.markdown("<div class='category-title'>📚 Top 5 Books</div>", unsafe_allow_html=True)
        for _, row in book_recs.sample(n=min(5, len(book_recs))).iterrows():
            title = row["title"]
            author = row["authors"]
            link = row["link"]
            st.markdown(
                f'<div class="recommendation-list">• <a href="{link}" target="_blank">{title}</a> by <i>{author}</i></div>',
                unsafe_allow_html=True
            )

    if not has_any:
        st.warning("😕 Sorry, no recommendations found for this emotion.")
