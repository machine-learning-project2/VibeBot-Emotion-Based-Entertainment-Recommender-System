import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import random
import nltk
import pickle
from nltk.corpus import wordnet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
from collections import Counter

nltk.download('wordnet')  # Only needed once

# Configuration
MAX_LEN = 100
BATCH_SIZE = 32
EPOCHS = 2
EMBED_DIM = 128
HIDDEN_DIM = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CSV
df = pd.read_csv("cleaned_data.csv")  # Ensure columns: 'cleaned_text' & 'emotion'
df = df.dropna(subset=['cleaned_text', 'emotion'])  # Safety

# Encode labels
le = LabelEncoder()
df['emotion'] = le.fit_transform(df['emotion'])

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['cleaned_text'], df['emotion'], test_size=0.2, stratify=df['emotion'], random_state=42)

# Build vocabulary
vocab_dict = {'<pad>': 0, '<unk>': 1}
index = 2

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

for text in train_texts:
    for token in tokenize(text):
        if token not in vocab_dict:
            vocab_dict[token] = index
            index += 1

# Convert text to padded sequences
def text_to_sequence(text):
    tokens = tokenize(text)
    ids = [vocab_dict.get(token, vocab_dict['<unk>']) for token in tokens[:MAX_LEN]]
    padded = ids + [vocab_dict['<pad>']] * (MAX_LEN - len(ids))
    return padded

X_train = torch.tensor([text_to_sequence(text) for text in train_texts])
X_val = torch.tensor([text_to_sequence(text) for text in val_texts])
y_train = torch.tensor(train_labels.tolist())
y_val = torch.tensor(val_labels.tolist())

# Dataloaders
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

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

model = BiLSTMWithAttention(
    vocab_size=len(vocab_dict),
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=len(le.classes_)
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train(model, loader):
    model.train()
    total_loss = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    print(classification_report(all_labels, all_preds, target_names=le.classes_))

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}")
    loss = train(model, train_loader)
    print(f"Train Loss: {loss:.4f}")
    evaluate(model, val_loader)

# Save model
torch.save(model.state_dict(), "bilstm_model.pt")

# Save vocab and label encoder
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab_dict, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
