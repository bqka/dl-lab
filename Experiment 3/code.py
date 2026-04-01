import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Data -------------------------------------------------------------------

MAX_VOCAB = 20000
MAX_LEN = 256
BATCH_SIZE = 64

def tokenize(text):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)
    return re.findall(r"\b\w+\b", text)

def load_imdb():
    from torchtext.datasets import IMDB
    train_data = list(IMDB(split="train"))
    test_data = list(IMDB(split="test"))
    train_labels = [1 if label == "pos" else 0 for label, _ in train_data]
    train_texts = [text for _, text in train_data]
    test_labels = [1 if label == "pos" else 0 for label, _ in test_data]
    test_texts = [text for _, text in test_data]
    return train_texts, train_labels, test_texts, test_labels

def build_vocab(texts, max_vocab):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    most_common = counter.most_common(max_vocab - 2)
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in most_common:
        word2idx[word] = len(word2idx)
    return word2idx

def encode(text, word2idx, max_len):
    tokens = tokenize(text)
    ids = [word2idx.get(t, 1) for t in tokens[:max_len]]
    ids += [0] * (max_len - len(ids))
    return ids

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(encode(self.texts[idx], self.word2idx, self.max_len), dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# --- Model ------------------------------------------------------------------

class TextCNN1D(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=100, kernel_sizes=(3, 4, 5), num_classes=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)  # (B, embed_dim, seq_len)
        conv_outs = [torch.relu(conv(x)).max(dim=2).values for conv in self.convs]
        x = torch.cat(conv_outs, dim=1)
        x = self.dropout(x)
        return self.fc(x)

# --- Training ---------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total

def main():
    print(f"Device: {DEVICE}")
    print("Loading IMDB dataset...")
    train_texts, train_labels, test_texts, test_labels = load_imdb()

    print("Building vocabulary...")
    word2idx = build_vocab(train_texts, MAX_VOCAB)
    vocab_size = len(word2idx)
    print(f"Vocab size: {vocab_size}")

    train_ds = IMDBDataset(train_texts, train_labels, word2idx, MAX_LEN)
    test_ds = IMDBDataset(test_texts, test_labels, word2idx, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = TextCNN1D(vocab_size).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 10
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        te_loss, te_acc = evaluate(model, test_loader, criterion)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        print(f"Epoch {epoch}/{EPOCHS}  train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  test_loss={te_loss:.4f}  test_acc={te_acc:.4f}")

    # --- Save outputs -------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["test_loss"], label="Test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["test_acc"], label="Test")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
    plt.close()

    # Sample predictions
    model.eval()
    samples = test_texts[:10]
    sample_labels = test_labels[:10]
    predictions = []
    with torch.no_grad():
        for text in samples:
            ids = torch.tensor([encode(text, word2idx, MAX_LEN)], dtype=torch.long).to(DEVICE)
            pred = model(ids).argmax(1).item()
            predictions.append(pred)

    label_map = {0: "negative", 1: "positive"}
    with open(os.path.join(OUTPUT_DIR, "sample_predictions.txt"), "w", encoding="utf-8") as f:
        f.write(f"Final Test Accuracy: {history['test_acc'][-1]:.4f}\n\n")
        for i, (text, true, pred) in enumerate(zip(samples, sample_labels, predictions)):
            f.write(f"--- Sample {i+1} ---\n")
            f.write(f"Text: {text[:200]}...\n")
            f.write(f"True: {label_map[true]}  |  Predicted: {label_map[pred]}\n\n")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "1dcnn_model.pth"))
    print(f"\nOutputs saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
