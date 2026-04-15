#!/usr/bin/env python
# coding: utf-8

# In[62]:


import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, TensorDataset
from einops import rearrange, einsum


# In[63]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")


# In[64]:


import requests

url = "https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt"
text = requests.get(url).text

# save to file
with open("shakespeare.txt", "w", encoding="utf-8") as f:
    f.write(text)


# In[65]:


text = open("shakespeare.txt", 'r').read()
chars = sorted(list(set(text)))

stoi = {ch : i for i, ch in enumerate(chars)}
itos = {i : ch for i, ch in enumerate(chars)}

data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
data = data[:1000000]
print(data.shape)


# In[66]:


from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - 2 * self.seq_len

    def __getitem__(self, idx):
        src = self.data[idx : idx + self.seq_len]
        tgt = self.data[idx+1 : idx + self.seq_len + 1]

        return src, tgt

seq_len = 50

dataset = TextDataset(data, seq_len)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# In[ ]:


class TokenEmbedding(nn.Module):
  def __init__(self, vocab_size, hidden_dim):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, hidden_dim)

  def forward(self, x):
    return self.embedding(x)


# In[ ]:


class PositionalEncoding(nn.Module):
  def __init__(self, hidden_dim, dropout=0.1, max_len=5000):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, hidden_dim)
    position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0)/hidden_dim))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)

    self.register_buffer('pe', pe)

  def forward(self, x):
    return self.dropout(x + self.pe[:, :x.size(1)]).to(device)


# In[69]:


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4):
      """
      input_dim: Dimensionality of the input.
      num_heads: The number of attention heads to split the input into.
      """
      super(MultiHeadAttention, self).__init__()
      self.hidden_dim = hidden_dim
      self.num_heads = num_heads
      assert hidden_dim % num_heads == 0, "Hidden dim must be divisible by no. of heads"
      self.Wv = nn.Linear(hidden_dim, hidden_dim, bias=False)
      self.Wq = nn.Linear(hidden_dim, hidden_dim, bias=False)
      self.Wk = nn.Linear(hidden_dim, hidden_dim, bias=False)
      self.Wo = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def check_sdpa_inputs(self, x):
      assert x.size(1) == self.num_heads, f"Expected size of x to be ({-1, self.num_heads, -1, self.hidden_dim // self.num_heads}), got {x.size()}"
      assert x.size(3) == self.hidden_dim // self.num_heads

    def scaled_dot_product_attention(self, query, key, value, attention_mask=None, key_padding_mask=None):
      # (batch_size, num_heads, seq_len, head_size)
      self.check_sdpa_inputs(query)
      self.check_sdpa_inputs(key)
      self.check_sdpa_inputs(value)

      d_k = query.size(-1)
      tgt_len, src_len = query.size(-2), key.size(-2)

      # (batch_size, num_heads, query_len, key_len)
      logits = query @ key.transpose(-2, -1) / math.sqrt(d_k)

      if attention_mask is not None:
        if attention_mask.dim() == 2:
          assert attention_mask.size() == (tgt_len, src_len)
          attention_mask = attention_mask.unsqueeze(0)
          logits = logits + attention_mask
        else:
          raise ValueError(f"Attention Mask Size ${attention_mask.size()}")

      if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        logits = logits + key_padding_mask

      attention = torch.softmax(logits, dim=-1)
      output = attention @ value

      return output, attention

    def split_into_heads(self, x, num_heads):
      return rearrange(x, "bs sl (nh hsz) -> bs nh sl hsz", nh=num_heads)

    def combine_heads(self, x):
      return rearrange(x, "bs nh sl hsz -> bs sl (nh hsz)")

    def forward(self, q, k, v, attention_mask=None, key_padding_mask=None):
      """
      q : tensor of shape (batch_size, query_sequence_length, hidden_dim)
      k : tensor of shape (batch_size, key_sequence_length, hidden_dim)
      v : tensor of shape (batch_size, key_sequence_length, hidden_dim)
      attention_mask : tensor of shape (query_sequence_length, key_sequence_length)
      key_padding_mask : tensor of shape (sequence_length, key_sequence_length)
      """
      Q = self.Wq(q)
      K = self.Wk(k)
      V = self.Wv(v)

      q = self.split_into_heads(Q, self.num_heads)
      k = self.split_into_heads(K, self.num_heads)
      v = self.split_into_heads(V, self.num_heads)

      attn_values, attn_weights = self.scaled_dot_product_attention(q, k, v, attention_mask, key_padding_mask)

      grouped = self.combine_heads(attn_values)
      output = self.Wo(grouped)

      self.attention_weights = attn_weights

      return output


# In[70]:


class PositionWiseFeedForward(nn.Module):
  def __init__(self, d_model: int, d_ff: int):
    super().__init__()
    self.fc1 = nn.Linear(d_model, d_ff)
    self.fc2 = nn.Linear(d_ff, d_model)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.fc2(self.relu(self.fc1(x)))


# In[ ]:


class EncoderBlock(nn.Module):
  def __init__(self, n_dim: int, dropout: float, n_heads: int):
    super().__init__()
    self.mha = MultiHeadAttention(n_dim, n_heads)
    self.norm1 = nn.LayerNorm(n_dim)
    self.ff = PositionWiseFeedForward(n_dim, n_dim)
    self.norm2 = nn.LayerNorm(n_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, src_padding_mask=None):
    assert x.ndim==3, "Expected input to be 3-dim, got {}".format(x.ndim)
    att_output = self.mha.forward(x, x, x, key_padding_mask=src_padding_mask)
    x = x + self.dropout(self.norm1(att_output))

    ff_output = self.ff(x)
    output = x + self.dropout(self.norm2(ff_output))

    return output


class Encoder(nn.Module):
  def __init__(self, vocab_size: int, n_dim: int, dropout: float, n_encoder_blocks: int, n_heads: int):
    super().__init__()
    self.n_dim = n_dim
    self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_dim)
    self.positional_encoding = PositionalEncoding(n_dim, dropout)
    self.n_encoder_blocks = nn.ModuleList([
        EncoderBlock(n_dim, dropout, n_heads) for _ in range(n_encoder_blocks)
    ])

  def forward(self, x, src_padding_mask=None):
    x = self.embedding(x) * math.sqrt(self.n_dim)
    x = self.positional_encoding(x)

    for block in self.n_encoder_blocks:
      x = block(x=x, src_padding_mask=src_padding_mask)

    return x


# In[72]:


class DecoderBlock(nn.Module):
    def __init__(self, n_dim: int, dropout: float, n_heads: int):
        super().__init__()
        self.self_attention = MultiHeadAttention(n_dim, n_heads)
        self.norm1 = nn.LayerNorm(n_dim)

        self.cross_attention = MultiHeadAttention(n_dim, n_heads)
        self.norm2 = nn.LayerNorm(n_dim)

        self.ff = PositionWiseFeedForward(n_dim, n_dim)
        self.norm3 = nn.LayerNorm(n_dim)

        # self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None, memory_padding_mask=None):
        masked_att_output = self.self_attention(
            q=tgt, k=tgt, v=tgt, attention_mask=tgt_mask, key_padding_mask=tgt_padding_mask)
        x1 = tgt + self.norm1(masked_att_output)

        cross_att_output = self.cross_attention(
            q=x1, k=memory, v=memory, attention_mask=None, key_padding_mask=memory_padding_mask)
        x2 = x1 + self.norm2(cross_att_output)

        ff_output = self.ff(x2)
        output = x2 + self.norm3(ff_output)

        return output

class Decoder(nn.Module):
    def __init__(self, vocab_size: int, n_dim: int, dropout: int, n_decoder_blocks: int, n_heads: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(n_dim, dropout)
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(n_dim, dropout, n_heads) for _ in range(n_decoder_blocks)]
        )

    def forward(self, tgt, memory, tgt_mask=None, tgt_padding_mask=None, memory_padding_mask=None):
        x = self.embedding(tgt)
        x = self.positional_encoding(x)

        for block in self.decoder_blocks:
            x = block(x, memory, tgt_mask, tgt_padding_mask, memory_padding_mask)

        return x


# In[76]:


n_dim = 128
n_heads = 4
n_encoder_blocks = 2
n_decoder_blocks = 2
dropout = 0.1
vocab_size = len(stoi)

encoder = Encoder(
    vocab_size=vocab_size,
    n_dim=n_dim,
    dropout=dropout,
    n_encoder_blocks=n_encoder_blocks,
    n_heads=n_heads
)

decoder = Decoder(
    vocab_size=vocab_size,
    n_dim=n_dim,
    dropout=dropout,
    n_decoder_blocks=n_decoder_blocks,
    n_heads=n_heads
)

encoder = encoder.to(device)
decoder = decoder.to(device)

fc_out = nn.Linear(n_dim, vocab_size).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    list(encoder.parameters()) +
    list(decoder.parameters()) +
    list(fc_out.parameters()),
    lr=1e-3
)


# In[75]:


src, tgt = next(iter(dataloader))

src = src.to(device)
tgt = tgt.to(device)

memory = encoder(src)
print("encoder output:", memory.shape)

tgt_input = tgt[:, :-1]

out = decoder(tgt_input, memory)
print("decoder output:", out.shape)


# In[78]:


def generate_mask(size):
    return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)

epochs = 5

for epoch in range(epochs):
    total_loss = 0

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        tgt_mask = generate_mask(tgt_input.size(1)).to(device)

        memory = encoder(src)

        output = decoder(tgt_input, memory, tgt_mask=tgt_mask)

        logits = fc_out(output)

        loss = criterion(
            logits.reshape(-1, vocab_size),
            tgt_output.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

