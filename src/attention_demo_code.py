


import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # Output projection
        self.W_O = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.size()
        # -> (batch_size, num_heads, seq_len, d_k)
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # x: (batch_size, num_heads, seq_len, d_k)
        batch_size, num_heads, seq_len, d_k = x.size()
        # -> (batch_size, seq_len, d_model)
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, x, mask=None):
        Q = self.split_heads(self.W_Q(x))
        K = self.split_heads(self.W_K(x))
        V = self.split_heads(self.W_V(x))

        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        context = attn @ V

        # Combine heads and project output
        output = self.combine_heads(context)
        return self.W_O(output), attn


batch_size = 2
seq_len = 4
d_model = 8     # embedding dimension
num_heads = 2


x = torch.randn(batch_size, seq_len, d_model)
attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
out, attn_weights = attention(x)

print("Output shape:", out.shape)  # (batch_size, seq_len, d_model)
print("Attention shape:", attn_weights.shape)  # (batch_size, num_heads, seq_len, seq_len)



import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # torch.nn.Embedding() is a simple lookup table that stores embeddings of a fixed dictionary and size.
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x)  # (batch_size, seq_len, d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd

        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.attn_norm = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.ff_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention + residual + norm
        attn_out, _ = self.attn(x, mask)
        x = self.attn_norm(x + self.dropout(attn_out))

        # Feedforward + residual + norm
        ff_out = self.ff(x)
        x = self.ff_norm(x + self.dropout(ff_out))

        return x

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len=512):
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)

        # nn.ModuleList() Arranges/Holds submodules in a list
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        x = self.embed(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

vocab_size = 1000
seq_len = 10
d_model = 32
num_heads = 4
num_layers = 2
batch_size = 2

model = MiniTransformer(vocab_size, d_model, num_heads, num_layers)
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

output = model(input_ids)  # (batch_size, seq_len, d_model)
print("Output shape:", output.shape)




class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x, encoder_output, tgt_mask=None, memory_mask=None):
        # Masked self-attention
        _x, _ = self.self_attn(x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(_x))

        # Encoder-decoder attention
        _x, _ = self.cross_attn(x, encoder_output, encoder_output, mask=memory_mask)
        x = self.norm2(x + self.dropout(_x))

        # Feedforward
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, num_classes):
        super().__init__()
        self.transformer = MiniTransformer(vocab_size, d_model, num_heads, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pool over sequence length
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.transformer(x)  # (batch_size, seq_len, d_model)
        x = x.transpose(1, 2)    # (batch_size, d_model, seq_len)
        x = self.pool(x).squeeze(-1)  # (batch_size, d_model)
        return self.fc(x)        # (batch_size, num_classes)
