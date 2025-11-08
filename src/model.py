import torch
import torch.nn as nn
import math

# 1. Scaled Dot-Product Attention（无修改）
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

# 2. Multi-Head Attention（无修改，保留之前的多头消融逻辑）
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=128, h=2, ablate_multihead=False):
        super().__init__()
        self.ablate_multihead = ablate_multihead
        self.h = 1 if ablate_multihead else h
        self.d_k = d_model // self.h
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention()

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        Q_proj = self.W_Q(Q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        K_proj = self.W_K(K).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        V_proj = self.W_V(V).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        attn_output, _ = self.attn(Q_proj, K_proj, V_proj, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        output = self.W_O(attn_output)
        return output

# 3. Position-Wise FFN（新增ablate_ffn参数，消融时返回恒等映射）
class PositionWiseFFN(nn.Module):
    def __init__(self, d_model=128, d_ff=512, ablate_ffn=False):
        super().__init__()
        self.ablate_ffn = ablate_ffn  # FFN消融开关
        if not ablate_ffn:
            self.linear1 = nn.Linear(d_model, d_ff)  # 正常FFN：d_model→d_ff
            self.linear2 = nn.Linear(d_ff, d_model)  # 正常FFN：d_ff→d_model
            self.relu = nn.ReLU()
        # 消融时不初始化FFN层，直接返回输入

    def forward(self, x):
        if self.ablate_ffn:
            return x  # 消融FFN：恒等映射（输入=输出）
        else:
            return self.linear2(self.relu(self.linear1(x)))  # 正常FFN前向传播

# 4. 残差连接+LayerNorm（无修改）
class SublayerConnection(nn.Module):
    def __init__(self, d_model=128, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.layer_norm(x + self.dropout(sublayer(x)))

# 5. 位置编码（无修改）
def get_positional_encoding(seq_len, d_model=128):
    pos_encoding = torch.zeros(seq_len, d_model)
    pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pos_encoding[:, 0::2] = torch.sin(pos * div_term)
    pos_encoding[:, 1::2] = torch.cos(pos * div_term)
    return pos_encoding.unsqueeze(0)

# 6. Encoder层（新增ablate_ffn参数，传递给FFN）
class EncoderLayer(nn.Module):
    def __init__(self, d_model=128, h=2, d_ff=512, dropout=0.1, ablate_multihead=False, ablate_ffn=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, ablate_multihead)
        self.ffn = PositionWiseFFN(d_model, d_ff, ablate_ffn)  # 传递FFN消融参数
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x, mask):
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer2(x, self.ffn)
        return x

# 7. Decoder层（新增ablate_ffn参数，传递给FFN）
class DecoderLayer(nn.Module):
    def __init__(self, d_model=128, h=2, d_ff=512, dropout=0.1, ablate_multihead=False, ablate_ffn=False):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, h, ablate_multihead)
        self.enc_dec_attn = MultiHeadAttention(d_model, h, ablate_multihead)
        self.ffn = PositionWiseFFN(d_model, d_ff, ablate_ffn)  # 传递FFN消融参数
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)

    def forward(self, x, enc_output, tgt_mask, src_mask):
        x = self.sublayer1(x, lambda x: self.masked_self_attn(x, x, x, tgt_mask))
        x = self.sublayer2(x, lambda x: self.enc_dec_attn(x, enc_output, enc_output, src_mask))
        x = self.sublayer3(x, self.ffn)
        return x

# 8. 完整Transformer（新增ablate_ffn参数，传递给Encoder/Decoder层）
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, n_enc_layers=3,
                 n_dec_layers=3, h=2, d_ff=512, dropout=0.1, max_seq_len=64,
                 ablate_pe=False, ablate_multihead=False, ablate_ffn=False):
        super().__init__()
        self.ablate_pe = ablate_pe
        self.ablate_multihead = ablate_multihead
        self.ablate_ffn = ablate_ffn  # FFN消融开关
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = get_positional_encoding(max_seq_len, d_model)
        # Encoder层：传递ablate_ffn参数
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, h, d_ff, dropout, self.ablate_multihead, self.ablate_ffn)
            for _ in range(n_enc_layers)
        ])
        # Decoder层：传递ablate_ffn参数
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, h, d_ff, dropout, self.ablate_multihead, self.ablate_ffn)
            for _ in range(n_dec_layers)
        ])
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def forward(self, src, tgt, src_mask, tgt_mask):
        src_seq_len = src.size(1)
        tgt_seq_len = tgt.size(1)
        device = src.device

        src_emb = self.src_emb(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_emb(tgt) * math.sqrt(self.d_model)
        if not self.ablate_pe:
            src_emb += self.pos_encoding[:, :src_seq_len, :].to(device)
            tgt_emb += self.pos_encoding[:, :tgt_seq_len, :].to(device)
        src_emb = self.dropout(src_emb)
        tgt_emb = self.dropout(tgt_emb)

        enc_output = src_emb
        for enc_layer in self.encoder:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_emb
        for dec_layer in self.decoder:
            dec_output = dec_layer(dec_output, enc_output, tgt_mask, src_mask)

        output = self.fc_out(dec_output)
        return output

# 9. 掩码生成工具（无修改）
def create_pad_mask(seq, pad_idx=0):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_future_mask(seq):
    seq_len = seq.size(1)
    future_mask = torch.triu(torch.ones(seq_len, seq_len, device=seq.device), diagonal=1)
    return (future_mask == 0).unsqueeze(1)