"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION  
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)  # last dimension of Q

    # Step 1 & 2: scaled scores → (..., seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Step 3: apply mask — True positions become -inf
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    # Step 4: softmax over key dimension
    attn_w = F.softmax(scores, dim=-1)

    # Step 5: weighted sum of values
    output = torch.matmul(attn_w, V)

    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS 
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    # src: [batch, src_len]
    # True where token == pad → will be masked out
    mask = (src == pad_idx)          # [batch, src_len]
    return mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, src_len]


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    batch_size, tgt_len = tgt.shape

    # Padding mask: [batch, 1, 1, tgt_len]
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

    # Causal mask: upper triangle True → future positions masked
    # torch.ones(tgt_len, tgt_len).triu(diagonal=1) gives upper triangle
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1
    ).bool()                              # [tgt_len, tgt_len]
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, tgt_len, tgt_len]

    # Combine: mask out if EITHER is a pad OR is a future position
    tgt_mask = pad_mask | causal_mask    # [batch, 1, tgt_len, tgt_len]
    return tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION 
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head
        

        # 4 linear projections: Q, K, V, and output
        # All are d_model → d_model (split into heads happens via reshape)
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.dropout  = nn.Dropout(p=dropout)
        self.attn_w   = None  # store for visualization (W&B attention maps)
    
    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]

        """
        # Step 1: Linear projections → [batch, seq, d_model]
        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)

        # Step 2: Split into heads → [batch, h, seq, d_k]
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # Step 3: Scaled dot-product attention on each head
        # attn_output: [batch, h, seq_q, d_k]
        # attn_w:      [batch, h, seq_q, seq_k]
        attn_output, attn_w = scaled_dot_product_attention(Q, K, V, mask)

        # Store attention weights for W&B visualization (experiment 2.3)
        self.attn_w = attn_w.detach()

        # Apply dropout to attention weights effect
        # (standard practice, applied on the output)
        attn_output = self.dropout(attn_output)

        # Step 4: Merge heads → [batch, seq_q, d_model]
        attn_output = self._merge_heads(attn_output)

        # Step 5: Final output projection → [batch, seq_q, d_model]
        output = self.W_O(attn_output)

        return output
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split d_model into h heads.
        [batch, seq, d_model] → [batch, h, seq, d_k]
        """
        batch, seq, d_model = x.size()
        # Reshape to [batch, seq, h, d_k] then transpose to [batch, h, seq, d_k]
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge h heads back into d_model.
        [batch, h, seq, d_k] → [batch, seq, d_model]
        """
        batch, h, seq, d_k = x.size()
        # Transpose back to [batch, seq, h, d_k] then reshape to [batch, seq, d_model]
        return x.transpose(1, 2).contiguous().view(batch, seq, self.d_model)


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING  
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build PE matrix [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims → sin
        pe[:, 1::2] = torch.cos(position * div_term)  # odd  dims → cos
        pe = pe.unsqueeze(0)                           # [1, max_len, d_model]

        # Buffer: not a parameter, but moves to GPU with model.to(device)
        # Autograder explicitly checks this!
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]  

        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK 
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO: Task 2.3 — define:
        #   self.linear1 = nn.Linear(d_model, d_ff)
        #   self.linear2 = nn.Linear(d_ff, d_model)
        #   self.dropout = nn.Dropout(p=dropout)

        self.linear1 = nn.Linear(d_model, d_ff)   # W1: expand to d_ff (e.g 512→2048)
        self.linear2 = nn.Linear(d_ff, d_model)   # W2: project back (e.g 2048→512)
        self.dropout = nn.Dropout(p=dropout)       # applied between the two linears

        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO:instantiate:
        # Self-attention + its layer norm
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1      = nn.LayerNorm(d_model)

        # Feed-forward + its layer norm
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm2      = nn.LayerNorm(d_model)

        self.dropout    = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]

        """
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))  # Add & Norm

        # ── Sub-layer 2: Feed-Forward ────────────────────────────────
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))   # Add & Norm

        return x
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER 
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO: instantiate:
        self.self_attn   = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1       = nn.LayerNorm(d_model)

        # Sub-layer 2: Cross-attention (Q=decoder, K=V=encoder memory)
        self.cross_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm2       = nn.LayerNorm(d_model)

        # Sub-layer 3: Feed-forward
        self.ffn         = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm3       = nn.LayerNorm(d_model)

        self.dropout     = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        # ── Sub-layer 1: Masked Self-Attention ──────────────────────
        # tgt_mask prevents attending to future positions
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # ── Sub-layer 2: Cross-Attention ─────────────────────────────
        # Q from decoder (x), K and V from encoder (memory)
        # src_mask prevents attending to encoder padding
        attn_out = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(attn_out))

        # ── Sub-layer 3: Feed-Forward ────────────────────────────────
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(N)]
        )
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

        # Re-initialise each layer independently so weights truly differ
        # deepcopy copies current values — reinit breaks the symmetry
        for L in self.layers:
            L._reset_parameters()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(N)]
        )
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

        for L in self.layers:
            L._reset_parameters()

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════


class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """

    def __init__(
        self,
        src_vocab_size: int   = None,   # if None, built from Multi30k train
        tgt_vocab_size: int   = None,   # if None, built from Multi30k train
        d_model:        int   = 512,
        N:              int   = 6,
        num_heads:      int   = 8,
        d_ff:           int   = 2048,
        dropout:        float = 0.1,
        checkpoint_path: str  = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        # ── Step 1: Load vocab + tokenizer (ALWAYS, per announcement) ─
        # Must happen before building model so we know vocab sizes
        # Also needed for infer() at test time
        self._load_vocab_and_tokenizer()

        # ── Step 2: Resolve vocab sizes ───────────────────────────────
        # If caller passed explicit sizes use them, else use built vocab
        if src_vocab_size is None:
            src_vocab_size = len(self.src_vocab)
        if tgt_vocab_size is None:
            tgt_vocab_size = len(self.tgt_vocab)

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        # ── Step 3: Build model components ───────────────────────────
        # Source and target token embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=1)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=1)

        # Positional encoding (sinusoidal, shared for src and tgt)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # Encoder stack: N identical EncoderLayer modules
        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)

        # Decoder stack: N identical DecoderLayer modules
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(dec_layer, N)

        # Final linear: maps d_model → tgt_vocab_size logits
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        # ── Step 4: Load checkpoint if provided ──────────────────────
        # Per announcement: download + load weights inside __init__
        if checkpoint_path is not None:
            gdown.download(
                id="1ZDZteVNIwFpT5frpvgAWAlKAtcAooPTb",  # ← replace after training
                output=checkpoint_path,
                quiet=False
            )
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            self.load_state_dict(ckpt['model_state_dict'])
            print(f"Loaded checkpoint from {checkpoint_path}")

    # ── Internal helpers ──────────────────────────────────────────────

    def _load_vocab_and_tokenizer(self):
        """
        Builds vocab from Multi30k train split and loads spaCy tokenizer.
        Called inside __init__ so infer() works with no external setup.
        """
        import spacy
        from spacy.cli import download
        from dataset import (Multi30kDataset,
                             SOS_IDX, EOS_IDX, PAD_IDX)

        self.SOS_IDX = SOS_IDX
        self.EOS_IDX = EOS_IDX
        self.PAD_IDX = PAD_IDX

        # German tokenizer for infer() safely
        try:
            self.spacy_de = spacy.load('de_core_news_sm')
        except OSError:
            download('de_core_news_sm')
            self.spacy_de = spacy.load('de_core_news_sm')

        # Build vocab from train split — same as during training
        train_ds = Multi30kDataset('train')
        train_ds.build_vocab()

        self.src_vocab = train_ds.src_vocab   # German vocab
        self.tgt_vocab = train_ds.tgt_vocab   # English vocab



    def _load_checkpoint(self, checkpoint_path: str):
        """
        Downloads weights from Google Drive and loads into this model.
        Per announcement: all weight loading inside __init__.
        """
        GDRIVE_FILE_ID = "1ZDZteVNIwFpT5frpvgAWAlKAtcAooPTb"  # replace after training

        if not os.path.exists(checkpoint_path):
            gdown.download(
                id=GDRIVE_FILE_ID,
                output=checkpoint_path,
                quiet=False
            )

        ckpt = torch.load(checkpoint_path, map_location='cpu')
        self.load_state_dict(ckpt['model_state_dict'])
        print(f"Checkpoint loaded from {checkpoint_path}")

    # ── AUTOGRADER HOOKS — do NOT modify signatures ───────────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
        # Scale embeddings by √d_model (paper §3.4)
        # Prevents positional encoding from dominating embedding signal
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.output_projection(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        """
        Translates a German sentence to English using greedy decoding.
        Full pipeline: tokenize → encode → greedy decode → detokenize.
        Vocab and tokenizer already loaded in __init__ — no setup needed.

        Args:
            src_sentence : Raw German text string.

        Returns:
            Translated English string.
        """
        self.eval()
        device = next(self.parameters()).device

        # Step 1: Tokenize German input using spaCy
        tokens = [tok.text.lower()
                  for tok in self.spacy_de.tokenizer(src_sentence)]

        # Step 2: Convert tokens → indices with SOS/EOS
        src_ids = ([self.SOS_IDX]
                   + self.src_vocab.encode(tokens)
                   + [self.EOS_IDX])
        src     = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)
        src_mask = make_src_mask(src, self.PAD_IDX)

        with torch.no_grad():
            # Step 3: Encode the source sentence
            memory = self.encode(src, src_mask)

            # Step 4: Greedy autoregressive decoding
            # Start with just the SOS token
            ys = torch.tensor([[self.SOS_IDX]],
                               dtype=torch.long).to(device)

            # Generate up to src_len + 50 tokens max
            max_len = src.size(1) + 50
            for _ in range(max_len):
                tgt_mask = make_tgt_mask(ys, self.PAD_IDX)
                logits   = self.decode(memory, src_mask, ys, tgt_mask)

                # Pick highest probability token at the last position only
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ys = torch.cat([ys, next_tok], dim=1)

                # Stop when EOS is generated
                if next_tok.item() == self.EOS_IDX:
                    break

        # Step 5: Convert indices → words, skip special tokens
        output_ids = ys.squeeze(0).tolist()
        words = [
            self.tgt_vocab.itos.get(idx, '<unk>')
            for idx in output_ids
            if idx not in (self.SOS_IDX, self.EOS_IDX, self.PAD_IDX)
        ]

        return ' '.join(words)


    


if __name__ == "__main__":
    Q = torch.randn(2, 4, 8, 64)  # batch=2, heads=4, seq=8, d_k=64
    K = torch.randn(2, 4, 8, 64)
    V = torch.randn(2, 4, 8, 64)
    
    out, attn = scaled_dot_product_attention(Q, K, V)
    print(out.shape)   # should be (2, 4, 8, 64)
    print(attn.shape)  # should be (2, 4, 8, 8)
    print(attn.sum(dim=-1))  # every row should sum to ~1.0
    src = torch.tensor([[2, 45, 67, 3, 1, 1]])   # last 2 are PAD
    tgt = torch.tensor([[2, 45, 67, 3, 1]])       # last 1 is PAD

    src_mask = make_src_mask(src, pad_idx=1)
    tgt_mask = make_tgt_mask(tgt, pad_idx=1)

    print(src_mask.shape)        # [1, 1, 1, 6]
    print(src_mask)              # True at positions 4,5

    print(tgt_mask.shape)        # [1, 1, 5, 5]
    print(tgt_mask[0, 0])        # upper triangle + last col True

    mha = MultiHeadAttention(d_model=512, num_heads=8)
    x   = torch.randn(2, 10, 512)   # batch=2, seq=10, d_model=512

    # Self-attention: Q=K=V=x
    out = mha(x, x, x)
    print(out.shape)        # should be [2, 10, 512]
    print(mha.attn_w.shape) # should be [2, 8, 10, 10]

    # With mask
    src = torch.ones(2, 10).long()
    src[:, 7:] = 1  # last 3 are PAD
    mask = make_src_mask(src)
    out_masked = mha(x, x, x, mask)
    print(out_masked.shape) # should be [2, 10, 512]
    ffn = PositionwiseFeedForward(d_model=512, d_ff=2048, dropout=0.0)
    x   = torch.randn(2, 10, 512)
    out = ffn(x)
    print(out.shape)  # [2, 10, 512]


    enc_layer = EncoderLayer(d_model=512, num_heads=8, d_ff=2048, dropout=0.0)
    x         = torch.randn(2, 10, 512)
    src_mask  = torch.zeros(2, 1, 1, 10).bool()   # no padding
    enc_out   = enc_layer(x, src_mask)
    print(enc_out.shape)   # [2, 10, 512]

    # DecoderLayer test
    dec_layer = DecoderLayer(d_model=512, num_heads=8, d_ff=2048, dropout=0.0)
    tgt       = torch.randn(2, 7, 512)
    tgt_mask  = make_tgt_mask(torch.ones(2, 7).long() * 2)  # no PAD, causal only
    dec_out   = dec_layer(tgt, enc_out, src_mask, tgt_mask)
    print(dec_out.shape)   # [2, 7, 512]

    # Encoder stack test
    enc_layer  = EncoderLayer(d_model=512, num_heads=8, d_ff=2048, dropout=0.0)
    encoder    = Encoder(enc_layer, N=6)
    x          = torch.randn(2, 10, 512)
    src_mask   = torch.zeros(2, 1, 1, 10).bool()
    memory     = encoder(x, src_mask)
    print(memory.shape)    # [2, 10, 512]

    # Decoder stack test
    dec_layer  = DecoderLayer(d_model=512, num_heads=8, d_ff=2048, dropout=0.0)
    decoder    = Decoder(dec_layer, N=6)
    tgt        = torch.randn(2, 7, 512)
    tgt_mask   = make_tgt_mask(torch.ones(2, 7).long() * 2)
    dec_out    = decoder(tgt, memory, src_mask, tgt_mask)
    print(dec_out.shape)   # [2, 7, 512]

    # Verify N independent layers (weights must differ)
    w0 = encoder.layers[0].self_attn.W_Q.weight
    w1 = encoder.layers[1].self_attn.W_Q.weight
    print(torch.equal(w0, w1))  # must be False — independent copies

    # Verify layers are truly independent
    enc_layer  = EncoderLayer(d_model=512, num_heads=8, d_ff=2048, dropout=0.0)
    encoder    = Encoder(enc_layer, N=6)
    w0 = encoder.layers[0].self_attn.W_Q.weight
    w1 = encoder.layers[1].self_attn.W_Q.weight
    print(torch.equal(w0, w1))   # must be False now ✅

    memory   = encoder(torch.randn(2, 10, 512), torch.zeros(2,1,1,10).bool())
    print(memory.shape)          # [2, 10, 512] ✅

    dec_layer  = DecoderLayer(d_model=512, num_heads=8, d_ff=2048, dropout=0.0)
    decoder    = Decoder(dec_layer, N=6)
    tgt_mask   = make_tgt_mask(torch.ones(2, 7).long() * 2)
    dec_out    = decoder(torch.randn(2,7,512), memory,
                         torch.zeros(2,1,1,10).bool(), tgt_mask)
    print(dec_out.shape)         # [2, 7, 512] ✅


    # Quick shape test with explicit vocab sizes (skips data download)
    model = Transformer(
        src_vocab_size=1000,
        tgt_vocab_size=800,
        d_model=128,
        N=2,
        num_heads=4,
        d_ff=256,
        dropout=0.0,
    )
    src    = torch.randint(2, 1000, (2, 10))
    tgt    = torch.randint(2, 800,  (2, 7))
    logits = model(src, tgt, make_src_mask(src), make_tgt_mask(tgt))
    print(logits.shape)   # [2, 7, 800]
    memory = model.encode(src, make_src_mask(src))
    print(memory.shape)   # [2, 10, 128]


    