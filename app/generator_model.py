"""
generator_model.py
--------------------
A small character-level LSTM that learns to "write" SMILES strings one
character at a time, the same way a phone keyboard's autocomplete predicts
the next letter.

This is intentionally lightweight (a few hundred thousand parameters) so
it trains on a laptop CPU in a few minutes, in contrast to the heavy 3D
diffusion/GNN generators surveyed in the source literature, which
typically require a GPU and much longer training.

Vocabulary: every unique character that appears in the training SMILES,
plus three special tokens:
  <PAD> - padding, ignored during loss calculation
  <BOS> - "beginning of sequence" - fed in as the very first input
  <EOS> - "end of sequence" - the model's signal that it's done writing
"""

import torch
import torch.nn as nn

PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"


class SmilesVocab:
    """Maps characters <-> integer ids."""

    def __init__(self, smiles_list):
        chars = sorted({c for s in smiles_list for c in s})
        self.itos = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN] + chars
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD_TOKEN]
        self.bos_id = self.stoi[BOS_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]

    def __len__(self):
        return len(self.itos)

    def encode(self, smiles: str):
        return [self.stoi[c] for c in smiles]

    def decode(self, ids):
        chars = []
        for i in ids:
            tok = self.itos[i]
            if tok == EOS_TOKEN:
                break
            if tok in (PAD_TOKEN, BOS_TOKEN):
                continue
            chars.append(tok)
        return "".join(chars)


class SmilesLSTM(nn.Module):
    """A small 2-layer character LSTM. ~500K-1M parameters depending on
    vocab size - trains fine on CPU."""

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.output_layer(out)
        return logits, hidden
