"""
train_generator.py
--------------------
Trains the character-level LSTM generator on the SMILES corpus prepared
from Tox21 (see README). Run once:

    python train_generator.py

Trains in a few minutes on a laptop CPU (no GPU required) on ~7,700
molecules. Saves the trained weights + vocabulary to models/ so the
Streamlit app can load them.
"""

import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from generator_model import SmilesVocab, SmilesLSTM

DATA_PATH = "data/generator_training_smiles.txt"
MODEL_OUT_PATH = "models/generator_lstm.pt"
VOCAB_OUT_PATH = "models/generator_vocab.pt"

BATCH_SIZE = 64
EPOCHS = 12
LEARNING_RATE = 2e-3
MAX_LEN = 90  # matches the filtering already applied when the corpus was built


class SmilesDataset(Dataset):
    def __init__(self, smiles_list, vocab: SmilesVocab, max_len: int):
        self.vocab = vocab
        self.max_len = max_len
        self.data = []
        for s in smiles_list:
            ids = [vocab.bos_id] + vocab.encode(s) + [vocab.eos_id]
            if len(ids) > max_len + 2:
                continue
            self.data.append(ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_batch(batch, pad_id):
    max_len = max(len(seq) for seq in batch)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return padded


def main():
    print(f"Loading training corpus from {DATA_PATH} ...")
    with open(DATA_PATH) as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    random.seed(42)
    random.shuffle(smiles_list)
    print(f"  {len(smiles_list)} molecules loaded")

    vocab = SmilesVocab(smiles_list)
    print(f"  Vocabulary size: {len(vocab)} characters")

    dataset = SmilesDataset(smiles_list, vocab, MAX_LEN)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_batch(b, vocab.pad_id),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmilesLSTM(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,} (device: {device})")

    # Resume support: if a checkpoint already exists (e.g. a previous run
    # was interrupted, or you're continuing training for more epochs),
    # pick up where it left off instead of starting over.
    start_epoch = 1
    if os.path.exists(MODEL_OUT_PATH):
        checkpoint = torch.load(MODEL_OUT_PATH, map_location=device)
        if isinstance(checkpoint, dict) and "epoch" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"  Resuming from checkpoint at epoch {start_epoch - 1}")

    print("Training ...")
    model.train()
    for epoch in range(start_epoch, EPOCHS + 1):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            logits, _ = model(inputs)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"  Epoch {epoch}/{EPOCHS} - avg loss: {avg_loss:.4f}")

        # Checkpoint after every epoch, so an interrupted run can resume.
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            MODEL_OUT_PATH,
        )

    # Final save: plain state_dict only (smaller file, what ml_predictor-style
    # inference code expects to load for generation).
    torch.save(model.state_dict(), MODEL_OUT_PATH)
    torch.save(vocab, VOCAB_OUT_PATH)
    print(f"Saved final model to {MODEL_OUT_PATH}")
    print(f"Saved vocabulary to {VOCAB_OUT_PATH}")


if __name__ == "__main__":
    main()
