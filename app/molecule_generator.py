"""
molecule_generator.py
-----------------------
Inference-side wrapper around the trained character-level LSTM
(generator_model.py / train_generator.py). Samples new candidate SMILES
strings one character at a time, temperature-controlled, then hands each
one straight to toxicity_checker.analyze_smiles() so nothing generated is
ever shown to the user without being validated first:

    generate -> validate -> explain -> visualize -> refine

Model files are located relative to this file, matching the project's
documented "models/ must live inside app/" gotcha.
"""

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger

from generator_model import SmilesLSTM

# The generator is expected to produce invalid SMILES a meaningful
# fraction of the time (see README: ~20% invalid, by design, with a
# small model). RDKit logs a parse error for every one of those - that
# is expected noise, not a bug, so it's silenced here rather than
# spamming the terminal/app logs on every generation click.
RDLogger.DisableLog("rdApp.*")

_THIS_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(_THIS_DIR, "models", "generator_lstm.pt")
VOCAB_PATH = os.path.join(_THIS_DIR, "models", "generator_vocab.pt")

MAX_GEN_LEN = 90  # matches MAX_LEN in train_generator.py

_gen_cache = {"model": None, "vocab": None, "load_error": None}


def _load():
    if _gen_cache["model"] is not None or _gen_cache["load_error"] is not None:
        return

    if not (os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH)):
        _gen_cache["load_error"] = (
            "No trained generator found. Run `python train_generator.py` "
            "first (see README Step 8) to generate models/generator_lstm.pt."
        )
        return

    try:
        vocab = torch.load(VOCAB_PATH, weights_only=False)
        model = SmilesLSTM(vocab_size=len(vocab))

        checkpoint = torch.load(MODEL_PATH, map_location="cpu")
        # train_generator.py's final save is a plain state_dict, but a
        # mid-training checkpoint (epoch/model_state/optimizer_state) can
        # also be present if training was interrupted - handle both.
        state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()

        _gen_cache["model"] = model
        _gen_cache["vocab"] = vocab
    except Exception as exc:  # pragma: no cover - defensive, surfaced to the UI
        _gen_cache["load_error"] = f"Could not load the generator model: {exc}"


@dataclass
class GeneratedCandidate:
    smiles: str
    is_novel: bool


def _sample_one(model, vocab, temperature: float, training_smiles_set: set) -> str | None:
    """Samples a single SMILES string, char by char, until <EOS> or MAX_GEN_LEN."""
    device = torch.device("cpu")
    input_id = torch.tensor([[vocab.bos_id]], dtype=torch.long, device=device)
    hidden = None
    generated_ids = []

    with torch.no_grad():
        for _ in range(MAX_GEN_LEN):
            logits, hidden = model(input_id, hidden)
            next_logits = logits[0, -1] / max(temperature, 1e-4)
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()

            if next_id == vocab.eos_id:
                break
            generated_ids.append(next_id)
            input_id = torch.tensor([[next_id]], dtype=torch.long, device=device)

    smiles = vocab.decode(generated_ids)
    return smiles if smiles else None


def generate_candidates(n_candidates: int, temperature: float, max_attempts: int = 60):
    """
    Generates up to n_candidates valid, RDKit-parseable, canonicalized,
    de-duplicated SMILES strings. Invalid attempts are silently discarded
    (this is the honestly-reported ~20% invalid rate from the README) -
    the caller only ever sees molecules that already passed validation.

    Returns a list of GeneratedCandidate. May return fewer than
    n_candidates if max_attempts is exhausted first.
    """
    _load()
    if _gen_cache["load_error"] is not None:
        raise RuntimeError(_gen_cache["load_error"])

    model = _gen_cache["model"]
    vocab = _gen_cache["vocab"]

    results = []
    seen = set()
    attempts = 0

    while len(results) < n_candidates and attempts < max_attempts:
        attempts += 1
        raw = _sample_one(model, vocab, temperature, seen)
        if not raw:
            continue

        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            continue  # invalid - discarded, as documented

        canonical = Chem.MolToSmiles(mol)
        if canonical in seen:
            continue
        seen.add(canonical)
        results.append(GeneratedCandidate(smiles=canonical, is_novel=True))

    return results


def is_generator_available() -> bool:
    _load()
    return _gen_cache["load_error"] is None
