"""
build_generator_corpus.py
---------------------------
Builds data/generator_training_smiles.txt from data/tox21.csv - the
cleaned, canonicalized, length-filtered SMILES corpus that
train_generator.py expects.

This matches the spec in README.md Step 8 / the project context doc:
"the same Tox21 SMILES, canonicalized and length-filtered to <=90
characters."

Run this once, before train_generator.py, if
data/generator_training_smiles.txt doesn't already exist:

    python build_generator_corpus.py
"""

import pandas as pd
from rdkit import Chem, RDLogger

DATA_PATH = "data/tox21.csv"
OUT_PATH = "data/generator_training_smiles.txt"
MAX_LEN = 90  # matches MAX_LEN in train_generator.py

RDLogger.DisableLog("rdApp.*")  # silence expected parse-failure noise


def main():
    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} rows loaded")

    cleaned = []
    seen = set()
    n_invalid = 0
    n_too_long = 0

    for raw_smiles in df["smiles"]:
        mol = Chem.MolFromSmiles(raw_smiles)
        if mol is None:
            n_invalid += 1
            continue

        canonical = Chem.MolToSmiles(mol)
        if len(canonical) > MAX_LEN:
            n_too_long += 1
            continue

        if canonical in seen:
            continue
        seen.add(canonical)
        cleaned.append(canonical)

    with open(OUT_PATH, "w") as f:
        for s in cleaned:
            f.write(s + "\n")

    print(f"  {n_invalid} invalid, {n_too_long} over {MAX_LEN} chars, discarded")
    print(f"  {len(cleaned)} molecules written to {OUT_PATH}")


if __name__ == "__main__":
    main()
