"""
train_model.py
----------------
Trains an explainable toxicity model on the real Tox21 dataset (8,014
molecules tested across 12 real toxicology assays - nuclear receptor and
stress-response pathways - from the NIH/EPA Tox21 program).

This is a SECOND, independent signal alongside the rule-based structural
alerts already in toxicity_rules.py. It does not replace them.

v2 upgrade (see README "Going further with more data"): the original
model only saw 10 summary descriptors (molecular weight, LogP, ring
counts, ...) and never looked at the molecule's actual structure. Two
structurally very different molecules could get near-identical scores if
their summary stats matched. This version adds a Morgan/ECFP4 fingerprint
(FP_RADIUS=2, FP_NBITS bits) - a bit-vector where each set bit means "a
specific small substructure is present in this molecule." That lets the
model see real chemical substructures, not just aggregate stats, which is
the single biggest lever for prediction accuracy here.

Model: HistGradientBoostingClassifier (scikit-learn's built-in gradient
boosting - no new dependency). It generally out-performs a plain Random
Forest on this kind of mixed dense/binary tabular feature set, and is
still a tree ensemble, so SHAP's TreeExplainer works on it exactly like
it did on the Random Forest (see ml_predictor.py).

Explainability is kept, just shifted: the 10 named descriptors are still
there and still show up as named SHAP features. For the new fingerprint
bits, ml_predictor.py resolves the top contributing bits back to the
actual substructure they represent (via RDKit's bitInfo) and shows that
substructure's SMILES, so predictions stay auditable rather than turning
into "bit #137 mattered."

Run this once to produce models/tox21_rf_model.joblib:

    python train_model.py

Requires: data/tox21.csv (downloaded from the public DeepChem/MoleculeNet
mirror - see README for the download command).
"""

import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from sklearn.ensemble import HistGradientBoostingClassifier

RDLogger.DisableLog("rdApp.*")  # silence RDKit's per-molecule parse warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
import os

DATA_PATH = "data/tox21.csv"
MODEL_OUT_PATH = "models/tox21_rf_model.joblib"
FEATURE_NAMES_OUT_PATH = "models/tox21_feature_names.joblib"

# Fingerprint settings - MUST match ml_predictor.py exactly, or predictions
# will be silently wrong (the app would be feeding the model a feature
# vector built a different way than the one it was trained on).
FP_RADIUS = 2      # ECFP4-equivalent (radius 2 = diameter 4)
FP_NBITS = 512

# The 12 real assay columns in the Tox21 dataset. Each is 0 (no toxic
# response), 1 (toxic response detected), or missing (not tested for that
# molecule).
ASSAY_COLUMNS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

DESCRIPTOR_NAMES = [
    "molecular_weight",
    "logP",
    "tpsa",
    "num_rings",
    "num_aromatic_rings",
    "num_rotatable_bonds",
    "num_h_donors",
    "num_h_acceptors",
    "num_heteroatoms",
    "fraction_csp3",
]

# Fingerprint feature names carry their bit index so ml_predictor.py can
# map a SHAP-important bit back to the substructure it represents.
FINGERPRINT_NAMES = [f"fp_bit_{i}" for i in range(FP_NBITS)]

FEATURE_NAMES = DESCRIPTOR_NAMES + FINGERPRINT_NAMES


def compute_descriptors(mol):
    """The original 10 transparent, directly-computed descriptors."""
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
    ]


def compute_fingerprint(mol, bit_info=None):
    """Morgan/ECFP4 fingerprint as a list of 0/1 ints.

    If bit_info is passed (an empty dict), RDKit fills it in-place with
    {bit_index: ((atom_idx, radius), ...)} so the caller can later resolve
    a bit back to the substructure that set it. Used at prediction time
    for explainability; not needed during training.
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=FP_RADIUS, nBits=FP_NBITS, bitInfo=bit_info
    )
    return list(fp)


def compute_features(mol):
    """Full feature vector: 10 descriptors + FP_NBITS fingerprint bits.
    Must exactly match compute_features() in ml_predictor.py."""
    return compute_descriptors(mol) + compute_fingerprint(mol)


def build_dataset(df: pd.DataFrame):
    """
    Builds an aggregate binary label: 1 if the molecule tested positive
    (toxic response) in AT LEAST ONE of the 12 real Tox21 assays, 0 if it
    tested negative in every assay it was measured in. Molecules with no
    valid SMILES are skipped.
    """
    X, y, kept_smiles = [], [], []

    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            continue

        assay_values = row[ASSAY_COLUMNS]
        tested = assay_values.dropna()
        if len(tested) == 0:
            continue  # this molecule wasn't actually tested in any assay

        label = 1 if (tested == 1).any() else 0

        try:
            feats = compute_features(mol)
        except Exception:
            continue

        X.append(feats)
        y.append(label)
        kept_smiles.append(row["smiles"])

    return np.array(X), np.array(y), kept_smiles


def main():
    os.makedirs("models", exist_ok=True)

    print(f"Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} rows loaded")

    print(f"Parsing molecules and computing features "
          f"(10 descriptors + {FP_NBITS}-bit ECFP4 fingerprint) ...")
    X, y, smiles_list = build_dataset(df)
    print(f"  {len(X)} molecules successfully featurized")
    print(f"  Positive (toxic in >=1 assay) rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Gradient Boosting model ...")
    model = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=6,
        learning_rate=0.08,
        l2_regularization=1.0,
        class_weight="balanced",  # dataset is imbalanced (more non-toxic than toxic)
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=42,
    )
    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Test accuracy:  {test_acc:.3f}")
    print(f"  Test ROC-AUC:   {test_auc:.3f}")

    joblib.dump(model, MODEL_OUT_PATH)
    joblib.dump(FEATURE_NAMES, FEATURE_NAMES_OUT_PATH)
    print(f"Saved model to {MODEL_OUT_PATH}")
    print(f"Saved feature name list to {FEATURE_NAMES_OUT_PATH}")


if __name__ == "__main__":
    main()
