"""
ml_predictor.py
-----------------
The second, independent signal: loads the Gradient Boosting model trained
on real Tox21 data (see train_model.py) and pairs every prediction with a
SHAP breakdown so it stays auditable, not a black box.

v2: the feature vector is now 10 named descriptors + a 512-bit ECFP4
Morgan fingerprint (see train_model.py for why). compute_features() here
deliberately mirrors train_model.py exactly - if the two ever drift out
of sync, predictions will be silently wrong, so keep them identical.

Explainability for the fingerprint part: a raw "bit #137 mattered" isn't
useful to a person. So for any fingerprint bit among the top SHAP
contributors, this module resolves it back to the actual substructure
that set it (RDKit's bitInfo -> the local atom environment -> that
environment's own SMILES) and shows THAT instead of a bit number. The 10
descriptor features are still shown by their plain name, as before.

Model files are located relative to this file (not the current working
directory), matching the project's documented "models/ must live inside
app/" gotcha.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem

RDLogger.DisableLog("rdApp.*")  # silence RDKit's per-molecule parse warnings

_THIS_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(_THIS_DIR, "models", "tox21_rf_model.joblib")
FEATURE_NAMES_PATH = os.path.join(_THIS_DIR, "models", "tox21_feature_names.joblib")

# Fingerprint settings - MUST match train_model.py exactly.
FP_RADIUS = 2
FP_NBITS = 512
NUM_DESCRIPTORS = 10  # the first 10 entries of FEATURE_NAMES are always the named descriptors

# Human-readable labels for the named descriptors - must stay in the same
# order as DESCRIPTOR_NAMES in train_model.py.
DISPLAY_NAMES = {
    "molecular_weight": "Molecular Weight",
    "logP": "LogP (Lipophilicity)",
    "tpsa": "Topological Polar Surface Area",
    "num_rings": "Ring Count",
    "num_aromatic_rings": "Aromatic Ring Count",
    "num_rotatable_bonds": "Rotatable Bonds",
    "num_h_donors": "H-Bond Donors",
    "num_h_acceptors": "H-Bond Acceptors",
    "num_heteroatoms": "Heteroatom Count",
    "fraction_csp3": "Fraction sp3 Carbons",
}


@dataclass
class MLPrediction:
    is_available: bool
    error_message: Optional[str] = None
    toxicity_probability: float = 0.0
    base_rate: float = 0.0
    shap_contributions: list = field(default_factory=list)  # [(feature, value, shap), ...] sorted by |shap| desc


def compute_descriptors(mol):
    """Must exactly match compute_descriptors() in train_model.py."""
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


def compute_features(mol, bit_info=None):
    """Must exactly match compute_features() in train_model.py."""
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=FP_RADIUS, nBits=FP_NBITS, bitInfo=bit_info
    )
    return compute_descriptors(mol) + list(fp)


def _describe_fingerprint_bit(mol, bit_index, bit_info):
    """Turn a fingerprint bit index into a human-readable substructure,
    e.g. 'fp_bit_312' -> 'matched substructure: c1ccccc1 (aromatic ring)'.
    Falls back to the raw bit number if RDKit can't resolve it (e.g. the
    bit was never set for this molecule, or radius 0 with no bonds)."""
    if bit_index not in bit_info:
        return f"Structural pattern #{bit_index}"

    atom_idx, radius = bit_info[bit_index][0]  # first environment that set this bit
    if radius == 0:
        atom = mol.GetAtomWithIdx(atom_idx)
        return f"Substructure: single {atom.GetSymbol()} atom"

    try:
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
        amap = {}
        submol = Chem.PathToSubmol(mol, env, atomMap=amap)
        frag_smiles = Chem.MolToSmiles(submol)
        return f"Substructure: {frag_smiles}"
    except Exception:
        return f"Structural pattern #{bit_index}"


_model_cache = {"model": None, "feature_names": None, "explainer": None, "load_error": None}


def _load():
    """Lazy-load the model + a SHAP TreeExplainer once per process."""
    if _model_cache["model"] is not None or _model_cache["load_error"] is not None:
        return

    if not (os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_NAMES_PATH)):
        _model_cache["load_error"] = (
            "No trained model found. Run `python train_model.py` first "
            "(see README Step 7) to generate models/tox21_rf_model.joblib."
        )
        return

    try:
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        import shap  # imported lazily - shap has a heavier import cost

        explainer = shap.TreeExplainer(model)
        _model_cache["model"] = model
        _model_cache["feature_names"] = feature_names
        _model_cache["explainer"] = explainer
    except Exception as exc:  # pragma: no cover - defensive, surfaced to the UI
        _model_cache["load_error"] = f"Could not load the ML model: {exc}"


def predict_toxicity(smiles: str) -> MLPrediction:
    """
    Runs the trained model on a SMILES string and returns a probability
    plus a per-feature SHAP breakdown (named descriptors shown by name;
    fingerprint bits resolved back to the substructure they represent).
    Returns is_available=False (with an explanatory message) if the model
    hasn't been trained yet, rather than crashing the dashboard - the
    rule-based signal still works fine on its own.
    """
    _load()
    if _model_cache["load_error"] is not None:
        return MLPrediction(is_available=False, error_message=_model_cache["load_error"])

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return MLPrediction(is_available=False, error_message="Invalid SMILES.")

    model = _model_cache["model"]
    feature_names = _model_cache["feature_names"]
    explainer = _model_cache["explainer"]

    try:
        bit_info = {}
        feats = np.array([compute_features(mol, bit_info=bit_info)])
    except Exception as exc:
        return MLPrediction(is_available=False, error_message=f"Could not featurize molecule: {exc}")

    proba = model.predict_proba(feats)[0, 1]

    shap_values = explainer.shap_values(feats)
    # shap.TreeExplainer has returned this in a few different shapes across
    # shap versions and model types - handle all:
    #   (a) list [class0_array, class1_array], each (n_samples, n_features)
    #   (b) a single 3D array (n_samples, n_features, n_classes)
    #   (c) a single 2D array (n_samples, n_features) - already class 1
    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[1])[0]
    elif np.asarray(shap_values).ndim == 3:
        sv = np.asarray(shap_values)[0, :, 1]
    else:
        sv = np.asarray(shap_values)[0]
    sv = np.ravel(sv)

    base_value = explainer.expected_value
    base_value = np.ravel(np.asarray(base_value))
    base_value = float(base_value[1]) if base_value.size > 1 else float(base_value[0])

    contributions = []
    for i, (name, val, contrib) in enumerate(zip(feature_names, feats[0], sv)):
        if i < NUM_DESCRIPTORS:
            display_name = DISPLAY_NAMES.get(name, name)
        else:
            bit_index = i - NUM_DESCRIPTORS
            if val > 0.5:  # bit is set (present) for this molecule
                display_name = _describe_fingerprint_bit(mol, bit_index, bit_info)
            else:
                display_name = f"Structural pattern #{bit_index} (absent)"
        contributions.append((display_name, float(val), float(contrib)))
    contributions.sort(key=lambda c: abs(c[2]), reverse=True)

    return MLPrediction(
        is_available=True,
        toxicity_probability=float(proba),
        base_rate=base_value,
        shap_contributions=contributions,
    )
