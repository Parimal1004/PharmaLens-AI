# Molecule Toxicity Explainer

Enter a molecule (as a SMILES string), see its 3D structure with atoms
colored by risk, get a plain-English explanation, and get suggestions for
safer alternatives — all driven by transparent, named rules (no black-box
model).

## Project structure

```text
molgen_toxicity/
├── requirements.txt
├── README.md
└── app/
    ├── toxicity_rules.py     # the rule book: known toxic patterns + safer swaps
    ├── toxicity_checker.py   # core engine: scans, scores, generates 3D, suggests fixes
    └── app.py                # Streamlit dashboard (the UI)
```

## Step 1 — Install Python

You need Python 3.9–3.12 installed. Check with:

```bash
python3 --version
```

If you don't have it, download from https://python.org.

## Step 2 — Set up a virtual environment (recommended, keeps things clean)

```bash
cd molgen_toxicity
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
```

## Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **streamlit** — the dashboard framework
- **rdkit** — the chemistry toolkit (parsing molecules, 3D structures, substructure matching)

Both are free and CPU-only. No GPU, no account, no API key needed.

## Step 4 — Run the app

```bash
cd app
streamlit run app.py
```

This opens a browser tab (usually `http://localhost:8501`) with the dashboard.

## Step 5 — Try it

Type a SMILES string into the input box, or pick an example from the
sidebar. Try these to see different behaviors:

| Molecule | SMILES | What you should see |
|---|---|---|
| Aspirin | `CC(=O)OC1=CC=CC=C1C(=O)O` | Low risk, no alerts |
| Nitrobenzene | `O=[N+]([O-])c1ccccc1` | High risk (nitro group flagged), suggested alternative `N#Cc1ccccc1` |
| Benzaldehyde | `O=Cc1ccccc1` | Medium risk (aldehyde), suggested alcohol alternative |
| Aniline | `Nc1ccccc1` | High risk (aromatic amine) |

If you don't know a molecule's SMILES off-hand, you can look it up for
free on **PubChem** (https://pubchem.ncbi.nlm.nih.gov) — search the
compound name, and its "Canonical SMILES" is listed on the compound page.

## Step 6 — Understand what's happening under the hood

1. **`toxicity_rules.py`** — a list of known "structural alerts" (chemical
   sub-patterns with documented toxicity concerns, e.g. nitro groups,
   epoxides, aromatic amines), each written as a SMARTS pattern (a search
   pattern for molecular substructures). Each rule also has a plain-English
   explanation and, where applicable, a simplified "safer swap."

2. **`toxicity_checker.py`** — the engine:
   - `analyze_smiles()` parses the input, checks it against every rule,
     records which atoms matched which rule, computes a transparent weighted
     score, computes a handful of plain RDKit descriptors, and generates a
     3D conformer.
   - `suggest_alternatives()` tries swapping out each flagged fragment for
     its safer counterpart and re-scores the result, returning only
     alternatives that are valid and score better.
   - `build_plain_english_summary()` turns the structured result into a
     readable paragraph.

3. **`app.py`** — the Streamlit UI. It calls the two functions above and
   renders:
   - A 3D viewer (via the free `3Dmol.js` library, loaded from a CDN) with
     atoms colored red/orange/yellow/green by risk level.
   - A summary card with the overall score and plain-English explanation.
   - An expandable breakdown of exactly which rule triggered which flag.
   - A list of suggested safer alternative molecules.

## Step 7 — The data-driven signal (trained on real Tox21 data)

In addition to the rule-based structural alerts, the app includes a
**second, independent signal**: a Gradient Boosting model trained on the
real **Tox21** dataset (8,014 molecules tested across 12 real NIH/EPA
toxicology assays covering nuclear receptor and stress-response
pathways).

**v2 (current):** the model looks at two kinds of information together:

- The original **10 named, human-readable descriptors** (molecular
  weight, fat-solubility, ring counts, etc.) — the same kind already
  shown in the "Molecular descriptors" panel.
- A **512-bit Morgan/ECFP4 fingerprint** — a bit-vector where each set
  bit means "a specific small substructure is present in this molecule."
  This is what actually lets the model tell structurally different
  molecules apart, instead of only comparing their summary stats, and is
  the main reason predictions got more accurate (test ROC-AUC 0.79 → 0.81,
  see below).

It's kept explainable despite the extra complexity:

- Every prediction is still paired with a **SHAP breakdown**. For the 10
  named descriptors, SHAP shows the feature by name, same as before. For
  a fingerprint bit, the app resolves it back to the actual substructure
  RDKit matched and shows that instead of a meaningless bit number.
- It is shown as a **second opinion alongside** the rule-based flags, not
  a replacement — the two systems capture different kinds of information
  and will sometimes disagree. That's expected, not a bug.

### The model is already included, but here's how to retrain it yourself

The trained model ships in `models/tox21_rf_model.joblib` (and a copy in
`app/models/`, which is the one the app actually loads), so the app works
out of the box. If you want to retrain it, do this:

**1. Download the real Tox21 dataset:**

```bash
cd molgen_toxicity
mkdir -p data
curl -L -o data/tox21.csv.gz "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/tox21.csv.gz"
gunzip -k data/tox21.csv.gz
```

**2. Run the training script:**

```bash
python train_model.py
```

This parses the molecules with RDKit, computes the descriptors plus the
fingerprint, trains a Gradient Boosting model, and saves the result to
`models/`.

**3. Copy the new model into `app/models/`:**

```bash
cp models/tox21_rf_model.joblib models/tox21_feature_names.joblib app/models/
```

**4. Restart the app** — it will automatically pick up the new model file.

### Going further with more data

If you want to expand this later:

- **Bigger fingerprints**: `FP_NBITS` in `train_model.py` and
  `ml_predictor.py` is currently 512. Both files must match exactly.
- **More or different data**: other public toxicity datasets can be used
  by updating the data path and assay columns in `train_model.py`.
- **Per-assay models**: separate models could be trained for each Tox21
  assay instead of using one aggregate toxicity label.

## License

No license has been specified for this project.
