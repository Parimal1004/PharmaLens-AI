# PharmaLens AI

PharmaLens AI is a Streamlit application for transparent molecule toxicity
analysis. Enter a molecule as a SMILES string to view its risk level,
structural alerts, molecular descriptors, 3D structure, and safer alternatives.
The application combines explainable chemistry rules with a machine-learning
signal trained on the Tox21 dataset.

## Deployed App

**Live demo:** [pharmalens-ai-app](https://pharmalens-ai-miniproject.streamlit.app/)

## Features

- Analyze molecules from SMILES strings
- Identify explainable structural toxicity alerts
- Display molecular descriptors and a 3D molecular view
- Provide plain-English analysis and safer alternatives
- Generate candidate molecules
- Compare rule-based results with a Tox21-trained ML prediction

## Run Locally

### Requirements

- Python 3.9–3.12
- pip

### Installation

```bash
git clone https://github.com/Parimal1004/PharmaLens-AI.git
cd PharmaLens-AI
python -m venv venv
```

Activate the virtual environment:

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

Install dependencies and start the application:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

The application will be available at `http://localhost:8501`.

## Project Structure

```text
app/
├── app.py                 # Streamlit interface
├── molecule_generator.py  # Candidate molecule generation
├── ml_predictor.py        # Tox21-based prediction
├── toxicity_checker.py    # Molecule analysis
└── toxicity_rules.py      # Explainable structural alerts
data/                      # Training data
models/                    # Trained model files
requirements.txt           # Python dependencies
```

## Disclaimer

PharmaLens AI is intended for research and educational use. Its predictions
are not a substitute for laboratory testing, clinical evaluation, or
professional medical advice.

## License

No license has been specified for this project.
