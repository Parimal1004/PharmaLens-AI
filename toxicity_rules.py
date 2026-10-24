"""
toxicity_rules.py
------------------
A small, hand-curated library of "structural alerts": chemical sub-patterns
that are well documented in the toxicology literature as being linked to
specific safety concerns (liver toxicity, mutagenicity, skin sensitization,
etc.).

Each pattern is written as SMARTS (a search pattern for molecular
substructures, similar in spirit to a regex for chemistry). RDKit uses
this to find *exactly which atoms* in a user's molecule match a known
risky pattern.

This is intentionally rule-based rather than a trained deep-learning
model. That is the point: every flag traces back to a named, citable
pattern, so nothing here is a "black box" - the code can always answer
"why" a molecule was flagged.

risk_level: "high", "medium", or "low" - drives both the score and the
            atom coloring in the 3D viewer.
replacement: an optional SMARTS -> SMILES fragment swap representing a
             commonly used "safer" substitute for that pattern
             (a simplified bioisostere lookup, not a full retrosynthesis
             engine).
"""

STRUCTURAL_ALERTS = [
    {
        "name": "Nitro group",
        "smarts": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
        "risk_level": "high",
        "explanation": (
            "Nitro groups are frequently linked to mutagenicity and liver "
            "toxicity. They are metabolized into reactive intermediates "
            "that can damage DNA or proteins."
        ),
        # NOTE: detection above uses a single-atom recursive SMARTS (fine for
        # flagging/coloring). Replacement needs a full multi-atom match so
        # RDKit knows exactly which atoms to cut out. RDKit always
        # canonicalizes nitro groups to the charged form [N+](=O)[O-],
        # regardless of how the input SMILES was written, so we match that
        # specific 3-atom shape here.
        "replacement_smarts": "[NX3+](=[OX1])[OX1-]",
        "replacement_smiles": "C#N",  # swap nitro -> nitrile as a simple demo substitute
        "replacement_name": "nitrile group",
    },
    {
        "name": "Aromatic nitro / aniline-type amine",
        "smarts": "[NX3;H2,H1;!$(NC=O)]c1ccccc1",
        "risk_level": "high",
        "explanation": (
            "Aromatic amines (anilines) are a classic mutagenicity alert; "
            "several are known or suspected carcinogens after metabolic "
            "activation."
        ),
        "replacement_smarts": "[NX3;H2,H1;!$(NC=O)]c1ccccc1",
        "replacement_smiles": "NC(=O)c1ccccc1",  # amide instead of free aniline
        "replacement_name": "amide-protected amine",
    },
    {
        "name": "Michael acceptor (alpha,beta-unsaturated carbonyl)",
        "smarts": "[CX3]=[CX3][CX3]=[OX1]",
        "risk_level": "high",
        "explanation": (
            "This reactive pattern can form covalent bonds with proteins "
            "or DNA (a 'Michael acceptor'), a common cause of unexpected "
            "toxicity and immune reactions."
        ),
        "replacement_smarts": None,
        "replacement_smiles": None,
        "replacement_name": None,
    },
    {
        "name": "Epoxide",
        "smarts": "[OX2]1[CX4][CX4]1",
        "risk_level": "high",
        "explanation": (
            "Epoxides are strained, reactive rings that readily react with "
            "DNA and proteins, a well-known mutagenicity/carcinogenicity "
            "alert."
        ),
        "replacement_smarts": None,
        "replacement_smiles": None,
        "replacement_name": None,
    },
    {
        "name": "Aldehyde",
        "smarts": "[CX3H1](=O)[#6]",
        "risk_level": "medium",
        "explanation": (
            "Aldehydes are chemically reactive and can form unwanted "
            "covalent adducts with biological molecules, sometimes "
            "causing irritation or sensitization."
        ),
        # Only match the aldehyde C=O itself, not the atom it's attached to -
        # matching a neighboring ring atom here would delete part of the
        # ring on replacement.
        "replacement_smarts": "[CX3H1]=[OX1]",
        "replacement_smiles": "CO",  # simplified: aldehyde -> alcohol
        "replacement_name": "alcohol group",
    },
    {
        "name": "Halogenated alkane (alkyl halide)",
        "smarts": "[CX4][F,Cl,Br,I]",
        "risk_level": "medium",
        "explanation": (
            "Alkyl halides can act as alkylating agents, reacting with DNA "
            "or proteins. Risk depends heavily on context, so this is "
            "flagged as medium rather than high."
        ),
        "replacement_smarts": None,
        "replacement_smiles": None,
        "replacement_name": None,
    },
    {
        "name": "Thiourea",
        "smarts": "[#7][CX3](=S)[#7]",
        "risk_level": "medium",
        "explanation": (
            "Thioureas have been associated with thyroid toxicity in some "
            "studies and are commonly screened as a caution flag."
        ),
        "replacement_smarts": "[#7][CX3](=S)[#7]",
        "replacement_smiles": "NC(=O)N",  # thiourea -> urea
        "replacement_name": "urea group",
    },
    {
        "name": "Hydrazine",
        "smarts": "[NX3][NX3]",
        "risk_level": "high",
        "explanation": (
            "Hydrazine and hydrazine derivatives are well-documented "
            "mutagens and are toxic to the liver in many species."
        ),
        "replacement_smarts": None,
        "replacement_smiles": None,
        "replacement_name": None,
    },
    {
        "name": "Polycyclic aromatic (fused benzene rings)",
        "smarts": "c1ccc2ccccc2c1",
        "risk_level": "low",
        "explanation": (
            "Simple fused aromatic rings are common in many safe drugs, "
            "but larger polycyclic aromatics are sometimes flagged for "
            "potential DNA intercalation. Marked as low risk here since "
            "context (ring count, substitution) matters a great deal."
        ),
        "replacement_smarts": None,
        "replacement_smiles": None,
        "replacement_name": None,
    },
]

# Colors used in the 3D viewer, keyed by risk level.
RISK_COLORS = {
    "high": "red",
    "medium": "orange",
    "low": "yellow",
    "none": "lightgreen",
}

RISK_SCORE_WEIGHTS = {
    "high": 3,
    "medium": 2,
    "low": 1,
}
