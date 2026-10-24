
"""
toxicity_checker.py
--------------------
The core engine. Given a SMILES string, this module:

1. Parses and validates the molecule (RDKit).
2. Scans it against the structural-alert rule book (toxicity_rules.py).
3. Maps every matched atom to a risk level for coloring the 3D view.
4. Computes a transparent heuristic structural-risk score.
5. Computes molecular descriptors directly using RDKit.
6. Generates a 3D conformer for visualization.
7. Suggests alternative molecules with lower structural-risk scores.

IMPORTANT:
The structural-risk score is a heuristic based on predefined structural
alerts. It is NOT a calibrated toxicity probability and does not establish
whether a chemical is safe or toxic in real-world exposure conditions.
"""

from dataclasses import dataclass, field
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from toxicity_rules import (
    STRUCTURAL_ALERTS,
    RISK_COLORS,
    RISK_SCORE_WEIGHTS,
)


@dataclass
class AlertMatch:
    name: str
    risk_level: str
    explanation: str
    atom_indices: list = field(default_factory=list)


@dataclass
class ToxicityReport:
    smiles: str
    is_valid: bool
    error_message: Optional[str] = None
    alerts: list = field(default_factory=list)
    atom_risk: dict = field(default_factory=dict)

    # Heuristic structural concern score, not a toxicity probability.
    overall_score: float = 0.0

    # Possible labels: Low, Low-Moderate, Moderate, High.
    overall_label: str = ""

    descriptors: dict = field(default_factory=dict)
    mol_block_3d: Optional[str] = None


def _highest_risk(current: str, new: str) -> str:
    """Keep the highest severity when multiple alerts match an atom."""

    order = {
        "none": 0,
        "low": 1,
        "medium": 2,
        "high": 3,
    }

    return new if order[new] > order[current] else current


def analyze_smiles(smiles: str) -> ToxicityReport:
    """
    Analyze a molecule from its SMILES representation.

    Returns a ToxicityReport containing structural alerts, atom-level
    risk assignments, molecular descriptors, a structural-risk score,
    and a 3D molecule representation when embedding succeeds.
    """

    smiles = smiles.strip()
    mol = Chem.MolFromSmiles(smiles)

    # ---------------------------------------------------------
    # 1. Validate the SMILES
    # ---------------------------------------------------------

    if mol is None:
        return ToxicityReport(
            smiles=smiles,
            is_valid=False,
            error_message=(
                "That doesn't parse as a valid molecule. Double check "
                "the SMILES string for typos (unmatched brackets/rings "
                "are the most common issue)."
            ),
        )

    # ---------------------------------------------------------
    # 2. Initialize atom-level risk information
    # ---------------------------------------------------------

    n_atoms = mol.GetNumAtoms()

    atom_risk = {
        i: "none"
        for i in range(n_atoms)
    }

    alerts_found = []

    # ---------------------------------------------------------
    # 3. Detect structural alerts
    # ---------------------------------------------------------

    for rule in STRUCTURAL_ALERTS:

        pattern = Chem.MolFromSmarts(rule["smarts"])

        if pattern is None:
            continue

        matches = mol.GetSubstructMatches(pattern)

        if not matches:
            continue

        matched_atoms = sorted(
            {
                idx
                for match in matches
                for idx in match
            }
        )

        risk_level = rule["risk_level"]

        for idx in matched_atoms:
            atom_risk[idx] = _highest_risk(
                atom_risk[idx],
                risk_level,
            )

        alerts_found.append(
            AlertMatch(
                name=rule["name"],
                risk_level=risk_level,
                explanation=rule["explanation"],
                atom_indices=matched_atoms,
            )
        )

    # ---------------------------------------------------------
    # 4. Calculate heuristic structural-risk score
    # ---------------------------------------------------------

    # Higher values represent greater structural concern.
    # These weights are heuristic and should be validated against
    # appropriate experimental or benchmark toxicity data.
    SEVERITY_WEIGHTS = {
        "high": 8.0,
        "medium": 5.0,
        "low": 2.0,
    }

    total_heavy_atoms = mol.GetNumHeavyAtoms()

    individual_scores = []

    for alert in alerts_found:

        severity = SEVERITY_WEIGHTS.get(
            alert.risk_level,
            0.0,
        )

        # Fraction of heavy atoms matched by this alert.
        matched_heavy_atoms = sum(
            1
            for idx in set(alert.atom_indices)
            if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1
        )

        if total_heavy_atoms > 0:
            coverage = (
                matched_heavy_atoms / total_heavy_atoms
            )
        else:
            coverage = 0.0

        # Keep a baseline contribution for a serious alert while
        # allowing its molecular coverage to influence the score.
        coverage_factor = 0.5 + (0.5 * coverage)

        adjusted_score = severity * coverage_factor

        individual_scores.append(adjusted_score)

    # Combine alert contributions with diminishing returns.
    # This prevents a simple sum from exceeding the scale.
    combined_risk = 0.0

    for score in individual_scores:

        normalized_score = min(
            max(score / 10.0, 0.0),
            1.0,
        )

        combined_risk = (
            1.0
            - (1.0 - combined_risk)
            * (1.0 - normalized_score)
        )

    overall_score = round(
        combined_risk * 10.0,
        1,
    )

    # Assign a human-readable label.
    if overall_score == 0:
        overall_label = "Low"

    elif overall_score < 4:
        overall_label = "Low-Moderate"

    elif overall_score < 7:
        overall_label = "Moderate"

    else:
        overall_label = "High"

    # ---------------------------------------------------------
    # 5. Calculate molecular descriptors
    # ---------------------------------------------------------

    # These are direct RDKit calculations, not ML predictions.
    descriptors = {
        "molecular_weight": round(
            Descriptors.MolWt(mol),
            1,
        ),

        "logP (fat-solubility)": round(
            Descriptors.MolLogP(mol),
            2,
        ),

        "num_rings": rdMolDescriptors.CalcNumRings(mol),

        "num_rotatable_bonds": (
            Descriptors.NumRotatableBonds(mol)
        ),

        "num_aromatic_rings": (
            rdMolDescriptors.CalcNumAromaticRings(mol)
        ),

        "topological_polar_surface_area": round(
            Descriptors.TPSA(mol),
            1,
        ),
    }

    # ---------------------------------------------------------
    # 6. Generate the 3D molecular structure
    # ---------------------------------------------------------

    mol_3d = Chem.AddHs(mol)

    embed_status = AllChem.EmbedMolecule(
        mol_3d,
        randomSeed=42,
        useRandomCoords=True,
    )

    mol_block = None

    if embed_status == 0:

        try:
            AllChem.MMFFOptimizeMolecule(mol_3d)

        except Exception:
            # Optimization is best-effort. The embedded coordinates
            # may still be usable if optimization fails.
            pass

        mol_block = Chem.MolToMolBlock(mol_3d)

    # ---------------------------------------------------------
    # 7. Return the complete report
    # ---------------------------------------------------------

    return ToxicityReport(
        smiles=smiles,
        is_valid=True,
        alerts=alerts_found,
        atom_risk=atom_risk,
        overall_score=overall_score,
        overall_label=overall_label,
        descriptors=descriptors,
        mol_block_3d=mol_block,
    )


def suggest_alternatives(
    report: ToxicityReport,
    max_suggestions: int = 3,
):
    """
    Suggest alternative molecules by replacing recognized fragments.

    An alternative is returned only when:
    1. The replacement produces a chemically valid molecule.
    2. The resulting molecule has a strictly lower structural-risk score.

    IMPORTANT:
    A lower heuristic score does not prove that an alternative is safer.
    """

    if not report.is_valid:
        return []

    base_mol = Chem.MolFromSmiles(report.smiles)

    if base_mol is None:
        return []

    suggestions = []
    seen_smiles = {report.smiles}

    for rule in STRUCTURAL_ALERTS:

        if not rule.get("replacement_smarts"):
            continue

        pattern = Chem.MolFromSmarts(
            rule["replacement_smarts"]
        )

        if pattern is None:
            continue

        if not base_mol.HasSubstructMatch(pattern):
            continue

        replacement = Chem.MolFromSmiles(
            rule["replacement_smiles"]
        )

        if replacement is None:
            continue

        try:
            new_mols = AllChem.ReplaceSubstructs(
                base_mol,
                pattern,
                replacement,
            )

        except Exception:
            continue

        for new_mol in new_mols:

            try:
                Chem.SanitizeMol(new_mol)

            except Exception:
                continue

            new_smiles = Chem.MolToSmiles(new_mol)

            if new_smiles in seen_smiles:
                continue

            seen_smiles.add(new_smiles)

            new_report = analyze_smiles(new_smiles)

            if (
                new_report.is_valid
                and new_report.overall_score < report.overall_score
            ):

                suggestions.append(
                    {
                        "smiles": new_smiles,
                        "report": new_report,
                        "change_description": (
                            f"Replaced '{rule['name']}' with "
                            f"'{rule['replacement_name']}'"
                        ),
                    }
                )

    # Return alternatives with the lowest heuristic scores first.
    suggestions.sort(
        key=lambda suggestion: (
            suggestion["report"].overall_score
        )
    )

    return suggestions[:max_suggestions]


def build_plain_english_summary(
    report: ToxicityReport,
) -> str:
    """
    Convert a structured toxicity report into a human-readable summary.
    """

    if not report.is_valid:
        return report.error_message or "Invalid molecule."

    # ---------------------------------------------------------
    # No structural alerts detected
    # ---------------------------------------------------------

    if not report.alerts:

        return (
            "No known structural risk patterns were found in this "
            "molecule. Structural risk: "
            f"**{report.overall_label}** "
            f"(score {report.overall_score}/10). "
            "This does not guarantee safety. It means none of the "
            "specific structural patterns in the current rule book "
            "were detected. This score is not a toxicity probability."
        )

    # ---------------------------------------------------------
    # Group alerts by severity
    # ---------------------------------------------------------

    high = [
        alert.name
        for alert in report.alerts
        if alert.risk_level == "high"
    ]

    medium = [
        alert.name
        for alert in report.alerts
        if alert.risk_level == "medium"
    ]

    low = [
        alert.name
        for alert in report.alerts
        if alert.risk_level == "low"
    ]

    lines = [
        (
            f"This molecule shows **{len(report.alerts)} "
            "structural alert(s)**. "
            f"Structural risk: **{report.overall_label}** "
            f"(score {report.overall_score}/10)."
        )
    ]

    if high:
        lines.append(
            f"- High concern: {', '.join(high)}"
        )

    if medium:
        lines.append(
            f"- Moderate concern: {', '.join(medium)}"
        )

    if low:
        lines.append(
            f"- Low concern: {', '.join(low)}"
        )

    # ---------------------------------------------------------
    # Identify the most severe detected alert
    # ---------------------------------------------------------

    severity_order = {
        "high": 3,
        "medium": 2,
        "low": 1,
    }

    top_alert = max(
        report.alerts,
        key=lambda alert: severity_order.get(
            alert.risk_level,
            0,
        ),
    )

    lines.append(
        f"\n**Main structural concern:** {top_alert.name} — "
        f"{top_alert.explanation}"
    )

    lines.append(
        "\n**Interpretation:** This score is a heuristic based on "
        "the structural alerts detected. It is not a measured "
        "toxicity value or a calibrated probability of harm."
    )

    return "\n".join(lines)

