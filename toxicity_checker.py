"""
toxicity_checker.py
--------------------
The core engine. Given a SMILES string, this module:

1. Parses and validates the molecule (RDKit).
2. Scans it against the structural-alert rule book (toxicity_rules.py).
3. Maps every matched atom to a risk level (for coloring the 3D view).
4. Computes a simple, transparent overall risk score.
5. Computes a handful of plain molecular descriptors (size, "greasiness",
   etc.) that also feed into the explanation card - these are NOT from a
   trained black-box model, they are direct RDKit calculations, so they
   can always be explained in one sentence.
6. Generates a 3D conformer (a plausible 3D shape) for visualization.
7. Suggests safer alternative molecules by swapping out flagged fragments.

Every function returns plain Python data (dicts/lists) so the Streamlit
app layer doesn't need to know anything about RDKit internals.
"""

from dataclasses import dataclass, field
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from toxicity_rules import STRUCTURAL_ALERTS, RISK_COLORS, RISK_SCORE_WEIGHTS


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
    alerts: list = field(default_factory=list)          # list[AlertMatch]
    atom_risk: dict = field(default_factory=dict)        # {atom_idx: risk_level}
    overall_score: float = 0.0                           # 0 (safe) - 10 (high risk)
    overall_label: str = ""                               # "Low" / "Moderate" / "High"
    descriptors: dict = field(default_factory=dict)
    mol_block_3d: Optional[str] = None                    # 3D structure, MDL molblock format


def _highest_risk(current: str, new: str) -> str:
    """When an atom is matched by more than one alert, keep the worse rating."""
    order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    return new if order[new] > order[current] else current


def analyze_smiles(smiles: str) -> ToxicityReport:
    """Main entry point: run the full toxicity analysis on a SMILES string."""
    smiles = smiles.strip()
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return ToxicityReport(
            smiles=smiles,
            is_valid=False,
            error_message=(
                "That doesn't parse as a valid molecule. Double check the "
                "SMILES string for typos (unmatched brackets/rings are the "
                "most common issue)."
            ),
        )

    n_atoms = mol.GetNumAtoms()
    atom_risk = {i: "none" for i in range(n_atoms)}
    alerts_found = []

    for rule in STRUCTURAL_ALERTS:
        pattern = Chem.MolFromSmarts(rule["smarts"])
        if pattern is None:
            continue  # skip malformed pattern definitions defensively
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            matched_atoms = sorted({idx for match in matches for idx in match})
            for idx in matched_atoms:
                atom_risk[idx] = _highest_risk(atom_risk[idx], rule["risk_level"])
            alerts_found.append(
                AlertMatch(
                    name=rule["name"],
                    risk_level=rule["risk_level"],
                    explanation=rule["explanation"],
                    atom_indices=matched_atoms,
                )
            )

    # --- Overall score: simple, transparent weighted sum, capped at 10 ---
    raw_score = sum(RISK_SCORE_WEIGHTS[a.risk_level] for a in alerts_found)
    overall_score = min(raw_score, 10)
    if overall_score == 0:
        overall_label = "Low"
    elif overall_score <= 3:
        overall_label = "Low-Moderate"
    elif overall_score <= 6:
        overall_label = "Moderate"
    else:
        overall_label = "High"

    # --- Plain descriptors (transparent, directly computed, no ML) ---
    descriptors = {
        "molecular_weight": round(Descriptors.MolWt(mol), 1),
        "logP (fat-solubility)": round(Descriptors.MolLogP(mol), 2),
        "num_rings": rdMolDescriptors.CalcNumRings(mol),
        "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
        "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "topological_polar_surface_area": round(Descriptors.TPSA(mol), 1),
    }

    # --- 3D structure generation ---
    mol_3d = Chem.AddHs(mol)
    embed_status = AllChem.EmbedMolecule(mol_3d, randomSeed=42, useRandomCoords=True)
    mol_block = None
    if embed_status == 0:
        try:
            AllChem.MMFFOptimizeMolecule(mol_3d)
        except Exception:
            pass  # optimization is best-effort; unoptimized 3D coords still usable
        mol_block = Chem.MolToMolBlock(mol_3d)

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


def suggest_alternatives(report: ToxicityReport, max_suggestions: int = 3):
    """
    For every high/medium-risk alert that has a known safer replacement,
    try swapping that fragment out and re-score the result. Only
    alternatives that (a) are chemically valid and (b) score strictly
    better are returned.
    """
    if not report.is_valid:
        return []

    base_mol = Chem.MolFromSmiles(report.smiles)
    suggestions = []
    seen_smiles = {report.smiles}

    for rule in STRUCTURAL_ALERTS:
        if not rule.get("replacement_smarts"):
            continue
        pattern = Chem.MolFromSmarts(rule["replacement_smarts"])
        if pattern is None or not base_mol.HasSubstructMatch(pattern):
            continue

        replacement = Chem.MolFromSmiles(rule["replacement_smiles"])
        if replacement is None:
            continue

        try:
            new_mols = AllChem.ReplaceSubstructs(base_mol, pattern, replacement)
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
            if new_report.is_valid and new_report.overall_score < report.overall_score:
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

    # Best (lowest score) suggestions first
    suggestions.sort(key=lambda s: s["report"].overall_score)
    return suggestions[:max_suggestions]


def build_plain_english_summary(report: ToxicityReport) -> str:
    """Turns the structured report into a short, human-readable paragraph."""
    if not report.is_valid:
        return report.error_message

    if not report.alerts:
        return (
            f"No known structural risk patterns were found in this molecule. "
            f"Overall risk: **{report.overall_label}** "
            f"(score {report.overall_score}/10). This does not guarantee "
            f"safety - it means none of the specific known-risk patterns in "
            f"our rule book were detected."
        )

    high = [a.name for a in report.alerts if a.risk_level == "high"]
    medium = [a.name for a in report.alerts if a.risk_level == "medium"]
    low = [a.name for a in report.alerts if a.risk_level == "low"]

    lines = [
        f"This molecule shows **{len(report.alerts)} structural alert(s)**. "
        f"Overall risk: **{report.overall_label}** (score {report.overall_score}/10)."
    ]
    if high:
        lines.append(f"- High concern: {', '.join(high)}")
    if medium:
        lines.append(f"- Moderate concern: {', '.join(medium)}")
    if low:
        lines.append(f"- Low concern: {', '.join(low)}")

    top_alert = max(report.alerts, key=lambda a: RISK_SCORE_WEIGHTS[a.risk_level])
    lines.append(f"\n**Main driver:** {top_alert.name} — {top_alert.explanation}")

    return "\n".join(lines)
