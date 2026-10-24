"""
app.py
------
PharmaLens AI - Transparent Drug Discovery Platform
Restyled to match the Google Stitch dashboard mockup (see ../DESIGN.md
and ../code.html), on top of the existing, tested analysis pipeline.

Run with:

    streamlit run app.py

Layout (matches the 3-section Stitch design):
  1. Generate new candidate molecules  - LSTM generator -> validated grid
  2. Analyze a molecule in depth       - rule-based alerts + ML/SHAP signal
  3. Safer alternatives                - side-by-side 3D comparison

Every generated or analyzed molecule flows through the exact same
validated pipeline: generate -> validate -> explain -> visualize -> refine.
"""

import base64
import os

import streamlit as st
import streamlit.components.v1 as components

from toxicity_checker import analyze_smiles, suggest_alternatives, build_plain_english_summary
from toxicity_rules import RISK_COLORS
import ml_predictor
import molecule_generator
import styles

st.set_page_config(page_title="PharmaLens AI - Transparent Drug Discovery Platform", layout="wide")
st.markdown(styles.global_css(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
_logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.svg")
if os.path.exists(_logo_path):
    with open(_logo_path, "rb") as f:
        _logo_b64 = base64.b64encode(f.read()).decode("utf-8")
    st.markdown(styles.header_bar(_logo_b64), unsafe_allow_html=True)
else:
    st.title("PharmaLens AI")
    st.caption("Transparent Drug Discovery Platform")


# ---------------------------------------------------------------------------
# Helper: 3Dmol.js viewer (extended from the original app.py with optional
# per-atom element-symbol labels, e.g. "N+", "O", "Cl" - matching the
# Stitch mockup's labeled-sphere style)
# ---------------------------------------------------------------------------
def render_3d_viewer(
    mol_block: str,
    atom_risk: dict,
    height: int = 380,
    viewer_id: str = "viewer",
    show_labels: bool = True,
    label_carbons: bool = False,
    label_hydrogens: bool = False,
) -> str:
    color_map_js = ", ".join(
        f'"{idx}": "{RISK_COLORS.get(risk, "lightgreen")}"'
        for idx, risk in atom_risk.items()
    )
    safe_mol_block = mol_block.replace("`", "'").replace("\\", "\\\\")

    # Which elements to skip labeling - by default carbon (implied by the
    # unlabeled backbone convention) and hydrogen (too many, too cluttered).
    excluded_elems = []
    if not label_carbons:
        excluded_elems.append("C")
    if not label_hydrogens:
        excluded_elems.append("H")
    excluded_js = ", ".join(f"'{e}'" for e in excluded_elems)

    label_js = ""
    if show_labels:
        label_js = f"""
      // Label each non-excluded atom with its element symbol (+ formal
      // charge, if any) - e.g. "N+", "O-" - matching the reference mockup.
      const excludedElems_{viewer_id} = [{excluded_js}];
      const modelAtoms_{viewer_id} = viewer_{viewer_id}.getModel().selectedAtoms({{}});
      modelAtoms_{viewer_id}.forEach(function(atom) {{
        if (excludedElems_{viewer_id}.includes(atom.elem)) {{ return; }}
        let labelText = atom.elem;
        if (atom.charge > 0) {{ labelText += "+"; }}
        else if (atom.charge < 0) {{ labelText += "-"; }}
        viewer_{viewer_id}.addLabel(labelText, {{
          position: {{x: atom.x, y: atom.y, z: atom.z}},
          font: "Inter, sans-serif",
          fontSize: 12,
          fontColor: "white",
          showBackground: false,
          inFront: true
        }});
      }});
"""

    html = f"""
    <div id="{viewer_id}" style="height: {height}px; width: 100%; position: relative;
         background:#F8FAFC; border-radius: 12px;"></div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
    <script>
      const molData_{viewer_id} = `{safe_mol_block}`;
      const riskColors_{viewer_id} = {{{color_map_js}}};

      let viewer_{viewer_id} = $3Dmol.createViewer(document.getElementById("{viewer_id}"), {{
        backgroundColor: "0xF8FAFC"
      }});
      viewer_{viewer_id}.addModel(molData_{viewer_id}, "mol");
      viewer_{viewer_id}.setStyle({{}}, {{
        stick: {{colorscheme: {{prop: "index", map: riskColors_{viewer_id}}}, radius: 0.15}},
        sphere: {{colorscheme: {{prop: "index", map: riskColors_{viewer_id}}}, scale: 0.28}}
      }});
      {label_js}
      viewer_{viewer_id}.zoomTo();
      viewer_{viewer_id}.render();
    </script>
    """
    return html


def render_risk_legend():
    legend_cols = st.columns(4)
    labels = [("High risk", "high"), ("Medium risk", "medium"), ("Low risk", "low"), ("No flag", "none")]
    for (label, level), c in zip(labels, legend_cols):
        c.markdown(styles.risk_badge(level, label_override=label), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "smiles_value" not in st.session_state:
    st.session_state.smiles_value = ""
if "generated_candidates" not in st.session_state:
    st.session_state.generated_candidates = []


def set_smiles(new_value: str):
    st.session_state.smiles_value = new_value


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f"<span style='font-family:Merriweather,serif; font-weight:700; color:{styles.COLORS['primary']}; "
    f"font-size:1.05rem;'>Example Molecules</span>",
    unsafe_allow_html=True,
)

examples = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Nitrobenzene": "O=[N+]([O-])c1ccccc1",
    "Benzaldehyde": "O=Cc1ccccc1",
    "Aniline": "Nc1ccccc1",
    "Thiourea example": "NC(=S)Nc1ccccc1",
}
for label, smi in examples.items():
    if st.sidebar.button(label, key=f"example_{label}", use_container_width=True):
        set_smiles(smi)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
<div style="background:{styles.COLORS['surface']}; border:1px solid {styles.COLORS['border']};
            border-radius:1rem; padding:0.9rem;">
  <div style="font-family:Merriweather,serif; font-weight:700; color:{styles.COLORS['primary']};
              font-size:0.95rem; margin-bottom:0.5rem;">Why this isn't a black box</div>
  <ul style="padding-left:1rem; margin:0; font-size:0.82rem; color:{styles.COLORS['on_surface_variant']};">
    <li style="margin-bottom:0.4rem;"><b>Rule-based alerts</b> flag known, named toxicophores instantly.</li>
    <li style="margin-bottom:0.4rem;"><b>SHAP attribution</b> shows the exact weight of every feature.</li>
    <li>Every explanation traces back to visible data - nothing is guessed.</li>
  </ul>
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.caption("Quick load SMILES")
st.sidebar.text_input("SMILES", key="sidebar_smiles_display", value=st.session_state.smiles_value, disabled=True, label_visibility="collapsed")


# ===========================================================================
# SECTION 1 - Generate new candidate molecules
# ===========================================================================
with st.container(border=True):
    st.markdown("### 1. Generate new candidate molecules")
    st.caption(
        "A small character-level LSTM proposes new SMILES strings. Every candidate is "
        "immediately RDKit-validated before you ever see it - invalid attempts are discarded."
    )

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 0.6])
    with ctrl_col1:
        n_candidates = st.slider("Number of candidates", 1, 8, 4)
    with ctrl_col2:
        temperature = st.slider("Creativity / Temperature", 0.5, 1.3, 0.85, step=0.05)
    with ctrl_col3:
        st.markdown("<div style='height:1.6rem;'></div>", unsafe_allow_html=True)
        generate_clicked = st.button("🎲 Generate Candidates", use_container_width=True)

    if generate_clicked:
        try:
            with st.spinner("Sampling and validating candidates..."):
                candidates = molecule_generator.generate_candidates(n_candidates, temperature)
            st.session_state.generated_candidates = candidates
            if not candidates:
                st.warning("No valid molecules were generated this round - try again or raise the temperature slightly.")
        except RuntimeError as exc:
            st.info(f"Generator not available yet: {exc}")

    if st.session_state.generated_candidates:
        cand_cols = st.columns(min(len(st.session_state.generated_candidates), 4))
        for i, cand in enumerate(st.session_state.generated_candidates):
            col = cand_cols[i % len(cand_cols)]
            with col:
                cand_report = analyze_smiles(cand.smiles)
                with st.container(border=True):
                    st.markdown(
                        f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                        f"<span style='font-family:Merriweather,serif; font-weight:700; color:{styles.COLORS['primary']};'>PL-{400+i}</span>"
                        f"{styles.score_pill(cand_report.overall_score) if cand_report.is_valid else ''}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(styles.mono(cand.smiles), unsafe_allow_html=True)
                    if cand_report.is_valid and cand_report.mol_block_3d:
                        components.html(
                            render_3d_viewer(cand_report.mol_block_3d, cand_report.atom_risk, height=200, viewer_id=f"cand_viewer_{i}"),
                            height=205,
                        )
                    st.button(
                        "🔍 Analyze this candidate",
                        key=f"analyze_cand_{i}",
                        on_click=set_smiles,
                        args=(cand.smiles,),
                        use_container_width=True,
                    )


# ===========================================================================
# SECTION 2 - Analyze a molecule in depth
# ===========================================================================
with st.container(border=True):
    st.markdown("### 2. Analyze a molecule in depth")

    with st.expander("ℹ️ How this decision was made (methodology, not a black box)"):
        st.markdown(
            """
- **The flags are rules, not predictions.** Each one is a named, published chemical pattern
  (e.g. *nitro group*, *epoxide*) matched directly against your molecule's structure using RDKit's substructure search.
- **The score is a visible formula.** `overall_score = sum of risk weights for every matched pattern`, capped at 10.
  No hidden neural network weights are involved.
- **The ML signal is a second, separately-labeled opinion.** It's a Random Forest trained on real Tox21 data,
  paired with SHAP so every prediction is broken down feature-by-feature - never conflated with the rule-based score.
- **Limitation, stated plainly:** the rule book is a small curated starting set, not an exhaustive toxicology database.
  A "no flags found" result means no *known pattern in this list* matched - it is not a guarantee of safety.
            """
        )

    smiles_input = st.text_input(
        "SMILES string",
        key="smiles_value",
        placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O  (this is aspirin)",
    )

    if not smiles_input:
        st.info("Enter a SMILES string above, pick an example from the sidebar, or generate a candidate above to begin.")
        report = None
    else:
        report = analyze_smiles(smiles_input)
        if not report.is_valid:
            st.error(report.error_message)
            report = None

    if report is not None:
        col1, col2 = st.columns([1.1, 1])

        # --- Left: 3D viewer + descriptors ---
        with col1:
            viewer_header_col, viewer_toggle_col = st.columns([2, 1])
            viewer_header_col.markdown("**3D structure (colored by risk)**")
            show_atom_labels = viewer_toggle_col.checkbox("Show element labels", value=True, key="show_atom_labels")
            if report.mol_block_3d:
                components.html(
                    render_3d_viewer(
                        report.mol_block_3d, report.atom_risk, viewer_id="main_viewer",
                        show_labels=show_atom_labels,
                    ),
                    height=390,
                )
                render_risk_legend()
            else:
                st.warning("Could not generate a 3D structure for this molecule.")

            with st.expander("Molecular descriptors (raw, transparent calculations)"):
                st.json(report.descriptors)

        # --- Right: rule-based summary + alerts + ML/SHAP ---
        with col2:
            st.markdown("**Summary**")
            st.markdown(styles.score_pill(report.overall_score), unsafe_allow_html=True)
            st.markdown(build_plain_english_summary(report))

            if report.alerts:
                st.markdown("**Why this result? (factor breakdown)**")
                for alert in report.alerts:
                    icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}[alert.risk_level]
                    with st.expander(f"{icon} {alert.name} — {alert.risk_level} risk"):
                        st.write(alert.explanation)
                        st.caption(f"Matched atom indices: {alert.atom_indices}")

        # --- Full width: ML / SHAP signal ---
        st.markdown("---")
        st.markdown("**Data-driven signal — Random Forest (trained on real Tox21 data)**")
        ml_result = ml_predictor.predict_toxicity(smiles_input)

        if not ml_result.is_available:
            st.info(f"ML signal not available yet: {ml_result.error_message}")
        else:
            proba_pct = ml_result.toxicity_probability * 100
            st.progress(min(max(ml_result.toxicity_probability, 0.0), 1.0), text=f"Estimated toxicity probability: {proba_pct:.1f}%")

            with st.expander("Why did the model predict this? (SHAP breakdown)"):
                max_abs = max((abs(c[2]) for c in ml_result.shap_contributions), default=0.0001)
                bars_html = "".join(
                    styles.shap_bar(name, val, contrib, max_abs)
                    for name, val, contrib in ml_result.shap_contributions[:8]
                )
                st.markdown(bars_html, unsafe_allow_html=True)
                st.caption(
                    "Red bars increase predicted toxicity risk, green bars decrease it. "
                    "This is a separate, post-hoc signal alongside the rule-based alerts above - "
                    "the two sometimes disagree, which is expected, not a bug."
                )


# ===========================================================================
# SECTION 3 - Safer alternatives, compare in 3D
# ===========================================================================
with st.container(border=True):
    st.markdown("### 3. Safer alternatives — compare in 3D")

    if report is None:
        st.caption("Analyze a molecule above to see suggested safer alternatives here.")
    else:
        alternatives = suggest_alternatives(report)

        if not alternatives:
            if report.alerts:
                st.write("No automatic safer substitute was found for the flagged group(s) in this simplified rule book.")
            else:
                st.write("No changes suggested — no structural alerts were found.")
        else:
            st.caption(f"Showing {len(alternatives)} alternative(s), best (lowest risk) first.")
            alt_cols = st.columns(len(alternatives))

            for i, (alt, col) in enumerate(zip(alternatives, alt_cols)):
                with col:
                    alt_report = alt["report"]
                    score_delta = report.overall_score - alt_report.overall_score

                    with st.container(border=True):
                        st.markdown(f"**Option {i + 1}**")
                        st.caption(alt["change_description"])

                        before_after = (
                            f"<div style='display:flex; align-items:center; justify-content:space-between; "
                            f"padding:0.5rem; background:{styles.COLORS['surface']}; border-radius:0.6rem; margin-bottom:0.5rem;'>"
                            f"{styles.score_pill(report.overall_score)}"
                            f"<span style='color:{styles.COLORS['secondary']};'>&rarr;</span>"
                            f"{styles.score_pill(alt_report.overall_score)}"
                            f"</div>"
                        )
                        st.markdown(before_after, unsafe_allow_html=True)

                        if alt_report.mol_block_3d:
                            components.html(
                                render_3d_viewer(alt_report.mol_block_3d, alt_report.atom_risk, height=260, viewer_id=f"alt_viewer_{i}"),
                                height=265,
                            )
                        else:
                            st.warning("3D structure unavailable for this candidate.")

                        st.markdown(styles.mono(alt["smiles"]), unsafe_allow_html=True)

                        st.button(
                            "🔄 Analyze this molecule instead",
                            key=f"use_alt_{i}",
                            on_click=set_smiles,
                            args=(alt["smiles"],),
                            use_container_width=True,
                        )

            render_risk_legend()


st.markdown(styles.footer_bar(), unsafe_allow_html=True)
