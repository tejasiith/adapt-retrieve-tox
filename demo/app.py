import html
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import streamlit as st
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

HERE = Path(__file__).resolve().parent
MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"
MAX_LEN = 128
K = 5

NAME = {1: "Mutagenic (Ames positive)", 0: "Non-mutagenic (Ames negative)"}
GREEN, RED, AMBER = "#1b7f3b", "#b3261e", "#9a6700"

st.set_page_config(page_title="AMES: LoRA ChemBERTa with a retrieval check",
                   layout="centered")


@st.cache_resource(show_spinner="Loading the model (the first start takes a minute)...")
def load_resources():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = PeftModel.from_pretrained(base, str(HERE / "adapter"))
    model.eval()

    train = pd.read_csv(HERE / "train.csv")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols = [Chem.MolFromSmiles(s) for s in train.smiles]
    train_canon = np.array([Chem.MolToSmiles(m) for m in mols])
    train_fp = np.stack([gen.GetFingerprintAsNumPy(m).astype(np.float32) for m in mols])
    return tok, model, train, gen, train_canon, train_fp


tok, model, train, gen, train_canon, train_fp = load_resources()
train_sum = train_fp.sum(1)


def fingerprint(mol):
    return gen.GetFingerprintAsNumPy(mol).astype(np.float32)


def card(color, title, body=""):
    return (f"<div style='border-left:6px solid {color};background:{color}22;"
            f"padding:10px 14px;margin:8px 0;border-radius:6px'>"
            f"<b style='color:{color}'>{title}</b>"
            f"<div style='margin-top:4px'>{body}</div></div>")


def analyse(smiles):
    """Returns (html_string, neighbours_dataframe_or_None)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return card(AMBER, "Could not parse this SMILES string. Please check it."), None
    canon = Chem.MolToSmiles(mol)

    # 1) Model prediction
    enc = tok(canon, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits
    prob = float(torch.softmax(logits, -1)[0, 1])
    pred = int(prob >= 0.5)

    # 2) Neighbours
    q = fingerprint(mol)
    inter = train_fp @ q
    sim = inter / np.maximum(train_sum + q.sum() - inter, 1e-9)
    exact = train_canon == canon
    in_train = bool(exact.any())
    matched_label = int(train.label[exact].iloc[0]) if in_train else None
    sim_search = sim.copy()
    sim_search[exact] = -1.0
    top = np.argsort(-sim_search)[:K]
    nb = train.iloc[top].copy()
    nb["similarity"] = np.round(sim_search[top].astype(np.float64), 3)
    nb["label"] = nb["label"].astype(int)

    # 3) Do the neighbours agree with the model?
    n_pos = int(nb["label"].sum())
    nb_pred = int(nb["label"].mean() >= 0.5)
    agree = nb_pred == pred

    color = RED if pred == 1 else GREEN
    out = [card(color, f"Model prediction: {NAME[pred]}",
                f"Probability of mutagenic: <b>{prob:.3f}</b>")]
    out.append(
        f"<div style='margin:4px 0'>Canonical SMILES: <code>{html.escape(canon)}</code></div>")
    out.append(f"<div style='margin:4px 0'>Nearest training molecules: "
               f"<b>{n_pos} of {K}</b> are mutagenic (neighbour majority: {NAME[nb_pred]})</div>")

    if agree:
        out.append(card(GREEN, "Neighbours agree with the model",
                        "In the pooled held-out test results, predictions in this group were correct "
                        "about 83% of the time. This describes the group, not this molecule."))
    else:
        out.append(card(RED, "Neighbours disagree with the model: treat with caution",
                        "In the pooled held-out test results, predictions in this group were correct "
                        "about 48% of the time. This describes the group, not this molecule."))

    if in_train:
        out.append(card(AMBER, "This molecule is present in the training dataset",
                        f"Its dataset label is <b>{NAME[matched_label]}</b>. It was excluded from the "
                        "neighbour vote above. The model has seen it in training, so the pooled "
                        "accuracy figures do not apply to this prediction."))
    else:
        top_sim = sim[np.argsort(-sim)[0]]
        if top_sim >= 0.999:
            out.append(card(AMBER, "A training molecule has an identical fingerprint",
                            "It may be a stereoisomer or a molecule the fingerprint cannot "
                            "distinguish. Fingerprints here ignore stereochemistry."))
        elif top_sim < 0.3:
            out.append(card(AMBER, "Closest training molecule is dissimilar",
                            "Top similarity is below 0.3, so the neighbours are weak evidence."))

    return "".join(out), nb[["mol_id", "smiles", "label", "similarity"]]


LIMITS = """
- **Research demo, not for safety or regulatory decisions.**
- On AMES, a random forest on fingerprints beat this model (mean test AUROC over 5 seeds: 0.847 vs 0.828).
- This demo uses one trained model (LoRA r8, seed 0), not the 5-seed average.
- The 83% / 48% figures are pooled results over 5 seeds on held-out scaffold test sets, using fingerprint retrieval. They describe groups of predictions, not the confidence of any single prediction.
- Neighbour disagreement is a warning sign, not a guarantee. In my hERG failure analysis, neighbours agreed with the wrong prediction for 18 of the 20 most confident errors.
- Fingerprints ignore stereochemistry, so stereoisomers get identical neighbours.
- Hyperparameters were not tuned and only one pretrained model was tested.
- Data: Ames mutagenicity dataset from Therapeutics Data Commons (TDC), listed as CC BY 4.0, aggregated from Xu et al., J. Chem. Inf. Model. 52(11), 2012. Changes: SMILES canonicalised with RDKit, duplicates removed, and only the training portion of one scaffold split (seed 0) is included.
- Base model: DeepChem/ChemBERTa-77M-MLM is downloaded from the Hugging Face Hub at start-up and is not redistributed here. Its model page declares no licence. The adapter is my own LoRA adapter trained on top of it.
- Code and write-up: github.com/tejasiith/adapt-retrieve-tox
"""

EXAMPLES_TEXT = (
    "These molecules are from my seed-0 scaffold test set (except nitrobenzene, which is in "
    "the training set). I picked them after seeing the results to illustrate each case, "
    "so they are **not a representative sample** and say nothing about overall accuracy.\n\n"
    "1. `C(CC1CO1)C1CO1`: model and neighbours agree, both correct (mutagenic).\n"
    "2. `O=C1CCCCCN1`: model and neighbours agree, both correct (non-mutagenic).\n"
    "3. `O=P1(N(CCCl)CCCl)NCCCO1`: neighbours disagree and are right, the model is wrong.\n"
    "4. `O=C1CCC(=O)N1Br`: neighbours disagree, but the model is right.\n"
    "5. `O=C1CCCCC1=O`: model and neighbours agree and are both wrong, which the check cannot catch.\n"
    "6. `O=[N+]([O-])c1ccccc1`: in the training set, so the warning card appears."
)

EXAMPLES = [
    ("Example 1", "C(CC1CO1)C1CO1"),
    ("Example 2", "O=C1CCCCCN1"),
    ("Example 3", "O=P1(N(CCCl)CCCl)NCCCO1"),
    ("Example 4", "O=C1CCC(=O)N1Br"),
    ("Example 5", "O=C1CCCCC1=O"),
    ("Example 6", "O=[N+]([O-])c1ccccc1"),
]


def set_smiles(value):
    st.session_state["smiles"] = value


# page
st.title("Mutagenicity (AMES): LoRA ChemBERTa with a retrieval check")
st.write("Paste a SMILES string. The model predicts mutagenicity, then the 5 most similar "
         "training molecules are shown, and the page says whether they agree with the model.")
st.markdown("**Research demo, not for safety or regulatory decisions.** "
            "Code and write-up: github.com/tejasiith/adapt-retrieve-tox")

if "smiles" not in st.session_state:
    st.session_state["smiles"] = ""

st.text_input("SMILES", key="smiles", placeholder="e.g. O=[N+]([O-])c1ccccc1")

st.caption("Try an example:")
for row in (EXAMPLES[:3], EXAMPLES[3:]):
    cols = st.columns(3)
    for col, (label, smi) in zip(cols, row):
        col.button(label, on_click=set_smiles, args=(smi,), use_container_width=True)

smiles = st.session_state["smiles"].strip()
if smiles:
    result_html, nb_df = analyse(smiles)
    st.markdown(result_html, unsafe_allow_html=True)
    if nb_df is not None:
        st.caption("5 nearest training molecules")
        st.dataframe(nb_df, hide_index=True, use_container_width=True)
else:
    st.info("Please enter a SMILES string, or press one of the examples.")

with st.expander("What the examples show"):
    st.markdown(EXAMPLES_TEXT)
with st.expander("Research limitations, data and licensing"):
    st.markdown(LIMITS)