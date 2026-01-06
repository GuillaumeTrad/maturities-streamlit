import os
import tempfile
from io import BytesIO

import pandas as pd
import streamlit as st

from parsing_engine import parse_pdf
import parsing_engine


st.set_page_config(page_title="Maturities Extractor", layout="wide")

st.title("PDF Maturities Extractor")
st.caption("Upload un PDF d’échéances, extraction automatique des trades + export CSV/Excel.")

# -----------------------------
# Helpers
# -----------------------------
def parse_uploaded_pdf(uploaded_file) -> pd.DataFrame:
    """
    Sauvegarde le PDF uploadé dans un fichier temporaire,
    puis appelle parse_pdf(tmp_path) du parsing_engine.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        return parse_pdf(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = BytesIO()
    # marche si openpyxl OU xlsxwriter est installé
    try:
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="echeances")
    except Exception:
        with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="echeances")

    bio.seek(0)
    return bio.read()


def apply_search_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sidebar: recherche + tri uniquement (comme demandé)
    """
    if df.empty:
        return df

    st.sidebar.header("Recherche")
    q = st.sidebar.text_input("Recherche (borrower / lender)", "")
    if q.strip():
        q_up = q.strip().upper()
        mask = pd.Series(False, index=df.index)

        for c in ["borrower", "lender"]:
            if c in df.columns:
                mask = mask | df[c].fillna("").astype(str).str.upper().str.contains(q_up, na=False)

        df = df[mask]

    st.sidebar.header("Filtres")
    only_pp = st.sidebar.checkbox("Afficher uniquement les PP", value=False)

    if only_pp and "deal_type" in df.columns:
        df = df[df["deal_type"] == "PP/Obligataire"]


    st.sidebar.header("Tri")
    sort_candidates = [c for c in ["maturity", "date_valeur", "montant", "taux", "borrower", "lender"] if c in df.columns]
    sort_col = st.sidebar.selectbox("Trier par", sort_candidates, index=0 if sort_candidates else 0)
    asc = st.sidebar.checkbox("Ordre croissant", True)

    if sort_col:
        if sort_col in ["maturity", "date_valeur"]:
            # convertit en vraie date pour trier correctement (supporte . et /)
            s = df[sort_col].astype(str).str.replace("/", ".", regex=False)

            df["_sort_dt"] = pd.to_datetime(
                s,
                format="%d.%m.%Y",
                dayfirst=True,
                errors="coerce"
            )

            df = df.sort_values(by="_sort_dt", ascending=asc, kind="mergesort", na_position="last")
            df = df.drop(columns=["_sort_dt"])
        else:
            df = df.sort_values(by=sort_col, ascending=asc, kind="mergesort")


    return df


def format_chf(x):
    """
    Format suisse: milliers avec ' et décimales avec virgule.
    Ex:
      10000000 -> 10'000'000
      1.65 -> 1,65 (si tu veux garder le point, dis-moi)
    """
    if pd.isna(x):
        return ""
    try:
        s = f"{float(x):,.2f}"
        # 1,234,567.89 -> 1'234'567,89
        s = s.replace(",", "X").replace(".", ",").replace("X", "'")
        # enlever ,00 si inutile
        if s.endswith(",00"):
            s = s[:-3]
        return s
    except Exception:
        return str(x)


# -----------------------------
# UI
# -----------------------------
uploaded = st.file_uploader("Glisse ton PDF ici", type=["pdf"])
auto_parse = st.checkbox("Parser automatiquement après upload", value=True)

if "df_raw" not in st.session_state:
    st.session_state.df_raw = None

if uploaded is not None:
    if auto_parse:
        with st.spinner("Parsing en cours..."):
            st.session_state.df_raw = parse_uploaded_pdf(uploaded)
    else:
        if st.button("Parser le PDF", type="primary"):
            with st.spinner("Parsing en cours..."):
                st.session_state.df_raw = parse_uploaded_pdf(uploaded)

df = st.session_state.df_raw


if df is None:
    st.info("Upload un PDF pour commencer.")
else:
    # On applique recherche + tri
    df_view = apply_search_and_sort(df)

    # -----------------------------
    # Affichage Streamlit (clean)
    # -----------------------------
    df_display = df_view.copy()

    df_display = df_display.rename(columns={
    "date_valeur": "date valeur",
    "deal_type": "produit",
    })


    # Colonnes à masquer dans Streamlit (comme demandé)
    cols_to_drop = ["client_final", "parse_status", "missing_fields", "page_number"]
    df_display = df_display.drop(columns=[c for c in cols_to_drop if c in df_display.columns])

    # Format suisse pour les montants et taux
    if "montant" in df_display.columns:
        df_display["montant"] = df_display["montant"].apply(format_chf)

    if "taux" in df_display.columns:
        df_display["taux"] = df_display["taux"].apply(format_chf)

    st.subheader("Tableau")
    st.dataframe(df_display, use_container_width=True, height=650)

    # -----------------------------
    # Exports
    # -----------------------------
    st.subheader("Exports")
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "Télécharger CSV",
            data=df_to_csv_bytes(df_view),
            file_name="echeances_extraites.csv",
            mime="text/csv",
        )

    with col2:
        try:
            xlsx_bytes = df_to_excel_bytes(df_view)
            st.download_button(
                "Télécharger Excel (.xlsx)",
                data=xlsx_bytes,
                file_name="echeances_extraites.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception:
            st.warning("Export Excel indisponible (installe openpyxl ou xlsxwriter).")
            st.code("python -m pip install openpyxl", language="powershell")

