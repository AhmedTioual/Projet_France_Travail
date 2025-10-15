import streamlit as st
import pandas as pd
import streamlit_shadcn_ui as ui

# Load the dataset
store_data = pd.read_csv("data/raw_francetravail.csv")

def truncate_text(text, max_len=80):
    """Truncate long strings for table display."""
    if isinstance(text, str) and len(text) > max_len:
        return text[:max_len] + "..."
    return text

def data_page():
    st.markdown("<h3>Data | Quick Overview</h3>", unsafe_allow_html=True)
    st.write("This section provides a quick overview of the dataset used for job offers analysis.")
    st.write("#### First 10 Rows of the Dataset")

    # --- Prepare Data ---
    df = store_data.head(10).copy()

    # Optional: keep only key columns for clarity
    columns_to_show = [
        "intitule", "typeContrat", "lieuTravail.libelle",
        "entreprise.nom", "description"
    ]
    df = df[[col for col in columns_to_show if col in df.columns]]

    # Truncate long text values
    for col in df.columns:
        df[col] = df[col].apply(truncate_text)

    # Convert to string for UI compatibility
    df = df.astype(str)

    # --- Display table ---
    ui.table(data=df, maxHeight=400)