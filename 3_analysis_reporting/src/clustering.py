import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import plotly.figure_factory as ff
import math
from plotly.subplots import make_subplots


def show_feature_sample(df, cols_cat, cols_num):
    
    # --- Display categorical and numeric columns as tags ---
    with st.expander("Détails des variables utilisées"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Variables catégorielles :**")
            for c in cols_cat:
                st.markdown(f"- `{c}`")
        with col2:
            st.markdown("**Variables numériques :**")
            for c in cols_num:
                st.markdown(f"- `{c}`")

    # --- Display first 5 rows of selected features ---
    features = cols_cat + cols_num
    if not features:
        st.warning("Aucune colonne sélectionnée.")
        return

    st.write("##### Extrait du jeu de données (5 premières lignes)")
    st.dataframe(df[features].head(5), use_container_width=True)

@st.cache_data  
def load_data():
    df = pd.read_csv("data/francetravail_salaire_normalise.csv")
    return df

def salary_overview(df):

    # --- Prepare data ---
    salary_data = df['salaire_annuel_estime'].dropna()

    if salary_data.empty:
        st.warning("Aucune donnée de salaire exploitable trouvée.")
        return

    # --- Interactive histogram ---
    fig = px.histogram(
        salary_data,
        nbins=40,
        #title="Distribution du salaire annuel estimé (€)",
        labels={'value': 'Salaire annuel (€)'},
        color_discrete_sequence=['#1B2256']
    )

    fig.update_layout(
        xaxis_title="Salaire annuel (€)",
        yaxis_title="Nombre d'offres",
        bargap=0.05,
        plot_bgcolor='white',
        title_font=dict(size=16),
    )

    fig.update_xaxes(showgrid=True, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridcolor='lightgray')

    return fig

def show_clustering():
    
    df = load_data()

    st.markdown("<h3>💼 | Clustering des Offres d’Emploii</h3>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.subheader("Distribution du salaire annuel estimé (€)")

    st.plotly_chart(salary_overview(df), use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.subheader("Aperçu des variables utilisées pour le clustering")

    cols_cat = ['typeContrat', 'natureContrat', 'secteurActiviteLibelle', 'experienceExige', 'romeLibelle']
    cols_num = ['nombrePostes']

    show_feature_sample(df,cols_cat,cols_num)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.subheader("Évaluation du nombre optimal de clusters")

    # --- Layout: two columns ---
    col1, col2 = st.columns(2)

    # --- Display elbow plot ---
    with col1:
        st.image("static/elbow.png", caption="Méthode du coude", use_container_width=True)

    # --- Display silhouette plot ---
    with col2:
        st.image("static/selhoute.png", caption="Score de silhouette", use_container_width=True)

    st.subheader("Profils des clusters identifiés (K-Means, k=3)")

    # --- Display numeric results ---
    prof_num = pd.read_csv("data/clustering_results.csv")
    st.dataframe(prof_num, use_container_width=True)

    # --- Cluster 0 ---
    st.markdown("""
    ### Cluster 0 — CDI expérimentés en informatique
    * **Type de contrat** : 82 % CDI  
    * **Nature du contrat** : 96 % “Contrat de travail”  
    * **Expérience** : exigée pour 88 % des offres  
    * **Métiers dominants** : développeur informatique, développeur web, chef de projet MOA  
    * **Secteurs représentés** : conseil, SSII, agences d’intérim IT  
    * **Profil résumé** : postes stables, qualifiés et bien rémunérés pour professionnels confirmés du numérique.
    """)

    # --- Cluster 1 ---
    st.markdown("""
    ### Cluster 1 — CDD et alternance : profils en entrée de carrière
    * **Type de contrat** : 70 % CDI, 20 % CDD, 12 % apprentissage  
    * **Expérience** : 98 % “débutant accepté”  
    * **Métiers dominants** : développeur web, ingénieur d’études, scrum master  
    * **Secteurs** : majoritairement IT / développement web  
    * **Profil résumé** : emplois de transition ou de formation, salaires plus faibles, population jeune diplômée.
    """)

    # --- Cluster 2 ---
    st.markdown("""
    ### Cluster 2 — Intérim industriel et métiers techniques rares
    * **Type de contrat** : 75 % missions d’intérim (MIS)  
    * **Secteurs** : industrie et travail temporaire  
    * **Métiers** : câbleur d’armoires électriques, administrateur de serveurs  
    * **Profil résumé** : groupe marginal et atypique, peu représentatif du cœur de l’échantillon (probable cluster d’outliers).
    """)
