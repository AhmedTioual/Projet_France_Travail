import streamlit as st
import base64
from processing.cleaning import TextCleaner
from streamlit_option_menu import option_menu
from src.accueil import show_accueil
from src.clustering import *
from src.data import data_page
from src.predictions import show_predictions

st.set_page_config( 
        layout="wide",
        page_title="Analyse des Tendances du Marché de l'Emploi Français",
    )

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# img = get_img_as_base64("static/background.png")

css = f"""
    <style>
        MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}

        [data-testid="stAppViewContainer"] {{
            
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            height: 100%;
        }}

        [data-testid="stTextAreaRootElement"] {{
            height: 122px;
        }}

        .st-emotion-cache-zy6yx3 {{
            padding : 0rem 5rem 0rem;
        }}

        [data-testid="stMainBlockContainer"] {{
            padding : 0rem 5rem 0rem;
        }}
    <style>
"""

st.markdown(css, unsafe_allow_html=True)

selected = option_menu(
    None, 
    ["Accueil", "Prédictions", "Analyse de Clustering", "Données"], 
    icons=['house', 'activity', 'bar-chart-steps', 'database'], 
    default_index=0,
    orientation="horizontal",
    styles={
        
        "nav-link-selected": {"background-color": "#1B2256"},
    }
)

if selected == "Accueil":
    show_accueil()
if selected == "Prédictions":
    show_predictions()
if selected == "Analyse de Clustering":
    pass
if selected == "Données":
    data_page()
