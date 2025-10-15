import streamlit as st
import pandas as pd
import streamlit_shadcn_ui as ui
import matplotlib.pyplot as plt
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from collections import Counter
from wordcloud import STOPWORDS
import numpy as np
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import streamlit as st

@st.cache_data  
def load_data():
    df = pd.read_csv("3_analysis_reporting/data/clean_francetravail.csv")
    return df

@st.cache_data
def get_info(df):

    num_rows = df.shape[0]

    num_comp = df['entreprise.nom'].nunique()

    tota_posts = int(df['nombrePostes'].sum())

    return num_rows,num_comp,tota_posts

def plot_offers_per_month(df):
    # Ensure dateCreation is datetime
    df['dateCreation'] = pd.to_datetime(df['dateCreation'], errors='coerce')

    # Drop rows with invalid dates
    df_time = df.dropna(subset=['dateCreation'])

    # Extract Year-Month
    df_time['year_month'] = df_time['dateCreation'].dt.to_period('M')

    # Group by month
    offers_per_month = df_time.groupby('year_month').size().reset_index(name='total_offers')

    # Convert year_month back to string for plotting
    offers_per_month['year_month'] = offers_per_month['year_month'].astype(str)

    # Create interactive Plotly figure
    fig = px.line(
        offers_per_month,
        x='year_month',
        y='total_offers',
        markers=True,
        height=500,
        width=600
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Total Offers",
        xaxis_tickangle=0,
        template="plotly_white"
    )
    return fig

def plot_offers_map(df, lat_col='lieuTravail.latitude', lon_col='lieuTravail.longitude', location_col='lieuTravail.commune'):
    
    # Ensure numeric coordinates
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
    
    # Drop rows without coordinates
    df_map = df.dropna(subset=[lat_col, lon_col])
    
    # Group by location
    location_counts = df_map.groupby(
        [location_col, lat_col, lon_col]
    ).size().reset_index(name='total_offers')
    
    # Compute center of map
    lat_center = location_counts[lat_col].mean()
    lon_center = location_counts[lon_col].mean()
    
    # Create Folium map
    m = folium.Map(location=[lat_center, lon_center], zoom_start=6)
    
    # Add markers with clustering
    marker_cluster = MarkerCluster().add_to(m)
    for idx, row in location_counts.iterrows():
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=5 + row['total_offers']*0.5,  # size proportional to total offers
            popup=f"{row[location_col]}: {row['total_offers']} offres",
            color='blue',
            fill=True,
            fill_opacity=0.6
        ).add_to(marker_cluster)
    
    return m

def plot_top_communes(df, commune_col='lieuTravail.libelle', top_n=10, height=500, width=600):
    # Drop rows without commune
    df_commune = df.dropna(subset=[commune_col])
    
    # Group by commune and count offers
    top_communes = df_commune.groupby(commune_col).size().reset_index(name='total_offers')
    
    # Sort descending and take top N
    top_communes = top_communes.sort_values(by='total_offers', ascending=False).head(top_n)
    
    # Create horizontal bar chart
    fig = px.bar(
        top_communes,
        x='total_offers',
        y=commune_col,
        orientation='h',
        text='total_offers',
        height=height,
        width=width
    )
    fig.update_traces(marker_color='#1B2256', textposition='outside')
    fig.update_layout(
        xaxis_title="Number of Offers",
        yaxis_title="Commune",
        yaxis=dict(autorange="reversed"),  # to have the largest on top
        template="plotly_white",
        margin={"r":20,"t":40,"l":20,"b":20}
    )
    return fig


def plot_contract_distribution(df, contract_col='typeContrat', top_n=3, height=400, width=400):
    # Drop rows without contract type
    df_contract = df.dropna(subset=[contract_col])
    
    # Count number of offers per contract type
    contract_counts = df_contract[contract_col].value_counts().reset_index()
    contract_counts.columns = [contract_col, 'total_offers']
    
    # Keep top N contract types
    contract_counts = contract_counts.head(top_n)
    
    # Create interactive pie chart
    fig = px.pie(
        contract_counts,
        values='total_offers',
        names=contract_col,
        title=None,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        height=height,
        width=width
    )
    
    # Pie chart labels inside
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    # Legend at top, centered, horizontal
    fig.update_layout(
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.05,  # slightly above the plot
            xanchor='center',
            x=0.5
        ),
        margin=dict(t=80, b=50, l=50, r=50)  # adjust top margin for title + legend
    )
    
    return fig

def plot_wordcloud(df, text_col='intitule', max_words=100, width=800, height=500):
    # Combine all text from the column
    text = ' '.join(df[text_col].dropna().astype(str))
    
    # Optional stopwords
    stopwords = set(STOPWORDS)
    
    # Create word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color='white',
        stopwords=stopwords,
        max_words=max_words
    ).generate(text)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    # Display in Streamlit
    st.pyplot(fig)

def show_accueil():
    
    df = load_data()

    st.markdown("<h3>💼 | Analyse des Tendances du Marché de l'Emploi Français </h3>", unsafe_allow_html=True)

    cols = st.columns(3)
    
    num_rows,num_comp,tota_posts = get_info(df)

    with cols[0]:
        ui.metric_card(title="Total d'offres", content=num_rows)
    with cols[1]:
        ui.metric_card(title="Total d'entreprises", content=num_comp)
    with cols[2]:
        ui.metric_card(title="Total de postes", content=tota_posts)
    
    st.markdown("<hr>", unsafe_allow_html=True)

    # Create two columns
    col1, col2 = st.columns(2)

    # Call the function inside the first column
    with col1:
        
        st.subheader("Carte des offres d'emploi")
        st_data = st_folium(plot_offers_map(df), width=600, height=500)
    with col2:
        st.subheader("Offres d'emploi au fil du temps")
        st.plotly_chart(plot_offers_per_month(df), use_container_width=False)
        
    st.markdown("<hr>", unsafe_allow_html=True)

    # Create two columns
    col1, col2 = st.columns(2)

    # Call the function inside the first column
    with col1:
        st.subheader("Offres d'emploi par type de contrat")
        st.plotly_chart(plot_contract_distribution(df), use_container_width=False)
        
    with col2:
        st.subheader("Nuage de mots sur la description de poste")
        plot_wordcloud(df, max_words=150)

    

