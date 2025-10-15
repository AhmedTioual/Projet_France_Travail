import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import plotly.figure_factory as ff
import math
from plotly.subplots import make_subplots
import joblib

@st.cache_data  
def load_clean_data():
    df = pd.read_csv("3_analysis_reporting/data/raw_francetravail.csv")
    return df
    
def job_offer_form():
    
    df = load_clean_data()

    st.subheader("Créer une nouvelle offre d’emploi")

    # --- Form section ---
    with st.form("new_job_form"):

        col1, col2 = st.columns(2)

        with col1:
            intitule = st.text_input("Intitulé du poste", "Développeur Python")
            description = st.text_area("Description", "Vous serez chargé de développer des applications web.")
            natureContrat = st.selectbox("Nature du contrat", list(df['natureContrat'].unique()))
            experienceExige = st.selectbox(
                "Expérience exigée", 
                list(df['experienceExige'].unique()), 
                help="D=Débutant, E=Expérimenté, S=Senior"
            )
            entreprise_nom = st.selectbox("Nom de l’entreprise", list(df['entreprise.nom'].unique()))

        with col2:
            commune_flt = st.selectbox("Commune", list(df['lieuTravail.libelle'].unique()))
            
            # Get commune-related data
            commune_data = df[df['lieuTravail.libelle'] == commune_flt].iloc[0]
            commune = commune_data['lieuTravail.commune']
            latitude = commune_data['lieuTravail.latitude']
            longitude = commune_data['lieuTravail.longitude']
            codePostal = commune_data['lieuTravail.codePostal']

            secteurActiviteLib = st.selectbox("Secteur d’activité (Libellé)", list(df['secteurActiviteLibelle'].unique()))
            secteurActivite = df[df['secteurActiviteLibelle'] == secteurActiviteLib]['secteurActivite'].iloc[0]

            nombrePostes = st.number_input("Nombre de postes", value=1, step=1)

            # Display latitude, longitude, postal code as non-editable fields
            st.number_input("Latitude", value=float(latitude), disabled=True)
            st.number_input("Longitude", value=float(longitude), disabled=True)
            st.number_input("Code postal", value=int(codePostal), disabled=True)

        entrepriseAdaptee = st.checkbox("Entreprise adaptée", value=False)
        employeurHandiEngage = st.checkbox("Employeur Handi Engagé", value=False)

        # --- Submit button ---
        submitted = st.form_submit_button("Créer l’offre")

        if submitted:
            new_job = pd.DataFrame([{
                'intitule': intitule,
                'description': description,
                'natureContrat': natureContrat,
                'experienceExige': experienceExige,
                'entrepriseAdaptee': entrepriseAdaptee,
                'employeurHandiEngage': employeurHandiEngage,
                'entreprise.nom': entreprise_nom,
                'lieuTravail.commune': commune,
                'secteurActivite': secteurActivite,
                'nombrePostes': nombrePostes,
                'lieuTravail.latitude': latitude,
                'lieuTravail.longitude': longitude,
                'lieuTravail.codePostal': codePostal
            }])

            # Load pipeline
            inference_pipeline = joblib.load("data/job_contract_pipeline.pkl")
            pred_numeric = inference_pipeline.predict(new_job)
            pred_label_num = pred_numeric[0]  # get scalar value

            # Map numeric prediction to contract type with explanation
            contract_mapping = {
                0: "CDD (Contrat à Durée Déterminée)",
                1: "CDI (Contrat à Durée Indéterminée)",
                2: "MIS (Contrat de Mise à Disposition)"
            }

            pred_label_text = contract_mapping.get(pred_label_num, "Inconnu")

            # Display in Streamlit
            st.success(f"Type de contrat prédit : {pred_label_text}")

            pass

@st.cache_data  
def load_data():
    df = pd.read_csv("3_analysis_reporting/data/all_classifiers_metrics.csv")
    return df

def plot_classifier_accuracies(df, report_col='test_report', classifier_col='classifier', height=500, width=800):
    accuracies = []

    # Extract accuracy from JSON reports
    for idx, row in df.iterrows():
        try:
            report = json.loads(row[report_col])
            acc = report.get("accuracy", None)
            if acc is not None:
                accuracies.append({'Classifier': row[classifier_col], 'Accuracy': acc})
        except (json.JSONDecodeError, TypeError):
            continue

    # Create DataFrame
    acc_df = pd.DataFrame(accuracies)
    if acc_df.empty:
        st.warning("No valid accuracy data found in test reports.")
        return None

    # Sort by accuracy descending
    acc_df = acc_df.sort_values(by='Accuracy', ascending=False)

    # Create interactive bar chart
    fig = px.bar(
        acc_df,
        x='Classifier',
        y='Accuracy',
        text='Accuracy',
        color='Accuracy',
        color_continuous_scale='Blues',
        height=height,
        width=width
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        title='Test Accuracy per Classifier',
        yaxis_range=[0, 1],
        template='plotly_white',
        margin={"r":20,"t":40,"l":20,"b":20}
    )

    return fig

def plot_classifier_radar(df, height=500, width=900):
    # Define metrics and classifiers
    categories = ["Balanced Accuracy (CV)", "F1 Macro (CV)", 
                  "Balanced Accuracy (Test)", "F1 Macro (Test)"]
    classifiers = list(df["classifier"])
    
    # Extract performance data
    stats = df[["balanced_accuracy_cv", "f1_macro_cv", 
                "balanced_accuracy_test", "f1_macro_test"]].values
    
    # Define color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
              '#9467bd', '#8c564b', 'yellow', 'blue']
    
    # Create radar traces
    traces = []
    for i, classifier in enumerate(classifiers):
        trace = go.Scatterpolar(
            r=stats[i].tolist() + [stats[i][0]],      # close circle
            theta=categories + [categories[0]],
            fill="none",
            name=classifier,
            opacity=1.0,
            line=dict(width=3, color=colors[i % len(colors)])  # handle more classifiers safely
        )
        traces.append(trace)
    
    # Layout configuration
    layout = go.Layout(
        title='Classifier Performance Comparison<br>(CV vs. Test - Balanced Accuracy & F1 Macro)',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0.7, 1],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=11)
            )
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        height=height,
        width=width,
        margin=dict(t=60, b=60, l=50, r=50),
        template="plotly_white"
    )

    fig = go.Figure(data=traces, layout=layout)
    return fig

def plot_classifier_balanced_accuracy_cv(df):

    # Create base x-axis positions
    x = list(df['classifier'])

    # Define bar colors
    color_cv = '#1B2256'
    color_test = '#DBE3FF'

    # Create the interactive bar chart
    fig = go.Figure()

    # Add CV bars
    fig.add_trace(go.Bar(
        x=x,
        y=df['balanced_accuracy_cv'],
        name='CV',
        marker_color=color_cv,
        text=[f"{v:.2f}" for v in df['balanced_accuracy_cv']],
        textposition='outside'
    ))

    # Add Test bars
    fig.add_trace(go.Bar(
        x=x,
        y=df['balanced_accuracy_test'],
        name='Test',
        marker_color=color_test,
        text=[f"{v:.2f}" for v in df['balanced_accuracy_test']],
        textposition='outside'
    ))

    # Update layout for aesthetics
    fig.update_layout(
        title='Balanced Accuracy: CV vs Test',
        xaxis_title='Classifiers',
        yaxis_title='Balanced Accuracy',
        barmode='group',
        yaxis=dict(range=[0.5, 1]),
        plot_bgcolor='white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        height=500,
        width=900
    )

    return fig

def plot_f1_macro_comparison(df):

    # Extract values
    classifiers = list(df['classifier'])
    f1_cv = df['f1_macro_cv']
    f1_test = df['f1_macro_test']

    # Define colors
    color_cv = '#1B2256'
    color_test = '#DBE3FF'

    # Create interactive plot
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=classifiers,
        y=f1_cv,
        name='CV',
        marker_color=color_cv,
        text=[f"{v:.2f}" for v in f1_cv],
        textposition='outside'
    ))

    fig.add_trace(go.Bar(
        x=classifiers,
        y=f1_test,
        name='Test',
        marker_color=color_test,
        text=[f"{v:.2f}" for v in f1_test],
        textposition='outside'
    ))

    fig.update_layout(
        title='F1 Macro Score: CV vs Test',
        xaxis_title='Classifiers',
        yaxis_title='F1 Macro Score',
        barmode='group',
        yaxis=dict(range=[0.5, 1]),
        plot_bgcolor='white',
        height=500,
        width=900,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def plot_jaccard_comparison(df):

    # Extract values
    classifiers = list(df['classifier'])
    f1_cv = df['jaccard_cv']
    f1_test = df['jaccard_test']

    # Define colors
    color_cv = '#1B2256'
    color_test = '#DBE3FF'

    # Create interactive plot
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=classifiers,
        y=f1_cv,
        name='CV',
        marker_color=color_cv,
        text=[f"{v:.2f}" for v in f1_cv],
        textposition='outside'
    ))

    fig.add_trace(go.Bar(
        x=classifiers,
        y=f1_test,
        name='Test',
        marker_color=color_test,
        text=[f"{v:.2f}" for v in f1_test],
        textposition='outside'
    ))

    fig.update_layout(
        title='Jaccard Score: CV vs Test',
        xaxis_title='Classifiers',
        yaxis_title='Jaccard Score',
        barmode='group',
        yaxis=dict(range=[0.5, 1]),
        plot_bgcolor='white',
        height=500,
        width=900,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def plot_ROC_AUC_comparison(df):

    df = df.dropna(subset=['roc_auc_cv', 'roc_auc_test']).reset_index(drop=True)

    # Extract values
    classifiers = list(df['classifier'])
    f1_cv = df['roc_auc_cv']
    f1_test = df['roc_auc_test']

    # Define colors
    color_cv = '#1B2256'
    color_test = '#DBE3FF'

    # Create interactive plot
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=classifiers,
        y=f1_cv,
        name='CV',
        marker_color=color_cv,
        text=[f"{v:.2f}" for v in f1_cv],
        textposition='outside'
    ))

    fig.add_trace(go.Bar(
        x=classifiers,
        y=f1_test,
        name='Test',
        marker_color=color_test,
        text=[f"{v:.2f}" for v in f1_test],
        textposition='outside'
    ))

    fig.update_layout(
        title='ROC-AUC Comparison : CV vs Test',
        xaxis_title='Classifiers',
        yaxis_title='ROC-AUC Score',
        barmode='group',
        yaxis=dict(range=[0.5, 1.03]),
        plot_bgcolor='white',
        height=500,
        width=900,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def plot_confusion_matrices(df):
    """
    Display multiple confusion matrices (one per classifier) as annotated heatmaps (values printed in cells).
    Works correctly inside subplots and Streamlit.
    Expects df['test_confusion_matrix'] to be a JSON-stringified 2D array.
    """
    # limit to first 4 classifiers (you already did)
    df = df.head(4)

    num_classifiers = len(df)
    if num_classifiers == 0:
        raise ValueError("No classifiers in dataframe")

    cols = 2
    rows = math.ceil(num_classifiers / cols)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=df['classifier'].tolist(),
        horizontal_spacing=0.12,
        vertical_spacing=0.12
    )

    labels = ['CDD', 'CDI', 'MIS']

    for idx, (_, row) in enumerate(df.iterrows()):
        # load confusion matrix (list of lists)
        cm_data = json.loads(row["test_confusion_matrix"])
        cm = pd.DataFrame(cm_data, index=labels, columns=labels)

        # Determine subplot position (1-indexed)
        r = (idx // cols) + 1
        c = (idx % cols) + 1

        # Create heatmap trace with embedded text (so values are always visible)
        trace = go.Heatmap(
            z=cm.values,
            x=cm.columns.tolist(),
            y=cm.index.tolist(),
            text=cm.values.astype(int).astype(str),   # text shown in cells
            texttemplate="%{text}",                    # template to render text
            hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>",
            colorscale='Blues',
            showscale=True,
            zmid=None  # optional: remove if you want per-plot normalization
        )

        fig.add_trace(trace, row=r, col=c)

        # Ensure y-axis goes top-to-bottom like a matrix
        fig.update_yaxes(autorange='reversed', row=r, col=c)

        # Optional: tweak tick fonts to be readable
        fig.update_xaxes(tickmode='array', tickvals=cm.columns.tolist(), row=r, col=c)
        fig.update_yaxes(tickmode='array', tickvals=cm.index.tolist(), row=r, col=c)

    # Layout
    fig.update_layout(
        title_text="Confusion Matrices by Classifier (TEST)",
        height=420 * rows,
        width=920,
        showlegend=False,
        margin=dict(t=100, b=50, l=50, r=25),
        plot_bgcolor='white'
    )

    return fig

def plot_classification_reports(df):
    """
    Display multiple classification reports (precision, recall, f1-score)
    as annotated heatmaps for each classifier.
    """
    df = df.head(4)

    # --- Safely parse test_report JSON strings ---
    def safe_json_load(x):
        try:
            return json.loads(x) if isinstance(x, str) else None
        except Exception:
            return None

    df['report_dict'] = df['test_report'].apply(safe_json_load)

    # --- Extract metrics ---
    records = []
    for idx, row in df.iterrows():
        classifier = row['classifier']
        report = row['report_dict']
        if isinstance(report, dict):
            for cls, metrics in report.items():
                if cls in ['0', '1', '2'] and isinstance(metrics, dict):
                    for metric in ['precision', 'recall', 'f1-score']:
                        if metric in metrics:
                            records.append({
                                'Classifier': classifier,
                                'Class': cls,
                                'Metric': metric,
                                'Value': metrics[metric]
                            })

    # --- Build DataFrame ---
    report_df = pd.DataFrame(records)
    if report_df.empty:
        st.warning("No valid classification report data found.")
        return

    class_mapping = {'0': 'CDD', '1': 'CDI', '2': 'MIS'}
    report_df['Class'] = report_df['Class'].map(class_mapping)

    classifiers = df['classifier'].tolist()
    cols = 2
    rows = math.ceil(len(classifiers) / cols)

    # --- Create subplot layout ---
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=classifiers,
        horizontal_spacing=0.12,
        vertical_spacing=0.12
    )

    # --- Add one heatmap per classifier ---
    for i, cl in enumerate(classifiers):
        filtered_df = report_df[report_df['Classifier'] == cl]
        pivot_df = filtered_df.pivot(index='Class', columns='Metric', values='Value')

        if pivot_df.empty:
            continue

        r = (i // cols) + 1
        c = (i % cols) + 1

        # Create annotated heatmap using go.Heatmap
        trace = go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns.tolist(),
            y=pivot_df.index.tolist(),
            text=[[f"{v:.2f}" for v in row] for row in pivot_df.values],  # show values
            texttemplate="%{text}",
            textfont=dict(size=14),
            hovertemplate="Class: %{y}<br>Metric: %{x}<br>Value: %{z:.2f}<extra></extra>",
            colorscale='Blues',
            zmin=0,
            zmax=1,  # since metrics are between 0 and 1
            showscale=True
        )

        fig.add_trace(trace, row=r, col=c)
        fig.update_yaxes(autorange='reversed', row=r, col=c)

    # --- Layout styling ---
    fig.update_layout(
        title="Classification Reports by Classifier",
        height=400 * rows,
        width=900,
        showlegend=False,
        plot_bgcolor='white',
        margin=dict(t=100)
    )

    return fig

def show_predictions():
    
    df = load_data()

    st.markdown("<h3>🤖 | Prédiction du Type de Contrat à partir d'une Offre d’Emploi</h3>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    job_offer_form()
        
    st.markdown("<hr>", unsafe_allow_html=True)

    st.subheader("Model Performance Comparison")

    col1,col2,col3,col4,col5 = st.tabs(["Overall Accuracy","Balanced Accuracy: CV vs Test", "F1 Macro","Jaccard", "ROC-AUC"])

    with col1:
        st.plotly_chart( plot_classifier_accuracies(df), use_container_width=True)

    with col2:
        st.plotly_chart(plot_classifier_balanced_accuracy_cv(df), use_container_width=True)

    with col3:
        st.plotly_chart(plot_f1_macro_comparison(df), use_container_width=True)

    with col4:
        st.plotly_chart(plot_jaccard_comparison(df), use_container_width=True)
        
    with col5:
        st.plotly_chart(plot_ROC_AUC_comparison(df), use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.subheader("Model Matrices Comparison")

    col1,col2 = st.tabs(["Confusion Matrices by Classifier","Classification Reports per Classifier"])

    with col1:
        st.plotly_chart(plot_confusion_matrices(df), use_container_width=True)  

    with col2:
        st.plotly_chart(plot_classification_reports(df), use_container_width=True) 
