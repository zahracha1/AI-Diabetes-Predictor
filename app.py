import streamlit as st
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime

# CONFIG PAGE
st.set_page_config(
    page_title="Diabetes AI Predictor - Diagnostic Intelligence",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)
theme = st.get_option("theme.base")  # retourne "dark" ou "light"
label_color = "#ffffff" if theme == "dark" else "#333333"
# STYLE CSS AMÉLIORÉ
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #F9F6EE 0%, #FFF9F0 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 18px;
        font-weight: 600;
        border-radius: 12px;
        padding: 12px 32px;
        border: none;
        width: 100%;
        transition: all 0.3s;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Input styling */
    .stNumberInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 8px 12px;
        transition: all 0.3s;
        
    }
    
    .stNumberInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
    }
    
    /* Label styling */
    .stNumberInput label {{
        font-weight: 600;
        color: {label_color} !important;
        margin-bottom: 0.5rem;
        display: inline-block;
    }}

    /* Result cards */
    .result-card-positive {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.3);
    }
    
    .result-card-negative {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 10px 30px rgba(79, 172, 254, 0.3);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        border-left: 4px solid #667eea;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    /* Divider */
    .custom-divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
        margin: 2rem 0;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    /* Sidebar styling */
    .sidebar-content {
        background: white;
        border-radius: 15px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# HEADER PERSONNALISÉ
st.markdown("""
<div class="main-header">
    <h1>🩺 Diabetes AI Predictor</h1>
    <p>Diagnostic intelligent basé sur l'intelligence artificielle</p>
</div>
""", unsafe_allow_html=True)

# Charger modèle
@st.cache_resource
def load_models():
    try:
        model = joblib.load("model_xgb.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {str(e)}")
        st.stop()

model, scaler = load_models()

# SIDEBAR INFORMATIONS
with st.sidebar:
    st.markdown("### ℹ️ À propos")
    st.markdown("""
    Cet outil utilise un modèle entraîné sur un jeu de données de diabète.
    
    ---
    ### 📋 Instructions
    1. Remplir tous les champs
    2. Cliquez sur "Prédire"
    3. Lire le résultat
    
    ---
    ### 🔬 Facteurs de risque
    - Glycémie élevée
    - IMC > 25
    - Antécédents familiaux
    - Âge > 45 ans
    """)
    
    st.markdown("---")
    st.caption(f"Charana Zahra | {datetime.now().year}")

# TITRE DE LA SECTION
st.markdown("### Informations cliniques du patient")
st.markdown("Veuillez renseigner les paramètres ci-dessous :")

# INPUTS AVEC DESIGN AMÉLIORÉ
with st.container():
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        with st.container():
            
            Pregnancies = st.number_input("📊 Nombre de grossesses")
            Glucose = st.number_input("🩸 Glycémie")
            BloodPressure = st.number_input("❤️ Pression artérielle")
            SkinThickness = st.number_input("📏 Épaisseur de la peau") 
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            Insulin = st.number_input("💉 Insuline")
            DiabetesPedigreeFunction = st.number_input("🧬 Score héréditaire")
            Age = st.number_input("🎂 Âge")
            BMI = st.number_input("👤 Indice de masse corporelle (BMI)")
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# BOUTON DE PRÉDICTION
col_button1, col_button2, col_button3 = st.columns([1, 2, 1])
with col_button2:
    predict_button = st.button("🔍 ANALYSER LE RISQUE", use_container_width=True)

if predict_button:
    # Feature engineering
    BMI_Age = BMI * Age
    Glucose_Insulin = Glucose * (Insulin if Insulin != 0 else 100)
    Age_log = np.log(Age + 1)
    
    data = np.array([[Pregnancies, Glucose, BloodPressure,
                      SkinThickness, Insulin, BMI,
                      DiabetesPedigreeFunction, Age_log,
                      BMI_Age, Glucose_Insulin]])
    
    # Scaling et prédiction
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    proba = model.predict_proba(data_scaled)[0][1]
    
    # AFFICHAGE DES RÉSULTATS
    st.markdown("### Résultat du diagnostic")
    
    # Affichage du résultat principal
    if prediction == 1:
        st.markdown(f"""
        <div class="result-card-positive">
            <h2 style="margin:0;">⚠️ Risque Élevé De Diabète</h2>
            <p style="font-size: 3rem; margin: 1rem 0;">{proba:.1%}</p>
            <p style="margin:0;">Probabilité de diabète</p>
            <hr style="background:rgba(255,255,255,0.3); margin: 1rem 0;">
            <p style="margin:0; font-size:0.9rem;">Consultation médicale recommandée</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card-negative">
            <h2 style="margin:0;"> Risque Faible De Diabète</h2>
            <p style="font-size: 3rem; margin: 1rem 0;">{proba:.1%}</p>
            <p style="margin:0;">Probabilité de diabète</p>
            <hr style="background:rgba(255,255,255,0.3); margin: 1rem 0;">
            <p style="margin:0; font-size:0.9rem;">Maintenez un mode de vie sain</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Barre de progression
    st.markdown("#### Niveau de risque")
    st.progress(int(proba * 100))
    
    # Indicateurs supplémentaires
    st.markdown("#### Analyse détaillée")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{Glucose}</div>
            <div class="metric-label">Glycémie (mg/dL)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m2:
        bmi_status = "Élevé" if BMI > 25 else "Normal"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{BMI:.1f}</div>
            <div class="metric-label">BMI {bmi_status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{Age}</div>
            <div class="metric-label">Âge (ans)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{DiabetesPedigreeFunction:.2f}</div>
            <div class="metric-label">Score héréditaire</div>
        </div>
        """, unsafe_allow_html=True)

    # Recommandations personnalisées
    st.markdown("#### 💡 Recommandations")
    
    if prediction == 1:
        st.warning("""
        **🔔 Consultation médicale recommandée :**
        - Prenez rendez-vous avec votre médecin traitant
        - Envisagez un test HbA1c pour confirmation
        - Adoptez une alimentation équilibrée
        - Pratiquez une activité physique régulière
        """)
    else:
        st.info("""
        ** Prévention et bien-être :**
        - Maintenez une alimentation saine et équilibrée
        - Pratiquez 30 minutes d'exercice par jour
        - Contrôlez régulièrement votre glycémie
        - Maintenez un poids santé
        """)
    

