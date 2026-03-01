# streamlit_app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import time
import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# =============================
# 1. CONFIGURATION DE LA PAGE
# =============================
st.set_page_config(
    page_title="EfficientNet Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================
# 2. INITIALISATION DE L'ÉTAT DE SESSION
# =============================
if 'language_selected' not in st.session_state:
    st.session_state.language_selected = False
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'
if 'page' not in st.session_state:
    st.session_state.page = 'language'

# =============================
# 3. DICTIONNAIRES DE TRADUCTION
# =============================
translations = {
    'en': {
        'select_language': "🌍 Select your language",
        'french': "🇫🇷 Français",
        'english': "🇬🇧 English", 
        'spanish': "🇪🇸 Español",
        'continue': "Continue →",
        'title': "🌿 EfficientNet-B0 Plant Disease Classifier",
        'subtitle': "Upload an image to identify plant diseases (17 classes)",
        'welcome': "Welcome to the plant disease classification project. This application uses a fine-tuned EfficientNet-B0 model to recognize 17 different diseases on 4 types of plants.",
        'instruction': "Upload an image below to start analysis",
        'categories': "The 17 recognized categories:",
        'start_prediction': "Go to Prediction →",
        'back_home': "← Back to home",
        'upload': "📤 Upload your plant image here",
        'threshold': "Confidence threshold:",
        'threshold_help': "Images below this confidence will be rejected",
        'analyze': "🔍 Analyze Image",
        'results': "📊 Analysis Results",
        'confidence': "Confidence Level",
        'prediction': "Prediction",
        'top3': "Top 3 Predictions",
        'in_base': "IN BASE",
        'not_in_base': "NOT IN BASE",
        'category': "Category",
        'probability': "Probability",
        'processing': "Processing image...",
        'success_msg': "Image successfully classified",
        'reject_msg': "Image rejected - not in database",
        'error_msg': "Error processing image",
        'file_error': "File not found or invalid",
        'threshold_info': "Adjust threshold (higher = stricter)",
        'all_categories': "All Categories",
        'no_image': "Please upload an image to start analysis",
        'examples': "Expected image examples:",
        'example1': "• Tomato leaves with spots, mold, or discoloration",
        'example2': "• Rice leaves showing disease symptoms",
        'example3': "• Strawberry leaves with scorch or spots",
        'example4': "• Squash leaves with powdery mildew"
    },
    'fr': {
        'select_language': "🌍 Choisissez votre langue",
        'french': "🇫🇷 Français",
        'english': "🇬🇧 English", 
        'spanish': "🇪🇸 Español",
        'continue': "Continuer →",
        'title': "🌿 Classificateur EfficientNet-B0 - Maladies des Plantes",
        'subtitle': "Téléchargez une image pour identifier les maladies (17 classes)",
        'welcome': "Bienvenue au projet de classification des maladies de plantes. Cette application utilise un modèle EfficientNet-B0 fine-tuné pour reconnaître 17 maladies différentes sur 4 types de plantes.",
        'instruction': "Téléchargez une image ci-dessous pour commencer l'analyse",
        'categories': "Les 17 catégories reconnues :",
        'start_prediction': "Aller à la Prédiction →",
        'back_home': "← Retour à l'accueil",
        'upload': "📤 Téléchargez votre image de plante ici",
        'threshold': "Seuil de confiance :",
        'threshold_help': "Les images sous ce seuil seront rejetées",
        'analyze': "🔍 Analyser l'image",
        'results': "📊 Résultats d'analyse",
        'confidence': "Niveau de confiance",
        'prediction': "Prédiction",
        'top3': "Top 3 Prédictions",
        'in_base': "DANS LA BASE",
        'not_in_base': "HORS BASE",
        'category': "Catégorie",
        'probability': "Probabilité",
        'processing': "Traitement en cours...",
        'success_msg': "Image classifiée avec succès",
        'reject_msg': "Image rejetée - pas dans la base",
        'error_msg': "Erreur lors du traitement",
        'file_error': "Fichier introuvable ou invalide",
        'threshold_info': "Ajustez le seuil (plus haut = plus strict)",
        'all_categories': "Toutes les Catégories",
        'no_image': "Veuillez télécharger une image pour commencer l'analyse",
        'examples': "Exemples d'images attendues :",
        'example1': "• Feuilles de tomate avec taches, moisissures ou décolorations",
        'example2': "• Feuilles de riz montrant des symptômes de maladie",
        'example3': "• Feuilles de fraise avec brûlures ou taches",
        'example4': "• Feuilles de courge avec oïdium"
    },
    'es': {
        'select_language': "🌍 Seleccione su idioma",
        'french': "🇫🇷 Français",
        'english': "🇬🇧 English",
        'spanish': "🇪🇸 Español",
        'continue': "Continuar →",
        'title': "🌿 Clasificador EfficientNet-B0 - Enfermedades de Plantas",
        'subtitle': "Sube una imagen para identificar enfermedades (17 clases)",
        'welcome': "Bienvenido al proyecto de clasificación de enfermedades de plantas. Esta aplicación utiliza un modelo EfficientNet-B0 fine-tuned para reconocer 17 enfermedades diferentes en 4 tipos de plantas.",
        'instruction': "Suba una imagen abajo para comenzar el análisis",
        'categories': "Las 17 categorías reconocidas:",
        'start_prediction': "Ir a Predicción →",
        'back_home': "← Volver al inicio",
        'upload': "📤 Suba su imagen de planta aquí",
        'threshold': "Umbral de confianza:",
        'threshold_help': "Las imágenes bajo este umbral serán rechazadas",
        'analyze': "🔍 Analizar Imagen",
        'results': "📊 Resultados del Análisis",
        'confidence': "Nivel de Confianza",
        'prediction': "Predicción",
        'top3': "Top 3 Predicciones",
        'in_base': "EN LA BASE",
        'not_in_base': "FUERA DE BASE",
        'category': "Categoría",
        'probability': "Probabilidad",
        'processing': "Procesando imagen...",
        'success_msg': "Imagen clasificada con éxito",
        'reject_msg': "Imagen rechazada - no en la base",
        'error_msg': "Error al procesar la imagen",
        'file_error': "Archivo no encontrado o inválido",
        'threshold_info': "Ajuste el umbral (más alto = más estricto)",
        'all_categories': "Todas las Categorías",
        'no_image': "Por favor suba una imagen para comenzar el análisis",
        'examples': "Ejemplos de imágenes esperadas:",
        'example1': "• Hojas de tomate con manchas, moho o decoloración",
        'example2': "• Hojas de arroz con síntomas de enfermedad",
        'example3': "• Hojas de fresa con quemaduras o manchas",
        'example4': "• Hojas de calabaza con mildiu polvoriento"
    }
}

# =============================
# 4. CLASSES RÉELLES (17 catégories)
# =============================
CLASSES = [
    "Rice_brownspot",
    "Rice_healthy",
    "Rice_hispa",
    "Rice_leafblast",
    "squash_powdery_mildew",
    "strawberry_healthy",
    "strawberry_leaf_scorch",
    "tomato_bacterial_spot",
    "tomato_early_blight",
    "tomato_healthy",
    "tomato_late_blight",
    "tomato_leaf_mold",
    "tomato_septoria_leaf_spot",
    "tomato_spider_mites_two-spotted_spider_mite",
    "tomato_target_spot",
    "tomato_tomato_mosaic_virus",
    "tomato_tomato_yellow_leaf_curl_virus"
]

# =============================
# 5. FONCTION DE FORMATAGE DES NOMS
# =============================
def format_class_name(name):
    if "_" in name:
        parts = name.split("_")
        if len(parts) >= 2:
            if parts[0] in ["Rice", "squash", "strawberry", "tomato"]:
                plante = parts[0].capitalize()
                maladie = " ".join(parts[1:]).replace("_", " ").title()
                return f"{plante} - {maladie}"
        return name.replace("_", " ").title()
    return name

# =============================
# 6. CHARGEMENT DU MODÈLE (CACHÉ)
# =============================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Créer l'architecture du modèle
    model = models.efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, len(CLASSES))
    )
    
    # Charger les poids
    model_path = 'best_efficientnet_acc_final.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    else:
        return None, None

# =============================
# 7. TRANSFORMATIONS
# =============================
def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# =============================
# 8. CSS PERSONNALISÉ
# =============================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #a8d5ba 0%, #8cc084 100%);
    }
    
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 12px 24px;
        border: none;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .success-box {
        background-color: #00FF00;
        color: #000000;
        padding: 1.5rem;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        border: 2px solid #000000;
        margin: 1rem 0;
    }
    
    .reject-box {
        background-color: #000000;
        color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        border: 2px solid #FF0000;
        margin: 1rem 0;
    }
    
    .category-card {
        background-color: rgba(255,255,255,0.7);
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
        border-left: 4px solid #4CAF50;
    }
    
    h1, h2, h3 {
        color: #000000 !important;
        font-weight: bold !important;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .spinning-img {
        animation: spin 15s linear infinite;
        border-radius: 50%;
        border: 3px solid #4CAF50;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin: 10px;
        transition: transform 0.3s;
    }
    
    .spinning-img:hover {
        animation: spin 3s linear infinite;
    }
    
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 5px;
    }
    
    .stSlider > div {
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# 9. PAGE 1 : SÉLECTION OBLIGATOIRE DE LA LANGUE
# =============================
if not st.session_state.language_selected:
    # Images décoratives
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
        <div style="display: flex; justify-content: center; gap: 20px; margin: 50px 0;">
            <img src="https://via.placeholder.com/100/4CAF50/ffffff?text=🌱" class="spinning-img" width="100">
            <img src="https://via.placeholder.com/100/4CAF50/ffffff?text=🌿" class="spinning-img" width="100">
            <img src="https://via.placeholder.com/100/4CAF50/ffffff?text=🍃" class="spinning-img" width="100">
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"<h1 style='text-align: center;'>{translations['en']['select_language']}</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🇫🇷 Français"):
            st.session_state.lang = 'fr'
            st.session_state.language_selected = True
            st.session_state.page = 'home'
            st.rerun()
    
    with col2:
        if st.button("🇬🇧 English"):
            st.session_state.lang = 'en'
            st.session_state.language_selected = True
            st.session_state.page = 'home'
            st.rerun()
    
    with col3:
        if st.button("🇪🇸 Español"):
            st.session_state.lang = 'es'
            st.session_state.language_selected = True
            st.session_state.page = 'home'
            st.rerun()

# =============================
# 10. PAGE 2 : ACCUEIL
# =============================
elif st.session_state.page == 'home':
    t = translations[st.session_state.lang]
    
    # Navigation
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🏠"):
            st.session_state.page = 'language'
            st.session_state.language_selected = False
            st.rerun()
    
    # Titre
    st.markdown(f"<h1 style='text-align: center;'>{t['title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size: 1.2rem;'>{t['subtitle']}</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Images décoratives
    st.markdown("""
    <div style="display: flex; justify-content: center; gap: 40px; margin: 40px 0;">
        <img src="https://via.placeholder.com/120/4CAF50/ffffff?text=🌱" class="spinning-img" width="120">
        <img src="https://via.placeholder.com/120/4CAF50/ffffff?text=🌿" class="spinning-img" width="120">
        <img src="https://via.placeholder.com/120/4CAF50/ffffff?text=🍃" class="spinning-img" width="120">
    </div>
    """, unsafe_allow_html=True)
    
    # Message de bienvenue
    st.markdown(f"<div style='background-color: rgba(255,255,255,0.7); padding: 1.5rem; border-radius: 10px; margin: 2rem 0;'>{t['welcome']}</div>", unsafe_allow_html=True)
    
    # Catégories
    st.subheader(f"📋 {t['categories']}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        for i in range(0, 6):
            st.markdown(f"<div class='category-card'>• {format_class_name(CLASSES[i])}</div>", unsafe_allow_html=True)
    
    with col2:
        for i in range(6, 12):
            st.markdown(f"<div class='category-card'>• {format_class_name(CLASSES[i])}</div>", unsafe_allow_html=True)
    
    with col3:
        for i in range(12, 17):
            st.markdown(f"<div class='category-card'>• {format_class_name(CLASSES[i])}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Bouton pour aller à la prédiction
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button(t['start_prediction']):
            st.session_state.page = 'prediction'
            st.rerun()

# =============================
# 11. PAGE 3 : PRÉDICTION
# =============================
elif st.session_state.page == 'prediction':
    t = translations[st.session_state.lang]
    
    # Navigation
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("←"):
            st.session_state.page = 'home'
            st.rerun()
    
    # Titre
    st.markdown(f"<h1 style='text-align: center;'>🔍 {t['title']}</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Images décoratives
    st.markdown("""
    <div style="display: flex; justify-content: center; gap: 20px; margin: 20px 0;">
        <img src="https://via.placeholder.com/80/4CAF50/ffffff?text=🌱" class="spinning-img" width="80">
        <img src="https://via.placeholder.com/80/4CAF50/ffffff?text=🌿" class="spinning-img" width="80">
        <img src="https://via.placeholder.com/80/4CAF50/ffffff?text=🍃" class="spinning-img" width="80">
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar pour les paramètres
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        threshold = st.slider(
            t['threshold'],
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help=t['threshold_help']
        )
        st.markdown(f"<small>{t['threshold_info']}</small>", unsafe_allow_html=True)
    
    # Upload
    uploaded_file = st.file_uploader(
        t['upload'],
        type=["png", "jpg", "jpeg", "JPG", "PNG", "JPEG"]
    )
    
    if uploaded_file is not None:
        # Afficher l'image
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="📷 Image")
            
            if st.button(t['analyze']):
                with st.spinner(t['processing']):
                    # Charger modèle
                    model, device = load_model()
                    
                    if model is None:
                        st.error(f"❌ {t['file_error']}")
                    else:
                        # Prédiction
                        transform = get_transform()
                        input_tensor = transform(image).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            outputs = model(input_tensor)
                            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                            pred_idx = np.argmax(probs)
                            max_prob = probs[pred_idx]
                        
                        # Stocker dans session state
                        st.session_state['probs'] = probs
                        st.session_state['max_prob'] = max_prob
                        st.session_state['pred_idx'] = pred_idx
                        st.session_state['analyzed'] = True
                        st.rerun()
        
        with col2:
            if 'analyzed' in st.session_state and st.session_state['analyzed']:
                st.markdown(f"### {t['results']}")
                
                max_prob = st.session_state['max_prob']
                pred_idx = st.session_state['pred_idx']
                probs = st.session_state['probs']
                
                # Décision
                if max_prob >= threshold:
                    st.markdown(
                        f"<div class='success-box'>✅ {t['in_base']}<br><br>"
                        f"🎯 {format_class_name(CLASSES[pred_idx])}<br><br>"
                        f"📊 {max_prob:.2%}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"<div class='reject-box'>❌ {t['not_in_base']}<br><br>"
                        f"📊 {max_prob:.2%} < {threshold:.0%}</div>",
                        unsafe_allow_html=True
                    )
                
                # Top 3
                st.markdown(f"### {t['top3']}")
                top3_idx = np.argsort(probs)[-3:][::-1]
                
                for i, idx in enumerate(top3_idx):
                    prob = probs[idx]
                    st.markdown(f"**{i+1}.** {format_class_name(CLASSES[idx])}: {prob:.2%}")
                    st.progress(float(prob))
                
                # Graphique
                st.markdown("---")
                st.subheader(f"📊 {t['all_categories']}")
                
                # DataFrame pour le graphique
                df = pd.DataFrame({
                    t['category']: [format_class_name(CLASSES[i]) for i in range(len(CLASSES))],
                    t['probability']: probs
                }).sort_values(t['probability'], ascending=False).head(10)
                
                fig = px.bar(
                    df,
                    x=t['probability'],
                    y=t['category'],
                    orientation='h',
                    color=t['probability'],
                    color_continuous_scale=['#000000', '#00FF00'],
                    text_auto='.2%'
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(255,255,255,0.7)',
                    paper_bgcolor='rgba(255,255,255,0)',
                    font_color='black',
                    height=400,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info(f"ℹ️ {t['no_image']}")
        st.markdown("---")
        st.subheader(f"📸 {t['examples']}")
        st.markdown(f"""
        {t['example1']}
        {t['example2']}
        {t['example3']}
        {t['example4']}
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #000; font-size: 0.9rem;'>"
        "🌿 Powered by EfficientNet-B0 | ✅ IN BASE | ❌ NOT IN BASE</p>",
        unsafe_allow_html=True
    )