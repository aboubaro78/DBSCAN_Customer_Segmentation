import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import normalize


# ============================================================
# CONFIGURATION DE LA PAGE
# ============================================================

st.set_page_config(
    page_title="DBSCAN - Segmentation Client",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# STYLE PERSONNALISÉ (Design moderne)
# ============================================================

st.markdown("""
<style>
    /* ===== GLOBAL ===== */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }

    /* ===== TITRES ===== */
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* ===== CARTES ===== */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.4rem 1.2rem;
        text-align: center;
        transition: all 0.25s ease;
        backdrop-filter: blur(8px);
    }

    .metric-card:hover {
        border-color: #60a5fa;
        transform: translateY(-3px);
        box-shadow: 0 10px 25px -5px rgba(96, 165, 250, 0.15);
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #60a5fa;
        margin: 0.3rem 0;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ===== RÉSULTAT SUCCÈS ===== */
    .result-success {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(6, 95, 70, 0.2));
        border: 1px solid #10b981;
        border-radius: 18px;
        padding: 1.8rem 2rem;
        margin-top: 1.5rem;
    }

    .result-anomaly {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(146, 64, 14, 0.2));
        border: 1px solid #f59e0b;
        border-radius: 18px;
        padding: 1.8rem 2rem;
        margin-top: 1.5rem;
    }

    .cluster-name {
        font-size: 1.7rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }

    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid #334155;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: #cbd5e1;
    }

    /* ===== INPUTS ===== */
    .stNumberInput > div > div > input {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border: 1px solid #475569 !important;
        border-radius: 10px !important;
    }

    .stNumberInput label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }

    /* ===== BOUTON ===== */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.35) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.45) !important;
    }

    /* ===== DIVIDER ===== */
    hr {
        border-color: #334155 !important;
        margin: 1.8rem 0 !important;
    }

    /* ===== EXPANDER ===== */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }

    /* ===== DATAFRAME ===== */
    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden;
    }

    /* ===== CAPTION ===== */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

@st.cache_resource
def charger_modele():
    artefacts = joblib.load("modele_dbscan.joblib")
    return artefacts


try:
    artefacts = charger_modele()
except Exception as e:
    st.error("❌ Impossible de charger le fichier `modele_dbscan.joblib`.")
    st.exception(e)
    st.stop()


# ============================================================
# RÉCUPÉRATION DES ARTEFACTS
# ============================================================

points_coeur = np.array(artefacts["points_coeur"])
labels_coeur = np.array(artefacts["labels_coeur"])
eps = float(artefacts["eps"])
colonnes = artefacts["colonnes"]
valeurs_defaut = artefacts["valeurs_defaut"]
exemples = artefacts["exemples"]
noms_classes = artefacts["noms_classes"]


# ============================================================
# FONCTION DE PRÉDICTION
# ============================================================

def predire_cluster(nouveau_client):
    client = np.array(nouveau_client, dtype=float).reshape(1, -1)
    client = normalize(client)

    distances = np.linalg.norm(points_coeur - client, axis=1)
    plus_proche = distances.argmin()
    distance_minimale = distances[plus_proche]
    cluster = labels_coeur[plus_proche]

    if distance_minimale <= eps:
        return int(cluster), float(distance_minimale), True
    else:
        return -1, float(distance_minimale), False


def obtenir_nom_cluster(cluster):
    if cluster == -1:
        return "Anomalie / Aucun cluster"

    try:
        if isinstance(noms_classes, dict):
            return noms_classes.get(cluster, f"Cluster {cluster}")
        if isinstance(noms_classes, list):
            if cluster < len(noms_classes):
                return noms_classes[cluster]
    except Exception:
        pass

    return f"Cluster {cluster}"


# ============================================================
# EN-TÊTE
# ============================================================

st.markdown('<div class="main-title">🔍 Segmentation Client avec DBSCAN</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Analysez un nouveau client et identifiez son groupe à partir des clusters obtenus avec DBSCAN.</div>',
    unsafe_allow_html=True
)


# ============================================================
# MÉTRIQUES DU MODÈLE
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Points cœur</div>
        <div class="metric-value">{len(points_coeur)}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Valeur de ε (eps)</div>
        <div class="metric-value">{round(eps, 4)}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Variables</div>
        <div class="metric-value">{len(colonnes)}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### ⚙️ Informations du modèle")
    st.markdown("---")
    
    st.markdown("**Algorithme**")
    st.code("DBSCAN", language=None)
    
    st.markdown("**Paramètre ε (eps)**")
    st.code(f"{eps}", language=None)
    
    st.markdown("**Nombre de variables**")
    st.code(f"{len(colonnes)}", language=None)
    
    st.markdown("**Points cœur**")
    st.code(f"{len(points_coeur)}", language=None)
    
    st.markdown("---")
    
    st.info(
        "La classification d'un nouveau client est basée sur sa proximité "
        "avec les points cœur du modèle DBSCAN."
    )


# ============================================================
# SAISIE DU NOUVEAU CLIENT
# ============================================================

st.markdown("### 👤 Caractéristiques du nouveau client")
st.caption("Entrez les valeurs correspondant aux variables utilisées lors de l'entraînement.")

valeurs_client = []

# Organisation des inputs en colonnes (2 ou 3 selon le nombre)
n_cols = 2 if len(colonnes) <= 6 else 3
cols = st.columns(n_cols)

for i, colonne in enumerate(colonnes):
    try:
        valeur_defaut = float(valeurs_defaut[i])
    except Exception:
        valeur_defaut = 0.0

    with cols[i % n_cols]:
        valeur = st.number_input(
            label=str(colonne),
            value=valeur_defaut,
            key=f"variable_{i}",
            format="%.4f"
        )
        valeurs_client.append(valeur)

st.write("")


# ============================================================
# BOUTON DE PRÉDICTION
# ============================================================

if st.button("🔎 Analyser le client", use_container_width=True, type="primary"):

    try:
        if len(valeurs_client) != len(colonnes):
            st.error("Le nombre de variables saisies ne correspond pas au modèle.")
            st.stop()

        cluster, distance, appartient = predire_cluster(valeurs_client)

        st.divider()
        st.markdown("### 📊 Résultat de l'analyse")

        if appartient:
            nom_cluster = obtenir_nom_cluster(cluster)

            st.success("✅ Le client appartient à un cluster.")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Cluster identifié", str(cluster))
            with c2:
                st.metric("Distance au point cœur", f"{distance:.4f}")

            st.markdown(f"""
            <div class="result-success">
                <div class="cluster-name">🎯 {nom_cluster}</div>
                <p style="color:#a7f3d0; margin:0.5rem 0 0 0;">
                    Le client est suffisamment proche d'un point cœur du cluster.
                </p>
                <p style="color:#94a3b8; margin:0.8rem 0 0 0; font-size:0.95rem;">
                    <b>Distance :</b> {distance:.4f} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Seuil ε :</b> {eps:.4f}
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.warning("⚠️ Le client est considéré comme une anomalie (n'appartient à aucun cluster).")

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Résultat", "Anomalie")
            with c2:
                st.metric("Distance minimale", f"{distance:.4f}")

            st.markdown(f"""
            <div class="result-anomaly">
                <div class="cluster-name">⚠️ Anomalie détectée</div>
                <p style="color:#fde68a; margin:0.5rem 0 0 0;">
                    La distance minimale ({distance:.4f}) est supérieure au seuil ε ({eps:.4f}).
                </p>
                <p style="color:#94a3b8; margin:0.8rem 0 0 0; font-size:0.95rem;">
                    Le nouveau client n'est pas suffisamment proche d'un point cœur existant.
                </p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error("❌ Une erreur est survenue pendant l'analyse.")
        st.exception(e)


# ============================================================
# EXEMPLES
# ============================================================

st.divider()

with st.expander("📌 Voir quelques exemples des données d'entraînement"):
    try:
        import pandas as pd
        df_exemples = pd.DataFrame(exemples, columns=colonnes)
        st.dataframe(df_exemples, use_container_width=True, hide_index=True)
    except Exception:
        st.write(exemples)


# ============================================================
# PIED DE PAGE
# ============================================================

st.markdown("""
<div class="footer">
    Application de segmentation client basée sur DBSCAN<br>
    Règle d'affectation par proximité aux points cœur
</div>
""", unsafe_allow_html=True)