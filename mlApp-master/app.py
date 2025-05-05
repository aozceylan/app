import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# Notwendige Imports für diesen Abschnitt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
        confusion_matrix, classification_report, roc_curve, auc, log_loss
    )
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px

import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import os


st.set_page_config(page_title="ML Workflow", layout="wide")

# İlk önce is_home_page fonksiyonunu kullanmadan önce active_components değişkeninin
# var olduğundan emin olmak için kontrol ekleyelim
if 'active_components' not in st.session_state:
    st.session_state.active_components = {
        "data_import": False,
        "data_visualization": False,
        "select_columns": False,
        "data_sampling": False,
        "modeling": False,
        "evaluation": False,
        "prediction": False 
    }

def is_home_page():
    # Eğer hiçbir component aktif değilse veya sadece ana sayfa aktifse
    return not any(st.session_state.active_components.values()) or (
        st.session_state.active_components.get("data_import", False) == False and
        st.session_state.active_components.get("data_visualization", False) == False and
        st.session_state.active_components.get("select_columns", False) == False and
        st.session_state.active_components.get("data_sampling", False) == False and
        st.session_state.active_components.get("modeling", False) == False and
        st.session_state.active_components.get("evaluation", False) == False
    )
def apply_custom_styling():
    home_page = is_home_page()
    top_right_size = "200px" if home_page else "200px"
    top_right_left = "100px" if home_page else "100px"
    
    bottom_left_size = "700px" if home_page else "500px"
    bottom_left_right = "300px" if home_page else "300px"
    
    # f-string kullanarak dinamik değerleri CSS'e ekleyelim
    st.markdown(f"""
     <style>
        /* Daha spesifik CSS seçicileri */
        div[data-testid="stHeader"] {{display: none;}}
        
        .stApp h1,
        .stApp .stMarkdown h1,
        .stApp header h1,
        section[data-testid="stSidebar"] + div h1,
        .main .block-container h1 {{
            color:#007E92 !important;
            text-align: center !important;
            font-size: 2.5rem !important;
            padding: 1rem 0 !important;
        }}
        
        /* Streamlit üst çubuğunu gizlemek */
        header {{
            visibility: hidden;
        }} 

        /* Sayfa üst boşluğunu tamamen kaldır */
        .block-container {{
            padding-top: 0 !important;
            margin-top: 0 !important;
        }}

        /* Logo-Container ile logoyu yukarı kaydırma */
        .logo-container {{
            position: absolute;
            top: -100px;  /* Logo'yu yukarı kaydır */
            left: 0;
            padding: 5px;
            z-index: 1000;
        }}

        /* Global Styling */
        .stApp {{
            background-color: #ffffff; /* Beyaz arka plan */
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }}

        /* Sidebar Styling */
        div[data-testid="stSidebar"] {{
            background-color: #ffffff !important;
            border-right: 2px solid #e0e0e0;
            box-shadow: 2px 0 5px rgba(0,0,0,0.0);
        }}

        .css-1aumxhk {{
            background-color: #ffffff;
        }}

        .stApp {{
            background-image: url("./images/logo2.png");
            background-size: cover;
        }}
        
        .stApp {{
            background-color: rgba(0, 0, 0, 0);  /* Arka planı şeffaf yap */
        }}        

        /* Yatay çizgi */
        .horizontal-line {{
            border-top: 1px solid #333;
            margin: 20px 0;
        }}

        /* Button Styling */
        .stButton>button {{
            background-color: #007E92; /* Buton rengi */
            color: white;
            border-radius: 8px;
            border: none;
            transition: all 0.3s ease;
        }}

        .stButton>button:hover {{
            background-color: #005f6b; /* Hover rengi */
            transform: scale(1.05);
        }}

        /* Header Styling */
        h1, h2, h3, h4, h5, h6 {{
            color: #2c3e50;
            font-weight: 600;
        }}

        /* Card-like containers */
        .stDataFrame, .stTable {{
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 10px;
            margin-bottom: 20px;
        }}

        /* Chart and Plot Styling */
        .stPlotlyChart {{
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 10px;
        }}

        /* Metric Styling */
        .stMetric {{
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            padding: 15px;
        }}

        /* Radio and Select Widgets */
        .stRadio, .stSelectbox {{
            background-color: white;
            border-radius: 8px;
            padding: 10px;
        }}

        /* Progress and Spinner */
        .stSpinner > div {{
            border-color: #007E92 transparent #007E92 transparent;
        }}

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: #f0f2f6;
            border-radius: 10px;
        }}

        .stTabs [data-baseweb="tab"] {{
            color: #2c3e50;
            background-color: transparent;
            transition: all 0.3s ease;
        }}

        .stTabs [data-baseweb="tab"]:hover {{
            color: #007E92; /* Hover rengi */
        }}

        .stTabs [data-baseweb="tab"][aria-selected="true"] {{
            background-color: #007E92; /* Seçili sekme rengi */
            color: white;
        }}
        
        /* Obere rechte Ecke Dreieck - angepasst nach Seitentyp */
        .top-right-triangle {{
            position: fixed;
            top: 0;
            right: 0;
            width: 0;
            height: 0;
            border-top: {top_right_size} solid #007E92;
            border-left: {top_right_left} solid transparent;
            z-index: 999;
        }}
           
        /* Untere linke Ecke Dreieck - angepasst nach Seitentyp */
        .bottom-left-triangle {{
            position: fixed;
            bottom: 0;
            left: 0; /* right yerine left kullanarak sol tarafa taşıdık */
            width: 0;
            height: 0;
            border-bottom: {bottom_left_size} solid #008E88;
            border-right: {bottom_left_right} solid transparent; /* border-left yerine border-right kullanarak sola hizaladık */
            z-index: 999;
        }}
        
        /* =============== GELİŞMİŞ CSS KODLARI =============== */
        
        /* Multiselect tag'leri için renkleri özelleştirme */
        span[data-baseweb="tag"] {{
            background-color: #007E92 !important;
            color: white !important;
            border-color: #007E92 !important;
        }}

        span[data-baseweb="tag"] span {{
            color: white !important;
        }}

        span[data-baseweb="tag"]:hover {{
            background-color: #005f6b !important;
        }}

        span[data-baseweb="tag"] button {{
            color: white !important;
        }}

        /* Streamlit default checkbox stilini değiştirme */
        [role="checkbox"][data-baseweb="checkbox"] {{
            background-color: #007E92 !important;
        }}
        .st-emotion-cache-1dj3ksd e8lt0n70,
        [role="slider"] {{
        background-color: #008E88 !important; /* İstediğiniz rengi buraya yazın */
        }}
        [data-baseweb="slider"] div[role="slider"] + div div {{
        background-color:#008E88 !important;
        }}
        
        /* Sadece dolgu kısmını hedefle */
        [data-testid="stSlider"] [data-baseweb="slider"] div div div div {{
        background-color: #008E88 !important;
        }}
        /* Hareketli thumb noktasını hedefle */
        [data-testid="stSlider"] [role="slider"] {{
        background-color: #FF5733 !important;
        }}
                .st-emotion-cache-b92z60,
        [data-testid="stSliderThumbValue"] {{
        background-color: #008E88 !important; /* Sayı kutucuğunun arkaplan rengi */
        color: white !important; /* Sayı rengi */
        }}
        /* Checkbox için ek seçiciler */
        [data-testid="stCheckbox"] > div[role="checkbox"] {{
            background-color: #007E92 !important;
            border-color: #007E92 !important;
        }}

        /* Checkbox işaretini beyaz yap */
        [data-testid="stCheckbox"] > div[role="checkbox"] > svg {{
            color: white !important;
        }}

        /* Radio button için ek seçiciler */
        [data-testid="stRadio"] label div[role="radio"][data-checked="true"] {{
            background-color: #007E92 !important;
            border-color: #007E92 !important;
        }}

        [data-testid="stRadio"] label div[role="radio"][data-checked="true"] div {{
            background-color: white !important;
        }}

        /* MultiSelect seçili etiketler için stil */
        div[data-baseweb="select"] > div > div > div > div > div > div span[data-baseweb="tag"] {{
            background-color: #007E92 !important;
            color: white !important;
        }}

        /* MultiSelect etiketlerdeki çarpı işareti */
        div[data-baseweb="select"] > div > div > div > div > div > div span[data-baseweb="tag"] svg {{
            color: white !important;
        }}

        /* MultiSelect etiketleri kenar renkleri */
        div[data-baseweb="select"] > div > div > div > div > div > div span[data-baseweb="tag"] div {{
            border-color: #007E92 !important;
        }}

        /* Açılır oku değiştir */
        div[data-baseweb="select"] > div > div > div:last-child > svg {{
            color: #007E92 !important;
        }}
        
        /* Dropdown ve selectbox stillerini değiştir */
        div[data-baseweb="select"] {{
            font-weight: 400;
        }}

        div[data-baseweb="select"] > div {{
            background-color: white;
            border-color: #007E92;
        }}

        div[data-baseweb="select"]:hover > div {{
            border-color: #005f6b;
        }}

        div[data-baseweb="base-input"] > div {{
            background-color: white;
            border-color: #007E92;
        }}

        div[data-baseweb="base-input"]:hover > div {{
            border-color: #005f6b;
        }}
        .stRadio label div[role="radio"] {{
           border-color: #007E92 !important;
        }}

        .stRadio label div[role="radio"][data-checked="true"] div {{
            background-color: #007E92 !important;
        }}


        /* Checkbox stilini değiştir */
        .stCheckbox > div > div > label > div[role="checkbox"] {{
            border-color: #007E92;
        }}

        .stCheckbox > div > div > label > div[data-checked="true"] {{
            background-color: #007E92;
            border-color: #007E92;
        }}
        
            
        
     </style>
     <div class="top-left-triangle"></div>
     <div class="top-right-triangle"></div>
     <div class="bottom-left-triangle"></div>
     <div class="bottom-right-triangle"></div>
    """, unsafe_allow_html=True)
# Logo kısmını apply_custom_styling fonksiyonundan sonra çağırmalıyız
# Böylece CSS stillemesi doğru şekilde uygulanacaktır
apply_custom_styling()

# Şimdi logoyu ekleyelim
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image('images/logo1.png', width=700)
st.title("ML Workflow")
# Seitenkonfiguration
#st.set_page_config(page_title="ML Workflow ", layout="wide")

# Initialisierung von session_state Variablen, falls sie noch nicht existieren
if 'data' not in st.session_state:
    st.session_state.data = None
if "serialized_model" not in st.session_state:
    st.session_state.serialized_model = None
if "model_to_save" not in st.session_state:
    st.session_state.model_to_save = None    
if 'selected_columns' not in st.session_state:
    st.session_state.selected_columns = None
if 'sampled_data' not in st.session_state:
    st.session_state.sampled_data = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'active_components' not in st.session_state:
    st.session_state.active_components = {
        "data_import": False,
        "data_visualization": False,
        "select_columns": False,
        "data_sampling": False,
        "modeling": False,
        "evaluation": False,
        "prediction": False 
    }

# Funktion zum Aktivieren eines bestimmten Workflow-Schritts
def activate_component(component_name):
    st.session_state.active_components = {k: False for k in st.session_state.active_components}
    st.session_state.active_components[component_name] = True
    st.rerun()
# Seitenleiste mit Workflow-Komponenten
st.sidebar.title("ML Workflow")
st.sidebar.write("Wählen Sie die Schritte für Ihren ML-Workflow aus")

# Komponenten in der Seitenleiste
components = {
    "data_import": "1. Daten importieren",
    "data_visualization": "2. Daten visualisieren",
    "select_columns": "3. Datenbereinigung",
    "data_sampling": "4. Daten-Sampling",
    "modeling": "5. Modellierung",
    "evaluation": "6. Modell-Evaluation",
    "prediction":"7. Prediction"
}

# Buttons für jeden Workflow-Schritt
for key, label in components.items():
    # Buttons nur aktivieren, wenn Voraussetzungen erfüllt sind
    disabled = False
    
    if key == "data_visualization" and st.session_state.data is None:
        disabled = True
    elif key == "select_columns" and st.session_state.data is None:
        disabled = True
    elif key == "data_sampling" and st.session_state.selected_columns is None:
        disabled = True
    elif key == "modeling" and st.session_state.sampled_data is None:
        disabled = True
    elif key == "evaluation" and (not 'models' in st.session_state or len(st.session_state.models) == 0):
        disabled = True
    elif key == "prediction" and (not 'models' in st.session_state or len(st.session_state.models) == 0):
        disabled = True    
    if st.sidebar.button(label, disabled=disabled, key=f"btn_{key}"):
        activate_component(key)

# Hauptbereich

# Ordnerstruktur im Container anlegen
pfad = '/app/datensätze'

if not os.path.exists(pfad):
    os.makedirs(pfad)

def speichere_datei(datei, pfad):
    with open(os.path.join(pfad, datei.name), 'r') as f:
        f.write(datei.getbuffer())
       
    
# 1. Daten importieren
if st.session_state.active_components["data_import"]:
    st.header("1. Daten importieren")
    
    # Option zum Hochladen einer CSV-Datei oder Verwendung von Beispieldaten
    data_option = st.radio(
        "Datenquelle auswählen",
        ["CSV-Datei hochladen"]
    )
    
    if data_option == "CSV-Datei hochladen":
        uploaded_file = st.file_uploader("Wählen Sie eine CSV-Datei", type="csv")
        if uploaded_file is not None:
            speichere_datei(uploaded_file, pfad)
            st.success("Daten erfolgreich geladen!")
            # Nach erfolgreicher Datenladung automatisch zum nächsten Schritt
            df = pd.read_csv(os.path.join(pfad, uploaded_file.name))
            st.session_state.data = df
            st.rerun()
        
    # Zeige die Daten an, wenn sie geladen sind
    if st.session_state.data is not None:
        st.write("Vorschau der Daten:")
        st.dataframe(st.session_state.data.head(10))
        st.write(f"Form: {st.session_state.data.shape[0]} Zeilen, {st.session_state.data.shape[1]} Spalten")
        
        # Button zum Fortfahren
        if st.button("Weiter zur Datenvisualisierung"):
            activate_component("data_visualization")
            st.experimental_rerun()

# 2. Daten visualisieren
elif st.session_state.active_components["data_visualization"]:
    st.header("2. Daten visualisieren")
    
    if st.session_state.data is not None:
        # Zwei Tabs für verschiedene Visualisierungsarten
        viz_tab1, viz_tab2 = st.tabs(["Scatter Plot", "Feature-Statistiken"])
        
        with viz_tab1:
            st.subheader("Scatter Plot")
            
            # Auswahl der Spalten für x und y
            numeric_columns = st.session_state.data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                x_col = st.selectbox("X-Achse", options=numeric_columns, index=0)
            with col2:
                y_col = st.selectbox("Y-Achse", options=numeric_columns, index=1 if len(numeric_columns) > 1 else 0)
            with col3:
                color_col = st.selectbox("Färben nach", options=st.session_state.data.columns)
            
            # Scatter Plot mit Plotly
            fig = px.scatter(
                st.session_state.data, 
                x=x_col, 
                y=y_col, 
                color=color_col,
                title=f"Scatter Plot: {x_col} vs {y_col}",
                labels={x_col: x_col, y_col: y_col},
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with viz_tab2:
            st.subheader("Feature-Statistiken")
            
            # Deskriptive Statistiken
            st.write("Deskriptive Statistiken")
            
            # Describe ile istatistikleri al ve virgülden sonra 2 rakam gösterecek şekilde biçimlendir
            desc_df = st.session_state.data.describe()
            
            # Tüm sayısal değerleri virgülden sonra 2 rakamla formatla
            formatted_df = desc_df.style.format("{:.2f}")
            
            # Tabloyu göster
            st.dataframe(formatted_df)
                # Button zum Fortfahren
        if st.button("Weiter zur Spaltenauswahl"):
            activate_component("select_columns")
            st.rerun()
    else:
        st.warning("Bitte laden Sie zuerst Daten.")
        if st.button("Zurück zum Datenimport"):
            activate_component("data_import")
            st.experimental_rerun()

# 3. Spalten auswählen
elif st.session_state.active_components["select_columns"]:
    st.header("3. Datenbereinigung")
    
    if st.session_state.data is not None:
        # Zeige alle verfügbaren Spalten an
        st.write("Verfügbare Spalten:")
        st.dataframe(pd.DataFrame({
            'Spaltenname': st.session_state.data.columns,
            'Datentyp': st.session_state.data.dtypes.values
        }))
        
        # Mehrfachauswahl für Features
        st.subheader("Features auswählen")
        feature_cols = st.multiselect(
            "Wählen Sie die Feature-Spalten aus:",
            options=[col for col in st.session_state.data.columns],
            default=[col for col in st.session_state.data.columns if col != 'target']
        )
        
        # Zielspaltenwahl
        st.subheader("Zielvariable auswählen")
        target_col = st.selectbox(
            "Wählen Sie die Zielspalte aus:",
            options=st.session_state.data.columns.tolist(),
            index=st.session_state.data.columns.get_loc('target') if 'target' in st.session_state.data.columns else 0
        )
        
        if st.button("Spalten übernehmen"):
            if not feature_cols:
                st.error("Bitte wählen Sie mindestens eine Feature-Spalte aus.")
            else:
                st.session_state.selected_columns = {
                    'features': feature_cols,
                    'target': target_col
                }
                st.success(f"{len(feature_cols)} Feature-Spalten und 1 Zielspalte ausgewählt!")
                
                # Zeige ausgewählte Spalten an
                selected_data = st.session_state.data[feature_cols + [target_col]]
                st.write("Vorschau der ausgewählten Spalten:")
                st.dataframe(selected_data.head())
                
                # Button zum Fortfahren
                if st.button("Weiter zum Daten-Sampling"):
                    activate_component("data_sampling")
                    st.experimental_rerun()
    else:
        st.warning("Bitte laden Sie zuerst Daten.")
        if st.button("Zurück zum Datenimport"):
            activate_component("data_import")
            st.experimental_rerun()

# 4. Daten-Sampling
elif st.session_state.active_components["data_sampling"]:
    st.header("4. Daten-Sampling")
    
    if st.session_state.selected_columns is not None:
        features = st.session_state.selected_columns['features']
        target = st.session_state.selected_columns['target']
        
        # Bereite Daten vor
        selected_data = st.session_state.data[features + [target]]
        
        # Optionen für Daten-Sampling
        st.subheader("Daten aufteilen")
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Testdaten-Anteil (%)", 10, 50, 30) / 100
        with col2:
            random_state = st.number_input("Random State (für Reproduzierbarkeit)", 0, 100, 42)
            
        # Stratifiziertes Sampling bei Klassifikationsproblemen
        stratify_option = st.checkbox("Stratifiziertes Sampling", value=True)
        
        if st.button("Daten aufteilen"):
            X = selected_data[features]
            y = selected_data[target]
            
            # Prüfe, ob wir stratifizieren sollten
            stratify_val = y if stratify_option else None
            
            # Daten aufteilen
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify_val
            )
            
            # Speichere aufgeteilte Daten im Session State
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.sampled_data = True
            
            # Zeige Zusammenfassung
            st.success("Daten erfolgreich aufgeteilt!")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Trainingsdaten: {X_train.shape[0]} Beispiele")
                st.dataframe(pd.concat([X_train, y_train], axis=1).head())
            with col2:
                st.write(f"Testdaten: {X_test.shape[0]} Beispiele")
                st.dataframe(pd.concat([X_test, y_test], axis=1).head())
            
            # Zeige Verteilung der Klassen
            if y.dtype == 'object' or y.dtype == 'category':
                st.subheader("Klassenverteilung")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Training:")
                    train_dist = y_train.value_counts().reset_index()
                    train_dist.columns = [target, 'Anzahl']
                    train_fig = px.pie(train_dist, values='Anzahl', names=target, title="Trainingsdaten")
                    st.plotly_chart(train_fig, use_container_width=True)
                with col2:
                    st.write("Test:")
                    test_dist = y_test.value_counts().reset_index()
                    test_dist.columns = [target, 'Anzahl']
                    test_fig = px.pie(test_dist, values='Anzahl', names=target, title="Testdaten")
                    st.plotly_chart(test_fig, use_container_width=True)
                
                # Button zum Fortfahren
                if st.button("Weiter zur Modellierung"):
                    activate_component("modeling")
                    st.experimental_rerun()
    else:
        st.warning("Bitte wählen Sie zuerst die Spalten aus.")
        if st.button("Zurück zur Spaltenauswahl"):
            activate_component("select_columns")
            st.experimental_rerun()
# 5. Verbesserter Modellierungsabschnitt mit Orange-ähnlicher Funktionalität
elif st.session_state.active_components["modeling"]:
    st.header("5. Modellierung", divider="orange")
    
    if st.session_state.sampled_data:
        # Initialisiere ein Dictionary für mehrere Modelle, falls nicht vorhanden
        if 'models' not in st.session_state:
            st.session_state.models = {}
        
        # Informationsbereich: Trainingsstatistik
        st.info(f"Trainings-/Testdaten Split: {len(st.session_state.X_train)} Training / {len(st.session_state.X_test)} Test")
        
        # Dashboard für Trainingsstatistik
        st.write("#### Dataset-Statistik")
        col1, col2, col3 = st.columns(3)
        col1.metric("Trainingsdaten", f"{len(st.session_state.X_train)}")
        col2.metric("Testdaten", f"{len(st.session_state.X_test)}")
        col3.metric("Gesamtdatensatz", f"{len(st.session_state.data)}")
        
        # Zeige bereits trainierte Modelle an
        if st.session_state.models:
            st.subheader("Trainierte Modelle")
            model_df = pd.DataFrame([
                {
                    "Modell": model_id,
                    "Typ": model_info['type'],
                    "Parameter": str(model_info['params']),
                    "Datenpunkte": len(model_info['full_predictions']),
                    "Accuracy (Test)": f"{accuracy_score(st.session_state.y_test, model_info['test_predictions']):.4f}"
                }
                for model_id, model_info in st.session_state.models.items()
            ])
            st.dataframe(model_df, use_container_width=True)
        
        # Orange-Style Tabs für unterschiedliche Learner
        learner_tab = st.tabs(["Learner",])[0]
        
        with learner_tab:
            # Modellauswahl mit Orange-ähnlichem Styling
            model_type = st.radio(
                "Lernalgorithmus auswählen",
                ["k-Nearest Neighbors (kNN)", "Entscheidungsbaum (Tree)", "Logistische Regression"],
                horizontal=True
            )
            
            # Eindeutige Modell-ID generieren
            with st.form(key="model_form"):
                custom_model_name = st.text_input("Modellname (zur Identifikation)"
                          )
    
                
                # Hyperparameter basierend auf Modelltyp
                if model_type == "k-Nearest Neighbors (kNN)":
                    st.subheader("kNN Hyperparameter")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        n_neighbors = st.slider("Anzahl der Nachbarn (k)", 1, 20, 3)
                    with col2:
                        weights = st.selectbox(
                            "Gewichtungsmethode",
                            options=["uniform", "distance"],
                            index=0,
                            help="'uniform': Alle Nachbarn gleich gewichtet, 'distance': Nach Abstand gewichtet"
                        )
                    with col3:
                        metric = st.selectbox(
                            "Distanzmetrik",
                            options=["euclidean", "manhattan", "chebyshev", "minkowski"],
                            index=0,
                            help="Metrik zur Berechnung der Distanz zwischen den Punkten"
                        )
                    
                    # Preprocessing-Optionen, wie in Orange
                    st.subheader("Preprocessing")
                    standardize = st.checkbox("Daten standardisieren (empfohlen für kNN)", value=True)
                    
                elif model_type == "Entscheidungsbaum (Tree)":
                    st.subheader("Entscheidungsbaum Hyperparameter")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        max_depth = st.slider("Maximale Tiefe", 1, 100, 100,
                                            help="Maximale Tiefe des Baums. 100 bedeutet praktisch unbegrenzt.")
                        min_samples_split = st.slider("Min. Samples für Split", 2, 20, 5)
                    with col2:
                        min_samples_leaf = st.slider("Min. Samples in Blättern", 1, 20, 2)
                        # Split-Kriterium seçeneğini kaldırıldı ve sabit değere ayarlandı
                        criterion = "gini"  # Daima "gini" olarak ayarlandı
                    
                    # Orange-spezifische Baumoptionen
                    binary_splits = st.checkbox("Binäre Splits erzwingen", value=True)
                    limit_depth = st.checkbox("Baumtiefe begrenzen", value=False)
                    if limit_depth:
                        max_depth = st.slider("Max. Tiefe", 1, 50, 10)
                    else:
                        max_depth = None
                    limit_majority = st.checkbox("Stop-Kriterium: Mehrheitsklasse", value=True)
                    if limit_majority:
                        majority_threshold = st.slider("Schwellenwert (%)", 50, 100, 95)
                    else:
                        majority_threshold = 100
                
                elif model_type == "Logistische Regression":  # Logistische Regression
                    st.subheader("Logistische Regression Hyperparameter")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        C = st.slider("Regularisierungsstärke C", 0.01, 10.0, 1.0, step=0.01,
                                    help="Kleinere Werte bedeuten stärkere Regularisierung")
                        solver = st.selectbox(
                            "Solver",
                            options=["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
                            index=0
                        )
                    with col2:
                        penalty = st.selectbox(
                            "Regularisierung",
                            options=["l2", "l1", "elasticnet", "none"],
                            index=0,
                            help="Regularisierungstyp (abhängig vom gewählten Solver)"
                        )
                        max_iter = st.slider("Max. Iterationen", 100, 1000, 100, step=50)
                    
                    # Preprocessing-Optionen
                    st.subheader("Preprocessing")
                    standardize = st.checkbox("Daten standardisieren (empfohlen)", value=True)
                
                # Orange-Style: Validierungsoptionen
                st.subheader("Validation")
                validation_method = st.selectbox(
                    "Validierungsmethode",
                    options=["Test on Train Data", "Test on Test Data"],
                    index=0
                )
                
                # Modell-Training
                submit_button = st.form_submit_button("Modell anwenden")
                
                if submit_button:
                    model_name = custom_model_name
                    with st.spinner("Lernalgorithmus wird trainiert..."):
                        # Daten vorbereiten
                        X_train = st.session_state.X_train
                        y_train = st.session_state.y_train
                        X_test = st.session_state.X_test
                        y_test = st.session_state.y_test
                        
                        # Orange-Stil: Alle Daten zusammenfassen für komplette Vorhersagen
                        features = st.session_state.selected_columns['features']
                        target = st.session_state.selected_columns['target']
                        
                        # Gesamter Datensatz für vollständige Vorhersagen
                        X_full = st.session_state.data[features]
                        y_full = st.session_state.data[target]
                        
                        # Modellspezifische Verarbeitung und Training
                        if model_type == "k-Nearest Neighbors (kNN)":
                            # Standardisierung falls ausgewählt
                            if standardize:
                                scaler = StandardScaler()
                                X_train_scaled = scaler.fit_transform(X_train)
                                X_test_scaled = scaler.transform(X_test)
                                X_full_scaled = scaler.transform(X_full)
                                
                                # Umwandlung in DataFrame für spätere Verwendung
                                X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
                                X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
                                X_full_scaled = pd.DataFrame(X_full_scaled, columns=X_full.columns)
                            else:
                                X_train_scaled = X_train
                                X_test_scaled = X_test
                                X_full_scaled = X_full
                                scaler = None
                            
                            # Modell erstellen und trainieren
                            model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, metric=metric)
                            model.fit(X_train_scaled, y_train)
                            
                            # Vorhersagen
                            train_predictions = model.predict(X_train_scaled)
                            test_predictions = model.predict(X_test_scaled)
                            full_predictions = model.predict(X_full_scaled)
                            
                            # Modell-Metadaten
                            model_metadata = {
                                'type': 'kNN',
                                'instance': model,
                                'standardize': standardize,
                                'scaler': scaler,
                                'params': {'n_neighbors': n_neighbors, 'weights': weights, 'metric': metric},
                                'X_train_processed': X_train_scaled,
                                'X_test_processed': X_test_scaled,
                                'X_full_processed': X_full_scaled,
                            }
                            
                        elif model_type == "Entscheidungsbaum (Tree)":
                            # Modell erstellen und trainieren
                            tree_depth = max_depth if limit_depth else None
                            model = DecisionTreeClassifier(
                                max_depth=tree_depth,
                                criterion=criterion,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                random_state=42
                            )
                            model.fit(X_train, y_train)
                            
                            # Vorhersagen
                            train_predictions = model.predict(X_train)
                            test_predictions = model.predict(X_test)
                            full_predictions = model.predict(X_full)
                            
                            # Modell-Metadaten
                            model_metadata = {
                                'type': 'DecisionTree',
                                'instance': model,
                                'standardize': False,
                                'params': {
                                    'max_depth': tree_depth,
                                    'criterion': criterion,
                                    'min_samples_split': min_samples_split,
                                    'min_samples_leaf': min_samples_leaf,
                                    'binary_splits': binary_splits,
                                    'majority_threshold': f"{majority_threshold}%" if limit_majority else "N/A"
                                },
                                'X_train_processed': X_train,
                                'X_test_processed': X_test,
                                'X_full_processed': X_full,
                            }
                            
                        else:  # Logistische Regression
                            # Standardisierung
                            if standardize:
                                scaler = StandardScaler()
                                X_train_scaled = scaler.fit_transform(X_train)
                                X_test_scaled = scaler.transform(X_test)
                                X_full_scaled = scaler.transform(X_full)
                                
                                # Umwandlung in DataFrame für spätere Verwendung
                                X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
                                X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
                                X_full_scaled = pd.DataFrame(X_full_scaled, columns=X_full.columns)
                            else:
                                X_train_scaled = X_train
                                X_test_scaled = X_test
                                X_full_scaled = X_full
                                scaler = None
                            
                            # Modell erstellen und trainieren
                            model = LogisticRegression(
                                C=C,
                                penalty=penalty,
                                solver=solver,
                                max_iter=max_iter,
                                random_state=42,
                                multi_class='auto'
                            )
                            model.fit(X_train_scaled, y_train)
                            
                            # Vorhersagen
                            train_predictions = model.predict(X_train_scaled)
                            test_predictions = model.predict(X_test_scaled)
                            full_predictions = model.predict(X_full_scaled)
                            
                            # Modell-Metadaten
                            model_metadata = {
                                'type': 'LogisticRegression',
                                'instance': model,
                                'standardize': standardize,
                                'scaler': scaler,
                                'params': {'C': C, 'penalty': penalty, 'solver': solver, 'max_iter': max_iter},
                                'X_train_processed': X_train_scaled,
                                'X_test_processed': X_test_scaled,
                                'X_full_processed': X_full_scaled,
                            }
                        
                        # Gemeinsame Modell-Speicherung für alle Typen
                        model_metadata.update({
                            'train_predictions': train_predictions,
                            'test_predictions': test_predictions,
                            'full_predictions': full_predictions,
                            'predictions': test_predictions,  # Für Kompatibilität mit altem Code
                            'y_full': y_full,  # Speichern der tatsächlichen Werte für den gesamten Datensatz
                            'y_train': y_train,
                            'y_test': y_test,
                            'validation_method': validation_method
                        })
                        
                        # Modell speichern
                        st.session_state.models[model_name] = model_metadata
                        
                        # Aktuelles Modell setzen für die Evaluation
                        st.session_state.current_model = model_name
                        
                        # Orange-Style Zusammenfassung
                        st.success(f"{model_type}-Modell '{model_name}' erfolgreich trainiert!")
                        
                        # Orange-Style: Zusammenfassung direkt nach dem Training anzeigen
                        acc_train = accuracy_score(y_train, train_predictions)
                        acc_test = accuracy_score(y_test, test_predictions)
                        
                        eval_col1, eval_col2 = st.columns(2)
                        eval_col1.metric("Trainingsgenauigkeit", f"{acc_train:.4f}")
                        eval_col2.metric("Testgenauigkeit", f"{acc_test:.4f}", 
                                       delta=f"{acc_test - acc_train:.4f}")
                        
                        # Bei Entscheidungsbäumen: Feature-Importance anzeigen
                        if model_type == "Entscheidungsbaum (Tree)":
                            st.subheader("Feature-Importance")
                            feature_importance = pd.DataFrame({
                                'Feature': X_train.columns,
                                'Importance': model.feature_importances_
                            }).sort_values('Importance', ascending=False)
                            
                            fig = px.bar(
                                feature_importance, 
                                x='Feature', 
                                y='Importance',
                                title=f"Feature-Importance des Entscheidungsbaums '{model_name}'"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        if model_type == "Logistische Regression" and penalty != "none":
                            st.subheader("Koeffizienten")
                            coef_df = pd.DataFrame({
                                'Feature': X_train.columns,
                                'Koeffizient': model.coef_[0] if model.coef_.shape[0] == 1 else model.coef_.mean(axis=0)
                            }).sort_values('Koeffizient', ascending=False)
                            
                            fig = px.bar(
                                coef_df, 
                                x='Feature', 
                                y='Koeffizient',
                                title=f"Koeffizienten der Logistischen Regression '{model_name}'"
                            )
                            st.plotly_chart(fig, use_container_width=True)
      # Save Model Widget (Orange-Style)
    if st.session_state.models:
        save_expander = st.expander("Modell speichern (Save Model)", expanded=False)
        with save_expander:
            st.write("#### 💾 Modell für spätere Verwendung speichern")
            
            # Modell zum Speichern auswählen
            model_to_save = st.selectbox(
                "Modell zum Speichern auswählen",
                options=list(st.session_state.models.keys())
            )
            
            # Speicheroption
            if st.button("Modell exportieren"):
             import pickle

            # Tüm gerekli bilgileri içeren bir model paketi oluştur
             model_package = {
                'model': st.session_state.models[model_to_save]['instance'],
                'scaler': st.session_state.models[model_to_save].get('scaler', None),  # .get() metodu ile güvenli erişim
                'standardize': st.session_state.models[model_to_save].get('standardize', False),  # .get() metodu
                'feature_names': st.session_state.models[model_to_save]['X_train_processed'].columns.tolist()
            }

            # Tüm paketi serialize et
             st.session_state.serialized_model = pickle.dumps(model_package)
             st.session_state.model_to_save = model_to_save
            # Download-Button anzeigen
              # Eğer model export edildiyse, indirme butonunu göster
        if st.session_state.serialized_model:
            st.download_button(
                label="Modell herunterladen (.pkl)",
                data=st.session_state.serialized_model,
                file_name=f"{st.session_state.model_to_save}.pkl",
                mime="application/octet-stream"
            )

            st.success(f"Modell '{st.session_state.model_to_save}' kann jetzt heruntergeladen werden.")
            st.info("Dieses Modell kann später mit dem Prediction-Widget wieder geladen werden.")
                    # Button zum Fortfahren zur Evaluation
        if st.session_state.models:
            if st.button("Weiter zur Evaluation", type="primary"):
                activate_component("evaluation")
                st.rerun()
                
    else:
        st.warning("Bitte führen Sie zuerst das Daten-Sampling durch.")
        if st.button("Zurück zum Daten-Sampling"):
            activate_component("data_sampling")
            st.rerun()
# 6. Sadeleştirilmiş Model Değerlendirme (Test and Score)
elif st.session_state.active_components["evaluation"]:
    st.header("6. Modell-Evaluation", divider="orange")
    
    if st.session_state.models:
        # Ana sayfa bir sekmeli arayüz oluşturun: karşılaştırma ve detaylar
        main_tab1, main_tab2 = st.tabs(["Modell Vergleich", "Modell Details"])
        
        with main_tab1:
            st.subheader("Modellvergleich")
            
            # Karşılaştırılacak modelleri seçin - maksimum 2 model
            model_options = list(st.session_state.models.keys())
            
            col1, col2 = st.columns(2)
            with col1:
                model1 = st.selectbox("Modell 1", model_options, 
                                    index=0 if model_options else 0,
                                    key="model1_select")
            with col2:
                remaining_models = [m for m in model_options if m != model1]
                model2 = st.selectbox("Modell 2", 
                                    options=remaining_models,
                                    index=0 if remaining_models else 0,
                                    key="model2_select")
            
            if len(model_options) > 1:
                # Veri setiyle değerlendirme
                test_or_train = st.radio(
                    "Datensatz wählen",
                    ["Testdaten", "Trainingsdaten"],
                    horizontal=True
                )
                
                # Seçilen modeller hakkında bilgi alın
                model1_info = st.session_state.models[model1]
                
                if model2 in st.session_state.models:
                    model2_info = st.session_state.models[model2]
                    
                    # Doğru değerlendirme verilerini belirleyin
                    if test_or_train == "Testdaten":
                        X_data = st.session_state.X_test
                        y_true = st.session_state.y_test
                        y_pred1 = model1_info['test_predictions']
                        y_pred2 = model2_info['test_predictions']
                    else:
                        X_data = st.session_state.X_train
                        y_true = st.session_state.y_train
                        y_pred1 = model1_info['train_predictions']
                        y_pred2 = model2_info['train_predictions']
                    
                    # Kıyaslama metriklerini hesaplayın
                    metrics = {
                        "Genauigkeit (Accuracy)": [
                            accuracy_score(y_true, y_pred1),
                            accuracy_score(y_true, y_pred2)
                        ],
                         "Precision": [
                            precision_score(y_true, y_pred1, average='weighted', zero_division=0),
                            precision_score(y_true, y_pred2, average='weighted', zero_division=0)
                        ],
                        "Recall": [
                            recall_score(y_true, y_pred1, average='weighted', zero_division=0),
                            recall_score(y_true, y_pred2, average='weighted', zero_division=0)
                        ],
                        "F1-Score": [
                            f1_score(y_true, y_pred1, average='weighted', zero_division=0),
                            f1_score(y_true, y_pred2, average='weighted', zero_division=0)
                        ],
                    }
                    
                    # 1. Metrikleri görselleştirin
                    st.write("##### Performance-Metriken im Vergleich")
                    metrics_df = pd.DataFrame(metrics, index=[model1, model2]).T
                    st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)
                    
                    # 2. Metrikleri grafik olarak karşılaştırın
                    metrics_melted = pd.melt(
                        metrics_df.reset_index(),
                        id_vars='index',
                        var_name='Modell',
                        value_name='Wert'
                    )
                    metrics_melted['Wert'] = pd.to_numeric(metrics_melted['Wert'], errors='coerce')
                    metrics_melted = metrics_melted.dropna(subset=['Wert'])
                    fig = px.bar(
                        metrics_melted, 
                        x='index', 
                        y='Wert', 
                        color='Modell', 
                        barmode='group',
                        title="Modellvergleich",
                        labels={'index': 'Metrik', 'Wert': 'Wert'},
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 3. Confusion Matrices
                    st.write("##### Confusion Matrices")
                    
                    cm_col1, cm_col2 = st.columns(2)
                    
                    with cm_col1:
                        st.write(f"**{model1}**")
                        cm1 = confusion_matrix(y_true, y_pred1)
                        classes = sorted(y_true.unique())
                        
                        fig1, ax1 = plt.subplots(figsize=(4, 3))
                        sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', 
                                xticklabels=classes, yticklabels=classes)
                        plt.title(f'Confusion Matrix - {model1}')
                        plt.ylabel('Tatsächliche Klasse')
                        plt.xlabel('Vorhergesagte Klasse')
                        st.pyplot(fig1)
                    
                    with cm_col2:
                        st.write(f"**{model2}**")
                        cm2 = confusion_matrix(y_true, y_pred2)
                        
                        fig2, ax2 = plt.subplots(figsize=(4, 3))
                        sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', 
                                xticklabels=classes, yticklabels=classes)
                        plt.title(f'Confusion Matrix - {model2}')
                        plt.ylabel('Tatsächliche Klasse')
                        plt.xlabel('Vorhergesagte Klasse')
                        st.pyplot(fig2)
                else:
                    st.warning("Bitte wählen Sie zwei verschiedene Modelle für den Vergleich aus.")
            else:
                st.warning("Sie benötigen mindestens zwei trainierte Modelle für einen Vergleich.")
        
        with main_tab2:
            st.subheader("Modelldetails")
            
            # Detaylı inceleme için model seçin
            detailed_model = st.selectbox(
                "Modell für detaillierte Analyse",
                options=model_options,
                index=0 if model_options else 0
            )
            
            if detailed_model:
                model_info = st.session_state.models[detailed_model]
                
                # Modeli yorumlayın
                st.write(f"##### {detailed_model} ({model_info['type']})")
                
                # Modelin parametrelerini göster
                st.write("**Parameter:**")
                params_df = pd.DataFrame([model_info['params']])
                st.dataframe(params_df.T.rename(columns={0: "Wert"}))
                
                # Veri setini seçin
                test_or_train_detail = st.radio(
                    "Datensatz für Detail-Analyse",
                    ["Testdaten", "Trainingsdaten"],
                    horizontal=True,
                    key="detail_data_choice"
                )
                
                # Doğru değerlendirme verilerini belirleyin
                if test_or_train_detail == "Testdaten":
                    X_data = st.session_state.X_test
                    y_true = st.session_state.y_test
                    y_pred = model_info['test_predictions']
                else:
                    X_data = st.session_state.X_train
                    y_true = st.session_state.y_train
                    y_pred = model_info['train_predictions']
                
                # Metrikler, Confusion Matrix ve Sınıflandırma Raporu için sekme
                det_tab1, det_tab2, det_tab3 = st.tabs(["Metriken", "Confusion Matrix", "Klassifikationsbericht"])
                
                with det_tab1:
                    # Detaylı metrikler
                    acc = accuracy_score(y_true, y_pred)
                    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{acc:.4f}")
                    col2.metric("Precision", f"{prec:.4f}")
                    col3.metric("Recall", f"{rec:.4f}")
                    col4.metric("F1-Score", f"{f1:.4f}")
                
                with det_tab2:
                    # Confusion Matrix
                    cm = confusion_matrix(y_true, y_pred)
                    classes = sorted(y_true.unique())
                    
                    # Normalize option
                    normalize_cm = st.checkbox("Confusion Matrix normalisieren", value=False)
                    
                    # Plot Confusion Matrix
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    if normalize_cm:
                        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                                   xticklabels=classes, yticklabels=classes)
                        plt.title(f'Normalisierte Confusion Matrix - {detailed_model}')
                    else:
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                   xticklabels=classes, yticklabels=classes)
                        plt.title(f'Confusion Matrix - {detailed_model}')
                    
                    plt.ylabel('Tatsächliche Klasse')
                    plt.xlabel('Vorhergesagte Klasse')
                    st.pyplot(fig)
                
                with det_tab3:
                    # Sınıflandırma raporu
                    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report).transpose()
                    
                    # Formatlamak
                    st.dataframe(report_df.style.format({
                        'precision': '{:.4f}',
                        'recall': '{:.4f}',
                        'f1-score': '{:.4f}',
                        'support': '{:.0f}'
                    }))
                
                # En iyi modeli önerebiliriz
                if test_or_train_detail == "Testdaten":
                    st.info(f"Hinweis: Um dieses Modell für Prediction zu verwenden, speichern Sie es bitte in der Modellierungskomponente.")
        
        # Navigation buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Zurück zur Modellierung"):
                activate_component("modeling")
                st.rerun()
        
        with col2:
            if st.button("Weiter zur Prediction", type="primary"):
                activate_component("prediction")
                st.rerun()
    
    else:
        st.warning("Es wurden noch keine Modelle trainiert. Bitte gehen Sie zurück zum Modellierungs-Abschnitt.")
        if st.button("Zurück zur Modellierung"):
            activate_component("modeling")
            st.rerun()
        
        # Button zum Neustart
        if st.button("Workflow neu starten"):
            activate_component("data_import")
            st.rerun()
            
        # Button zur Rückkehr zur Modellierung
        if st.button("Zurück zur Modellierung (weitere Modelle trainieren)"):
            activate_component("modeling")
            st.rerun()
        if st.button("Weiter zur Prediction", type="primary"):
            activate_component("prediction")
            st.rerun()
        else:
         st.warning("Bitte trainieren Sie zuerst ein Modell.")
        if st.button("Zurück zur Modellierung"):
            activate_component("modeling")
# 7. Prediction Komponente (Orange-Style Modell-Laden Workflow)
elif st.session_state.active_components["prediction"]:
    st.header("7. Prediction", divider="orange")
    
    # Initialisierung des Session States für Prediction
    if 'loaded_models' not in st.session_state:
        st.session_state.loaded_models = {}
    if 'prediction_dataset' not in st.session_state:
        st.session_state.prediction_dataset = None
    
    # Hauptbereich für Prediction teilen
    col1, col2 = st.columns([1, 2])
    
    # Linke Spalte: Orange-Style Workflow (Modelle laden, Datei laden)
    with col1:
        st.subheader("Workflow")
        
        # Orange-Style Load Model Widget
        st.write("#### 📂 Load Model")
        
        # Modell 1 laden
        with st.expander("Modell 1 laden", expanded=True):
            load_model1 = st.file_uploader("Modell 1 (.pkl)", type="pkl", key="model1_uploader")
            if load_model1 is not None:
                try:
                    import pickle
                    model1 = pickle.load(load_model1)
                    # Verwende den originalen Dateinamen statt der generischen Bezeichnung
                    model_name = load_model1.name
                    st.session_state.loaded_models[model_name] = model1
                    st.success(f"Modell erfolgreich geladen: {model_name} ({type(model1).__name__})")
                except Exception as e:
                    st.error(f"Fehler beim Laden des Modells: {str(e)}")
        
        # Modell 2 laden
        with st.expander("Modell 2 laden", expanded=True):
            load_model2 = st.file_uploader("Modell 2 (.pkl)", type="pkl", key="model2_uploader")
            if load_model2 is not None:
                try:
                    import pickle
                    model2 = pickle.load(load_model2)
                    # Verwende den originalen Dateinamen statt der generischen Bezeichnung
                    model_name = load_model2.name
                    st.session_state.loaded_models[model_name] = model2
                    st.success(f"Modell erfolgreich geladen: {model_name} ({type(model2).__name__})")
                except Exception as e:
                    st.error(f"Fehler beim Laden des Modells: {str(e)}")
        
        # Orange-Style File Widget
        st.write("#### 📊 File")
        uploaded_file = st.file_uploader("Neuen Datensatz laden (.csv)", type="csv")
        if uploaded_file is not None:
            try:
                new_data = pd.read_csv(uploaded_file)
                st.session_state.prediction_dataset = new_data
                st.success(f"Datensatz erfolgreich geladen: {new_data.shape[0]} Zeilen, {new_data.shape[1]} Spalten")
                
                # Vorschau anzeigen
                st.write("Vorschau des Datensatzes:")
                st.dataframe(new_data.head(5))
            except Exception as e:
                st.error(f"Fehler beim Laden der Datei: {str(e)}")
        
        # Orange-Style Select Columns Widget
        if st.session_state.prediction_dataset is not None:
            st.write("#### 🔍 Select Columns")
            with st.expander("Spalten auswählen", expanded=True):
                # Klassenvariable auswählen
                all_columns = st.session_state.prediction_dataset.columns.tolist()
                
                target_column = st.selectbox(
                    "Klassenvariable (Target)",
                    options=all_columns,
                    index=len(all_columns)-1 if len(all_columns) > 0 else 0
                )
                
                # Features auswählen
                feature_columns = st.multiselect(
                    "Features",
                    options=[col for col in all_columns if col != target_column],
                    default=[col for col in all_columns if col != target_column]
                )
                
                # In Session State speichern
                if st.button("Spalten übernehmen", key="select_cols_btn"):
                    st.session_state.prediction_columns = {
                        'target': target_column,
                        'features': feature_columns
                    }
                    st.success(f"Spaltenauswahl übernommen: {len(feature_columns)} Features und 1 Zielvariable")
    
    # Rechte Spalte: Predictions und Confusion Matrix
    with col2:
        # Überprüfen, ob alle Voraussetzungen erfüllt sind
        ready_for_prediction = (
            len(st.session_state.loaded_models) > 0 and 
            st.session_state.prediction_dataset is not None and
            'prediction_columns' in st.session_state
        )
        
        if ready_for_prediction:
            st.subheader("Modellvorhersagen")
            
            # Daten vorbereiten
            X_pred = st.session_state.prediction_dataset[st.session_state.prediction_columns['features']]
            y_true = st.session_state.prediction_dataset[st.session_state.prediction_columns['target']]
            
            # Modell für Vorhersage auswählen
            model_options = list(st.session_state.loaded_models.keys())
            selected_model_key = st.selectbox(
                "Modell für Vorhersage auswählen",
                options=model_options
            )
            
            selected_model = st.session_state.loaded_models[selected_model_key]
            
            # Orange-Style Predictions Widget
            st.write("#### 🔮 Predictions")
            
            with st.spinner("Berechne Vorhersagen..."):
                # Vorhersagen
                # Modeli pickle'dan çıkardığımızda yapısını kontrol et
                if isinstance(selected_model, dict):
                    # Model paketi yapısında
                    model = selected_model['model']
                    scaler = selected_model.get('scaler')
                    standardize = selected_model.get('standardize', False)
                    feature_names = selected_model.get('feature_names')
                    
                    # Aynı özellikleri kullan
                    if feature_names:
                        # Sadece gerekli sütunları seç ve doğru sırada kullan
                        missing_cols = [col for col in feature_names if col not in X_pred.columns]
                        if missing_cols:
                            st.error(f"Folgende Spalten fehlen: {', '.join(missing_cols)}")
                            st.stop()
                        
                        X_pred = X_pred[feature_names]
                    
                    # Standardizasyon uygula
                    if standardize and scaler:
                        X_pred_processed = scaler.transform(X_pred)
                        y_pred = model.predict(X_pred_processed)
                    else:
                        y_pred = model.predict(X_pred)
                else:
                    # Direkt model nesnesi
                    y_pred = selected_model.predict(X_pred)
                
                # Wahrscheinlichkeiten (falls verfügbar)
                proba_available = hasattr(selected_model, 'predict_proba')
                if proba_available:
                    y_proba = selected_model.predict_proba(X_pred)
                
                # Ergebnisse anzeigen
                results_df = X_pred.copy()
                results_df['Tatsächlich'] = y_true
                results_df['Vorhersage'] = y_pred
                results_df['Korrekt'] = results_df['Tatsächlich'] == results_df['Vorhersage']
                
                # Tab-Ansicht
                eval_tab , view_tab = st.tabs(["Evaluierung" , "Vorhersagen"])
                
                with eval_tab:
                    # Orange-Style Confusion Matrix Widget
                    st.write("#### 📊 Confusion Matrix")
                    
                    # Metriken berechnen
                    accuracy = accuracy_score(y_true, y_pred)
                    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                    
                    # Metriken anzeigen
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{accuracy:.4f}")
                    col2.metric("Precision", f"{precision:.4f}")
                    col3.metric("Recall", f"{recall:.4f}")
                    col4.metric("F1-Score", f"{f1:.4f}")
                    
                    # Confusion Matrix
                    cm = confusion_matrix(y_true, y_pred)
                    classes = sorted(pd.Series(y_true).unique())
                    
                    # Normalisierte CM-Option
                    normalize_cm = st.checkbox("Confusion Matrix normalisieren", value=False)
                    
                    # Confusion Matrix visualisieren
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    if normalize_cm:
                        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                                  xticklabels=classes, yticklabels=classes)
                        # Titel mit Modellnamen anzeigen
                        plt.title(f'Normalisierte Confusion Matrix - {selected_model_key}')
                    else:
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                  xticklabels=classes, yticklabels=classes)
                        # Titel mit Modellnamen anzeigen
                        plt.title(f'Confusion Matrix - {selected_model_key}')
                    
                    plt.ylabel('Tatsächliche Klasse')
                    plt.xlabel('Vorhergesagte Klasse')
                    st.pyplot(fig)
                    
                    # Detaillierter Klassifikationsbericht
                    st.subheader("Klassifikationsbericht")
                    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report).transpose()
                    
                    # Orange-Style formatierte Tabelle
                    st.dataframe(report_df.style.format({
                        'precision': '{:.4f}',
                        'recall': '{:.4f}',
                        'f1-score': '{:.4f}',
                        'support': '{:.0f}'
                    }))
                with view_tab:
                    # Anzeige-Optionen
                    display_options = st.radio(
                        "Anzeige filtern nach:",
                        ["Alle anzeigen", "Nur korrekte Vorhersagen", "Nur falsche Vorhersagen"],
                        horizontal=True
                    )
                    
                    # Daten filtern
                    if display_options == "Nur korrekte Vorhersagen":
                        filtered_df = results_df[results_df['Korrekt']]
                    elif display_options == "Nur falsche Vorhersagen":
                        filtered_df = results_df[~results_df['Korrekt']]
                    else:
                        filtered_df = results_df
                    
                    # Anzahl der anzuzeigenden Zeilen
                    num_rows = st.slider("Anzahl der anzuzeigenden Zeilen", 5, 100, 20)
                    
                    # Ergebnisse anzeigen
                    if len(filtered_df) > 0:
                        st.dataframe(filtered_df.head(num_rows), use_container_width=True)
                        st.info(f"Zeige {min(num_rows, len(filtered_df))} von {len(filtered_df)} Ergebnissen")
                    else:
                        st.warning(f"Keine Ergebnisse für den Filter '{display_options}'")   
        else:
            # Anleitung anzeigen, wenn nicht alles bereit ist
            st.info("#### 🔍 Anleitung für Predictions")
            st.write("""
            Um Vorhersagen wie in Orange durchzuführen, folgen Sie diesen Schritten:
            
            1. **Modelle laden**: Verwenden Sie die Funktion 'Load Model', um Ihre gespeicherten Modelle zu laden (.pkl-Format).
            2. **Datensatz laden**:  Laden Sie einen neuen Datensatz im .csv-Format hoch.
            3. **Spalten auswählen**: Spalten auswählen: Wählen Sie die Feature-Spalten und die Ziel-Spalte aus.
            4. **Vorhersagen**: Sobald alles geladen ist, werden die Vorhersagen berechnet und die Ergebnisse angezeigt
            5. **Evaluierung**: Die Confusion Matrix und andere Metriken helfen bei der Bewertung der Modellergebnisse
            """)
            
            missing_items = []
            if len(st.session_state.loaded_models) == 0:
                missing_items.append("- Mindestens ein Modell laden")
            if st.session_state.prediction_dataset is None:
                missing_items.append("- Einen neuen Datensatz laden")
            if 'prediction_columns' not in st.session_state and st.session_state.prediction_dataset is not None:
                missing_items.append("- Spalten auswählen und übernehmen")
            
            if missing_items:
                st.warning("Fehlende Elemente für Vorhersagen:")
                for item in missing_items:
                    st.markdown(item)
    # Funktionalität zum Speichern von Modellen (Save Model Widget Äquivalent)

    
    # Buttons für Navigation
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Zurück zur Modellierung"):
            activate_component("modeling")
        
            st.rerun()
           # Button zum Neustart
        if st.button("Workflow neu starten"):
            activate_component("data_import")
            st.rerun()
else:
    # Startseite
    #st.markdown("<h1 style='text-align: center;'>ML-Workflow</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Willkommen bei der ML-Workflow-App</h2>", unsafe_allow_html=True)
    #st.markdown("<h3 style='text-align: center;'>Bitte wählen Sie in der Seitenleiste einen Schritt aus, um zu beginnen.</h3>", unsafe_allow_html=True)
    st.markdown("""
<div style='text-align: center; width: 100%;'>
    <p style='font-size: 1.2em;'>Diese App führt Sie durch einen typischen maschinellen Lernablauf.</p>
    <div style='display: inline-block; text-align: left; max-width: 800px;'>
        <p style='font-size: 1em;'>1. <strong>Daten importieren</strong>: Laden Sie Ihre eigenen Daten hoch oder verwenden Sie Beispieldaten</p>
        <p style='font-size: 1em;'>2. <strong>Daten visualisieren</strong>: Untersuchen Sie Ihre Daten mit Scatter Plots und Feature-Statistiken</p>
        <p style='font-size: 1em;'>3. <strong>Datenbereinigung</strong>: Wählen Sie Features und Zielvariablen aus</p>
        <p style='font-size: 1em;'>4. <strong>Daten-Sampling</strong>: Teilen Sie Ihre Daten in Trainings- und Testdaten auf</p>
        <p style='font-size: 1em;'>5. <strong>Modellierung</strong>: Trainieren Sie kNN- oder Entscheidungsbaum-Modelle</p>
        <p style='font-size: 1em;'>6. <strong>Evaluation</strong>: Bewerten Sie die Modellleistung mit Confusion Matrix und Metriken</p>
        <p style='font-size: 1em;'>7. <strong>Prediction</strong>: Unsere trainierten Modelle auf neuen Datensätzen validieren</p>
    </div>        
    <p style="font-weight: bold; font-size: 1em; margin-top: 15px; color: #007E92;">Beginnen Sie in der Seitenleiste auf "<span style="text-decoration: underline;">1. Daten importieren</span>" klicken.</p>
</div>
""", unsafe_allow_html=True)

    # Kurze Beschreibung der App
 # st.subheader("Über diese App")
  #  st.write("""
                                                                   
    
    # Beispiel-Workflow als Bild darstellen (optional)
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
     st.image('images/logo.png', use_container_width=True)
     #        caption="Beispiel eines ML-Workflows in Orange (Platzhalterbild)")    
     