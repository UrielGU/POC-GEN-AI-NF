import streamlit as st
import google.generativeai as genai
import pandas as pd
import plotly.express as px
import re
import os

# --- 1. CONFIGURACIÓN DE PÁGINA ESTILO FACHERO) ---
st.set_page_config(page_title="Netflix Intelligence Agent", page_icon="🎬", layout="wide")

# CSS
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .css-1r6slb0 { border-radius: 10px; border: 1px solid #eee; padding: 15px; background-color: white; }
    .cl-card { padding: 20px; background-color: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.02); margin-bottom: 15px; border: 1px solid #f0f0f0; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. API KEY ---
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    with st.sidebar:
        st.warning("🔑 Secret 'GEMINI_API_KEY' no encontrado.")
        api_key = st.text_input("Ingresa tu Gemini API Key manual:", type="password")
        st.info("Obtén tu llave en [Google AI Studio](https://aistudio.google.com/)")

# --- 3. CARGA Y PREPARACIÓN DE DATOS (BIG DATA) ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("utils/netflix_titles.csv", encoding='latin1')
        # Limpieza básica: Rellenar nulos para que el Agente no se confunda
        df['director'] = df['director'].fillna('No especificado')
        df['cast'] = df['cast'].fillna('No disponible')
        df['country'] = df['country'].fillna('Global')
        return df
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return pd.DataFrame()

df_netflix = load_data()

# --- 4. TÍTULO Y MÉTRICAS DE INTELIGENCIA ---
st.title("🎬 POC: Netflix Content Agent AI")
st.subheader("Análisis y Consultas Avanzadas sobre el Catálogo Global de Netflix")

if not df_netflix.empty:
    # Fila de métricas para un look profesional
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Títulos", f"{len(df_netflix):,}")
    m2.metric("Películas", f"{len(df_netflix[df_netflix['type'] == 'Movie']):,}")
    m3.metric("Series (TV Shows)", f"{len(df_netflix[df_netflix['type'] == 'TV Show']):,}")
    m4.metric("Países Productores", f"{df_netflix['country'].nunique():,}")
    st.divider()

# --- 5. LÓGICA DEL AGENTE Y CHAT ---
if api_key and not df_netflix.empty:
    genai.configure(api_key=api_key)
    # Usamos Gemini 2.5 Flash, que es excelente para análisis de datos y extracción de entidades
    model = genai.GenerativeModel('models/gemini-2.5-flash')

    # Inicializar historial de chat
    if "netflix_messages" not in st.session_state:
        st.session_state.netflix_messages = []

    # Layout de dos columnas: Chat y Visualización
    col_chat, col_viz = st.columns([2, 1])

    # --- 5. LÓGICA DEL AGENTE MEJORADA ---
    with col_chat:
        # ... (historial y chat_input igual) ...

        if prompt := st.chat_input("Escribe tu pregunta..."):
            # ... (guardar mensaje igual) ...

            with st.chat_message("assistant"):
                with st.spinner("Consultando inteligencia de contenido..."):
                    
                    # PROMPT FLEXIBLE
                    reasoning_prompt = f"""
                    Eres un Analista de Contenido de Netflix Experto.
                    DATASET_INFO: Tienes un catálogo con {len(df_netflix)} títulos.
                    COLUMNAS: {df_netflix.columns.tolist()}
                    
                    INSTRUCCIONES:
                    1. Si el usuario pide un TOP, LISTA o CONTEO, responde con este formato:
                    ACCION: BUSCAR | CANTIDAD: [n] | PAIS: [pais] | GENERO: [genero] | TIPO: [Movie/TV Show]
                    
                    2. Si el usuario pide RECOMENDACIONES basadas en un título, usa tu conocimiento para identificar el género de ese título y luego responde:
                    ACCION: RECOMENDAR | TITULO_REF: [titulo]
                    
                    3. Si pregunta sobre TENDENCIAS o POPULARIDAD, usa tu conocimiento general para responder de forma fachera.
                    
                    PREGUNTA: "{prompt}"
                    """
                    
                    try:
                        res_obj = model.generate_content(reasoning_prompt)
                        res_text = res_obj.text

                        # LÓGICA DE ACCIÓN DINÁMICA
                        if "ACCION: BUSCAR" in res_text:
                            # Extraer datos de forma segura (sin que truene si falta uno)
                            qty = int(re.search(r'CANTIDAD: (\d+)', res_text).group(1)) if re.search(r'CANTIDAD: (\d+)', res_text) else 5
                            country = re.search(r'PAIS: \[(.*?)\]', res_text).group(1) if "PAIS: [None]" not in res_text else None
                            
                            df_temp = df_netflix.copy()
                            if country:
                                df_temp = df_temp[df_temp['country'].str.contains(country, na=False, case=False)]
                            
                            results = df_temp.head(qty)
                            
                            if not results.empty:
                                st.table(results[['title', 'release_year', 'listed_in']])
                                full_answer = f"Basado en nuestro dataset, aquí tienes los títulos más relevantes para {country or 'el mundo'}."
                            else:
                                full_answer = "No encontré datos específicos en el CSV, pero según las tendencias generales..."
                        
                        elif "ACCION: RECOMENDAR" in res_text:
                            # Aquí el agente usa su "cerebro" para recomendar
                            final_res = model.generate_content(f"El usuario vio {prompt}. Recomienda 3 series similares que estén en Netflix.")
                            full_answer = final_res.text
                            
                        else:
                            full_answer = res_text

                    except Exception as e:
                        # Si algo falla, que Gemini responda normal para no dejar al usuario colgado
                        backup_res = model.generate_content(prompt)
                        full_answer = backup_res.text

                    st.markdown(full_answer)
                    st.session_state.netflix_messages.append({"role": "assistant", "content": full_answer})
    # Columnas de Visualización de Datos (El toque "Fachero" y Profesional)
    with col_viz:
        st.markdown("#### 📊 Insights Rápidos")
        
        with st.container(border=True):
            st.markdown("**Distribución de Contenido**")
            type_counts = df_netflix['type'].value_counts()
            st.plotly_chart({
                "data": [{"labels": type_counts.index, "values": type_counts.values, "type": "pie", "hole": .4}],
                "layout": {"showlegend": False, "height": 300, "margin": {"l": 10, "r": 10, "b": 10, "t": 10}}
            }, use_container_width=True)

        with st.container(border=True):
            st.markdown("**Top 5 Géneros**")
            # Separar géneros múltiples y contar
            genres_series = df_netflix['listed_in'].str.split(', ').explode()
            top_genres = genres_series.value_counts().head(5)
            st.bar_chart(top_genres)

else:
    st.error("🔑 Falta la API Key o el Dataset. Configura la llave en los Secrets de Streamlit y asegúrate de que 'netflix_titles.csv' esté en tu repositorio de GitHub.")