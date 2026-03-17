import streamlit as st
import google.generativeai as genai
import pandas as pd
import re
import os

# --- 1. CONFIGURACIÓN DE PÁGINA Y ESTILO (FACHERO) ---
st.set_page_config(page_title="Netflix Intelligence Agent", page_icon="🎬", layout="wide")

# CSS personalizado para tarjetas sutiles y diseño limpio
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .css-1r6slb0 { border-radius: 10px; border: 1px solid #eee; padding: 15px; background-color: white; }
    .cl-card { padding: 20px; background-color: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.02); margin-bottom: 15px; border: 1px solid #f0f0f0; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. GESTIÓN DE SECRETOS / API KEY ---
# Prioridad: st.secrets (Streamlit Cloud) -> Sidebar (Manual)
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
    """Carga el dataset de Netflix desde el repositorio."""
    try:
        # Intenta cargar el archivo. Asegúrate de que se llame exactamente así en GitHub
        df = pd.read_csv("utils/netflix_titles.csv")
        return df
    except FileNotFoundError:
        st.error("❌ Archivo 'netflix_titles.csv' no encontrado en el repositorio.")
        return pd.DataFrame() # Retorna un DataFrame vacío si falla

df_netflix = load_data()

# --- 4. TÍTULO Y MÉTRICAS DE INTELIGENCIA ---
st.title("🎬 Netflix Content Intelligence Agent")
st.subheader("Análisis y Consultas Avanzadas sobre el Catálogo Global")

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

    with col_chat:
        st.markdown("#### 💬 Consulta Exclusiva del Catálogo")
        st.caption("Ejemplos: 'Dame un top 10 de películas de acción de EE.UU.' o '¿Qué géneros son populares en Japón?'")
        
        # Mostrar mensajes previos
        for message in st.session_state.netflix_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Escribe tu pregunta sobre películas o series aquí..."):
            st.session_state.netflix_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analizando el catálogo..."):
                    # --- BUCLE RE-ACT DE AGENTE ---
                    # 1. Razonamiento: ¿Es una consulta de datos o general?
                    reasoning_prompt = f"""
                    Eres un Analista de Contenido de Netflix. Tu ÚNICA fuente de información es este dataset:
                    COLUMNAS: {df_netflix.columns.tolist()}
                    
                    INSTRUCCIONES:
                    1. Si el usuario pide una lista, top o conteo de títulos, responde EXACTAMENTE: 
                       DATOS: [cantidad_solicitada], [país_o_None], [género_o_None], [tipo_Movie/TV_Show/None]
                    2. Si es una pregunta sobre el dataset (ej: '¿qué columnas tienes?'), responde normal.
                    3. Si la pregunta NO es sobre Netflix, películas o series, responde: 
                       "Lo siento, mi experiencia se limita exclusivamente al análisis del catálogo de Netflix. No puedo responder preguntas fuera de ese tema."
                    
                    PREGUNTA DEL USUARIO: "{prompt}"
                    """
                    
                    try:
                        reasoning_res = model.generate_content(reasoning_prompt)
                        
                        # 2. Acción: Procesamiento de Datos con Python
                        if "DATOS:" in reasoning_res.text:
                            # Extraer entidades usando Regex (Día 3)
                            params = re.findall(r'\[(.*?)\]', reasoning_res.text)[0].split(',')
                            qty = int(params[0].strip()) if params[0].strip().isdigit() else 10
                            country = params[1].strip() if "None" not in params[1] else None
                            genre = params[2].strip() if "None" not in params[2] else None
                            content_type = params[3].strip() if "None" not in params[3] else None
                            
                            # Filtrado dinámico con Pandas
                            df_temp = df_netflix.copy()
                            if country:
                                df_temp = df_temp[df_temp['country'].str.contains(country, na=False, case=False)]
                            if genre:
                                df_temp = df_temp[df_temp['listed_in'].str.contains(genre, na=False, case=False)]
                            if content_type:
                                df_temp = df_temp[df_temp['type'].str.contains(content_type, na=False, case=False)]
                            
                            results = df_temp[['title', 'release_year', 'country', 'listed_in']].head(qty)
                            
                            if results.empty:
                                full_answer = f"No encontré títulos que coincidan con los criterios: {country or ''} {genre or ''}."
                            else:
                                st.write(f"Aquí tienes los {len(results)} títulos solicitados:")
                                st.table(results) # Visualización tabular limpia
                                full_answer = "Espero que esta lista te sea útil."
                        else:
                            full_answer = reasoning_res.text
                    
                    except Exception as e:
                        full_answer = f"Hubo un error al procesar tu solicitud: {str(e)}"

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