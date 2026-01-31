import streamlit as st
import google.generativeai as genai

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="Ramana Maharshi AI", 
    page_icon="🧘",
    layout="centered"
)

# Estética simple y espiritual
st.title("🧘 Ramana Maharshi AI Guide")
st.markdown("*La respuesta a cada pregunta es: ¿Quién soy yo?*")
st.divider()

# 2. CARGA SEGURA DE API KEY
try:
    # Busca la clave en Settings > Secrets
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
except Exception:
    st.error("⚠️ Error: No se encontró la 'GOOGLE_API_KEY' en los Secrets de Streamlit.")
    st.info("Ve a 'Manage App' > 'Settings' > 'Secrets' y añade: GOOGLE_API_KEY = 'tu_clave'")
    st.stop()

# 3. CONFIGURACIÓN DEL MODELO (Versión 2026)
# Usamos gemini-2.0-flash por su velocidad y disponibilidad
model = genai.GenerativeModel('gemini-2.0-flash-lite')

# Instrucción de personalidad (System Prompt)
SYSTEM_PROMPT = (
    "Eres un guía espiritual basado exclusivamente en las enseñanzas de Ramana Maharshi. "
    "Tus respuestas deben ser extremadamente breves, llenas de paz y silencio. "
    "Tu objetivo no es dar información académica, sino dirigir la mente del usuario "
    "hacia su origen a través de la auto-indagación (Atma-Vichara). "
    "Si el usuario está confundido, recuérdale investigar quién es el que tiene esa duda."
)

# 4. GESTIÓN DEL HISTORIAL DE CHAT
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes previos
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. ENTRADA DE PREGUNTAS
if prompt := st.chat_input("Consulta al Silencio..."):
    # Guardar y mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generar respuesta de la IA
    with st.chat_message("assistant"):
        try:
            # Combinamos la instrucción con la pregunta para asegurar la personalidad
            full_query = f"{SYSTEM_PROMPT}\n\nUsuario pregunta: {prompt}"
            
            response = model.generate_content(full_query)
            respuesta_texto = response.text
            
            st.markdown(respuesta_texto)
            st.session_state.messages.append({"role": "assistant", "content": respuesta_texto})
            
        except Exception as e:
            st.error(f"El flujo de sabiduría se ha interrumpido: {e}")
