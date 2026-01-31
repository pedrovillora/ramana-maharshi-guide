import streamlit as st
import google.generativeai as genai

# Título de la App
st.set_page_config(page_title="Ramana AI", page_icon="🧘")
st.title("🧘 Ramana Maharshi AI Guide")

# BUSCAR LA CLAVE (Intentamos varias formas)
api_key = None

if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["AIzaSyDwNLOvF5tpnxF7wI2JMVd27a-61FlfPzI"]
elif "google_api_key" in st.secrets:
    api_key = st.secrets["google_api_key"]

if not api_key:
    st.warning("⚠️ No se encontró la API Key en Secrets.")
    st.info("Asegúrate de que en Settings > Secrets diga: GOOGLE_API_KEY = 'tu_clave'")
    st.stop()

# CONFIGURACIÓN DEL MODELO
genai.configure(api_key=api_key)

instruction = (
    "Eres un sabio basado en las enseñanzas de Ramana Maharshi. "
    "Responde de forma breve y pacífica. Tu mensaje central es que la felicidad "
    "está en el interior y se alcanza mediante la pregunta '¿Quién soy yo?'."
)

model = genai.GenerativeModel('gemini-1.5-flash', system_instruction=instruction)

# CHAT INTERFACE
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("¿Quién soy yo?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        try:
            response = model.generate_content(prompt)
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"Error: {e}")
