import streamlit as st
import google.generativeai as genai

# CONFIGURACIÓN
st.set_page_config(page_title="Ramana AI", page_icon="🧘")
st.title("🧘 Ramana Maharshi AI Guide")

# Intentar obtener la API Key de Secrets o de texto plano (para pruebas)
try:
    api_key = st.secrets["AIzaSyDwNLOvF5tpnxF7wI2JMVd27a-61FlfPzI"]
except:
    st.error("⚠️ No se encontró la API Key en 'Advanced Settings > Secrets'.")
    st.stop()

genai.configure(api_key=api_key)

# Aquí definimos el comportamiento de Ramana Maharshi
instruction = (
    "Actúa como un sabio que sigue las enseñanzas de Ramana Maharshi. "
    "Tus respuestas deben ser breves, pacíficas y directas al Sí Mismo. "
    "Si te preguntan algo complejo, redirige al usuario a la pregunta: '¿Quién soy yo?'."
)

# Usamos 'gemini-1.5-flash' que es el más estable
model = genai.GenerativeModel(
    model_name='gemini-1.5-flash',
    system_instruction=instruction
)

# CHAT
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
            # Generar respuesta
            response = model.generate_content(prompt)
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"Hubo un problema con la respuesta: {e}")
