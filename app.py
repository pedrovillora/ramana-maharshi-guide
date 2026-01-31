import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="Ramana AI", page_icon="🧘")
st.title("🧘 Ramana Maharshi AI Guide")

# Cargar API Key
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
except:
    st.error("Configura GOOGLE_API_KEY en Secrets.")
    st.stop()

# Configurar el modelo (Sin system_instruction por ahora para evitar el 404)
model = genai.GenerativeModel('models/gemini-1.5-flash')

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
        # Añadimos la instrucción directamente en el mensaje para mayor compatibilidad
        full_prompt = f"Instrucción: Responde como Ramana Maharshi. Pregunta: {prompt}"
        try:
            response = model.generate_content(full_prompt)
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"Error de conexión: {e}")
