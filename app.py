%%writefile app.py
import streamlit as st
import google.generativeai as genai

# CONFIGURACIÓN (Pega aquí tu clave directamente para esta prueba)
genai.configure(api_key="AIzaSyDwNLOvF5tpnxF7wI2JMVd27a-61FlfPzI")
model = genai.GenerativeModel('gemini-1.5-flash')

st.set_page_config(page_title="Ramana AI", page_icon="🧘")
st.title("🧘 Ramana Maharshi AI Guide")

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
        response = model.generate_content(prompt)
        st.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})
