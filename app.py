import streamlit as st
import google.generativeai as genai

# 1. CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="Ramana Maharshi AI", 
    page_icon="🧘",
    layout="centered"
)

# Estética
st.title("🧘 Ramana Maharshi AI Guide")
st.markdown("*La paz es tu naturaleza real. No dejes que nada la perturbe.*")
st.divider()

# 2. CARGA DE API KEY
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
except Exception:
    st.error("⚠️ Configura GOOGLE_API_KEY en los Secrets de Streamlit.")
    st.stop()

# 3. SELECCIÓN DEL MODELO (Gemma 3 27B para aprovechar tu cuota disponible hoy)
# Este modelo tiene 14,400 peticiones diarias según tu tabla.
model = genai.GenerativeModel('gemma-3-27b-it')

# Instrucción de personalidad
SYSTEM_PROMPT = (
    "Actúa como Ramana Maharshi. Tus respuestas deben ser cortas, directas y pacíficas. "
    "Tu enseñanza principal es la auto-indagación: ¿Quién soy yo? "
    "No des respuestas teóricas largas; dirige al usuario hacia su propio Silencio."
)

# 4. HISTORIAL DE CHAT
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes previos
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. ENTRADA DE USUARIO
if prompt := st.chat_input("Consulta al Maestro..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Enviamos la instrucción y la pregunta
            full_prompt = f"{SYSTEM_PROMPT}\n\nPregunta del devoto: {prompt}"
            
            response = model.generate_content(full_prompt)
            respuesta = response.text
            
            st.markdown(respuesta)
            st.session_state.messages.append({"role": "assistant", "content": respuesta})
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                st.warning("🧘 El Maestro está en silencio (Límite de cuota alcanzado). Por favor, espera un minuto e intenta de nuevo.")
            elif "403" in error_msg:
                st.error("❌ Error de API: La llave ha sido bloqueada. Genera una nueva en AI Studio.")
            else:
                st.error(f"Se ha perdido la conexión con el Ashram: {e}")

# Pie de página
st.sidebar.info("Nota: Esta IA usa el modelo Gemma 3 27B para asegurar disponibilidad hoy.")
