from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import re

# -----------------------------
# Configuración del backend
# -----------------------------
app = FastAPI()

# Permitir que Flutter web acceda a la API
origins = [
    "*",  # Para pruebas: permite todos los orígenes. En producción, reemplazar por tu dominio
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configuración Nebius/OpenAI
client = OpenAI(api_key="v1.CmMKHHN0YXRpY2tleS1lMDBnYXRxam4wbTcxMG5rdGUSIXNlcnZpY2VhY2NvdW50LWUwMGJtZzIwcjhxa2p6amcxczIMCObpm80GEITSn9ICOgsI5-yzmAcQgO3mQUACWgNlMDA.AAAAAAAAAAHqf7Ja1EvkLDRagvAlDPfZvbO7b78TNU7l_nbYkQhzILxI8IQAB0dq-AduKIfvxeK2_sUtbrYzrSw235m1yScG", base_url="https://api.studio.nebius.ai/v1")

# -----------------------------
# Cargar base vectorial
# -----------------------------
with open("base_vectorial.json", "r", encoding="utf-8") as f:
    base = json.load(f)

# -----------------------------
# Modelos de datos
# -----------------------------
class Pregunta(BaseModel):
    pregunta: str

# -----------------------------
# Funciones auxiliares
# -----------------------------
def limpiar_texto(texto: str) -> str:
    patrones = [
        r"Según el texto[,:\s]*",
        r"Como se explica en el texto[,:\s]*",
        r"De acuerdo al texto[,:\s]*",
        r"El texto dice que[,:\s]*",
        r"<think>.*?</think>"
    ]
    for p in patrones:
        texto = re.sub(p, "", texto, flags=re.IGNORECASE | re.DOTALL)
    return texto.strip()

# -----------------------------
# Endpoint principal
# -----------------------------
@app.post("/preguntar")
def preguntar(p: Pregunta):
    # -------------------------
    # Generar embedding de la pregunta
    # -------------------------
    response = client.embeddings.create(
        model="BAAI/bge-multilingual-gemma2",  # Modelo de embeddings
        input=p.pregunta
    )
    embedding_pregunta = response.data[0].embedding

    # -------------------------
    # Buscar los top 3 fragmentos
    # -------------------------
    scores = []
    for item in base:
        score = cosine_similarity([embedding_pregunta], [item["embedding"]])[0][0]
        scores.append((score, item))
    scores.sort(reverse=True, key=lambda x: x[0])
    top_fragmentos = [limpiar_texto(s[1]["texto"]) for s in scores[:3]]
    contexto = "\n\n".join(top_fragmentos)

    # -------------------------
    # Construir prompt para el LLM
    # -------------------------
    prompt = f"""
Eres un maestro Advaita realizado.
Habla con claridad y compasión, directo a la experiencia.
Evita "según el texto" y razonamiento interno.
Responde intermedio: profundo pero con párrafos cortos.
Contexto:
{contexto}
Pregunta:
{p.pregunta}
Responde con presencia y guía hacia la autoindagación.
"""

    # -------------------------
    # Generar respuesta del LLM
    # -------------------------
    chat_response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3.2",  # Modelo de LLM
        messages=[
            {"role": "system", "content": "Eres un maestro Advaita."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
        max_tokens=280
    )

    respuesta_final = chat_response.choices[0].message.content
    respuesta_final = limpiar_texto(respuesta_final)

    return {"respuesta": respuesta_final}


