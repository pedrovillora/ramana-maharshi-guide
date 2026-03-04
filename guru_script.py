from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import re
import os

# -----------------------------
# Configuración del backend
# -----------------------------
app = FastAPI()

# Permitir que Flutter web acceda a la API
origins = ["*"]  # Para pruebas. En producción, reemplazar por tu dominio

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Configuración Nebius/OpenAI
# -----------------------------
client = OpenAI(
    api_key="TU_API_KEY_AQUI",
    base_url="https://api.studio.nebius.ai/v1"
)

# -----------------------------
# Cargar base vectorial
# -----------------------------
with open("base_vectorial.json", "r", encoding="utf-8") as f:
    base = json.load(f)

# -----------------------------
# Cargar estilos de maestros
# -----------------------------
ruta_maestros = os.path.join(os.path.dirname(__file__), "estilos_maestros.json")
with open(ruta_maestros, "r", encoding="utf-8") as f:
    maestros = json.load(f)

# Crear un diccionario rápido para buscar por nombre
maestros_dict = {m["nombre"].lower(): m for m in maestros}

# -----------------------------
# Modelos de datos
# -----------------------------
class Pregunta(BaseModel):
    pregunta: str
    nombre_maestro: str

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
    nombre_maestro = p.nombre_maestro.strip().lower()
    
    if nombre_maestro not in maestros_dict:
        raise HTTPException(status_code=400, detail=f"Maestro '{p.nombre_maestro}' no encontrado.")
    
    maestro = maestros_dict[nombre_maestro]

    # -------------------------
    # Generar embedding de la pregunta
    # -------------------------
    response = client.embeddings.create(
        model="BAAI/bge-multilingual-gemma2",
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
Eres {maestro['nombre']}, un maestro Advaita realizado.
{maestro['descripcion']}
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
        model="deepseek-ai/DeepSeek-V3.2",
        messages=[
            {"role": "system", "content": f"Eres {maestro['nombre']}."},
            {"role": "user", "content": prompt}
        ],
        temperature=maestro.get("temperatura", 0.5),
        max_tokens=280
    )

    respuesta_final = chat_response.choices[0].message.content
    respuesta_final = limpiar_texto(respuesta_final)

    return {"respuesta": respuesta_final}
