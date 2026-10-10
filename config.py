"""Configuración central: parámetros del pipeline RAG, modelos y secretos.

Las claves NUNCA se escriben en el código ni en el repositorio:
- En local se leen del archivo .env (ignorado por git).
- En la nube (Streamlit Community Cloud, etc.) se leen de los Secrets de la plataforma.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # Carga variables desde .env (solo en local)


def get_secret(nombre, por_defecto=None):
    """Busca primero en variables de entorno y luego en st.secrets (Streamlit Cloud)."""
    valor = os.getenv(nombre)
    if valor:
        return valor
    try:
        import streamlit as st
        return st.secrets.get(nombre, por_defecto)
    except Exception:
        return por_defecto


# --- Credenciales y modelo generativo (proveedor intercambiable: Groq o Gemini) ---
GROQ_API_KEY = get_secret("GROQ_API_KEY")
GENAI_API_KEY = get_secret("GENAI_API_KEY")

# Si no se indica LLM_PROVIDER, se usa Groq cuando hay GROQ_API_KEY y Gemini en caso contrario
LLM_PROVIDER = (get_secret("LLM_PROVIDER") or ("groq" if GROQ_API_KEY else "gemini")).lower()
GROQ_MODEL = get_secret("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = get_secret("GEMINI_MODEL", "gemini-3-flash-preview")

LLM_MODEL = GROQ_MODEL if LLM_PROVIDER == "groq" else GEMINI_MODEL
API_KEY = GROQ_API_KEY if LLM_PROVIDER == "groq" else GENAI_API_KEY
NOMBRE_VARIABLE_CLAVE = "GROQ_API_KEY" if LLM_PROVIDER == "groq" else "GENAI_API_KEY"

# --- Embeddings (se ejecutan LOCALMENTE en CPU: el corpus no sale de tu equipo/servidor) ---
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# --- Corpus y base vectorial ---
DOCS_DIR = "docs"
CHROMA_BASE_DIR = "chroma_db"
COLLECTION_NAME = "tutor_ia"

# --- Parámetros del pipeline (línea base; se modifican en la iteración de mejora) ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4
LIMPIAR_ENCABEZADOS = True  # quita encabezados/pies de página repetidos en cada página

# --- Conversación ---
MAX_HISTORIAL_TURNOS = 6  # nº de turnos (pregunta+respuesta) que se reenvían al modelo

# Frase fija que el asistente debe devolver cuando la respuesta no está en el corpus
FRASE_SIN_INFO = (
    "No tengo información suficiente en el material de estudio "
    "para responder esto con precisión."
)
