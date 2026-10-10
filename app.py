"""Interfaz de chat del Tutor Académico (RAG) con Streamlit.

Ejecutar en local:   streamlit run app.py
"""
import streamlit as st

import config
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="Tutor IA (RAG)", page_icon="📚", layout="centered")


@st.cache_resource(show_spinner="Cargando modelo de embeddings e índice vectorial...")
def cargar_pipeline():
    # Se crea una sola vez por servidor: carga embeddings locales y el índice Chroma
    return RAGPipeline()


def mostrar_fuentes(fuentes):
    with st.expander(f"📎 Fuentes consultadas ({len(fuentes)})"):
        for i, f in enumerate(fuentes, 1):
            st.markdown(
                f"**{i}. {f['fuente']}** · pág. {f['pagina']} · "
                f"fragmento #{f['chunk_id']} · similitud {f['score']:.2f}"
            )
            st.caption(f["texto"])


st.title("📚 Tutor Académico — Creación de Aplicaciones con IA")
st.caption(
    "Asistente RAG: responde únicamente con el material de estudio indexado "
    "y cita las fuentes de cada respuesta."
)

if not config.API_KEY:
    st.error(
        f"No se encontró {config.NOMBRE_VARIABLE_CLAVE}. Configúrala en el archivo .env (local) "
        "o en los Secrets de la plataforma de despliegue."
    )
    st.stop()

try:
    pipeline = cargar_pipeline()
except Exception as e:
    st.error(f"No se pudo inicializar el sistema: {e}")
    st.stop()

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.header("⚙️ Sistema")
    info = pipeline.info()
    st.markdown(
        f"- **LLM:** `{info['modelo_llm']}` ({info['proveedor_llm']})\n"
        f"- **Embeddings (local):** `{info['modelo_embeddings']}`\n"
        f"- **Chunk:** {info['chunk_size']} car. / solape {info['chunk_overlap']}\n"
        f"- **top_k:** {info['top_k']}\n"
        f"- **Fragmentos indexados:** {info['fragmentos_indexados']}"
    )
    st.divider()
    if st.button("🗑️ Nueva conversación", use_container_width=True):
        st.session_state.mensajes = []
        st.rerun()
    st.caption(
        "Privacidad: el corpus y el índice vectorial permanecen en este servidor. "
        "Al modelo solo se le envían los fragmentos recuperados para cada pregunta."
    )

# ------------------------------------------------------------------ historial en pantalla
for m in st.session_state.mensajes:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            if m.get("fuentes"):
                mostrar_fuentes(m["fuentes"])
            elif m.get("sin_info"):
                st.info("La información no está en el corpus, por eso no se muestran fuentes.")

# ------------------------------------------------------------------ nuevo turno
pregunta = st.chat_input("Escribe tu pregunta sobre el material de estudio...")

if pregunta:
    # El historial se toma ANTES de agregar el turno actual
    historial = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.mensajes
    ]

    with st.chat_message("user"):
        st.markdown(pregunta)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en el material de estudio..."):
            try:
                res = pipeline.responder(pregunta, historial)
            except Exception as e:
                st.error(f"Error al procesar la solicitud: {e}")
                st.stop()

        st.markdown(res["respuesta"])

        if res["pregunta_independiente"].strip() != pregunta.strip():
            st.caption(f"🔎 Búsqueda realizada con: «{res['pregunta_independiente']}»")

        fuentes = [] if res["sin_informacion"] else res["fuentes"]
        if fuentes:
            mostrar_fuentes(fuentes)
        else:
            st.info("La información no está en el corpus, por eso no se muestran fuentes.")

    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    st.session_state.mensajes.append(
        {
            "role": "assistant",
            "content": res["respuesta"],
            "fuentes": fuentes,
            "sin_info": res["sin_informacion"],
        }
    )
