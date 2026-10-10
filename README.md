# Tutor Académico Personalizado (RAG) — Creación de Aplicaciones con IA

Asistente conversacional que responde preguntas sobre el material de la asignatura
**Creación de Aplicaciones con IA** usando una arquitectura **RAG** (Retrieval-Augmented
Generation). Responde únicamente con la información del corpus, cita el documento y fragmento
que respalda cada respuesta y reconoce cuándo la información no está disponible.

**App desplegada:** `https://TU-APP.streamlit.app` *(completar tras el despliegue)*

## Privacidad y control de los datos

- Los **embeddings se calculan localmente** (CPU) y el **índice vectorial (Chroma) vive en el
  propio servidor**: el corpus completo nunca se envía a ningún servicio externo.
- Al LLM (Groq o Gemini, configurable) solo se le envían, en cada consulta, los **`top_k` fragmentos recuperados**
  más la pregunta y el historial reciente; nunca el corpus completo.
- Las claves se gestionan con variables de entorno (`.env` en local) o Secrets de la plataforma
  de despliegue. **Nunca** se incluyen en el repositorio.
- Nota: revisa los términos de uso de datos del plan de la API que utilices (en el nivel gratuito
  de algunas APIs el proveedor puede usar los prompts para mejorar sus productos). Si el corpus es
  sensible, usa un plan/proveedor con política de no entrenamiento.

## Arquitectura

```
docs/ (PDF, DOCX, TXT, HTML)
   │  ingest.py
   ▼
Extracción de texto → Chunking → Embeddings locales → Chroma (persistente, coseno)
                                                            │
Pregunta ─► Reformulación (si hay historial) ─► Búsqueda top_k ┘
                                                            ▼
              Prompt aumentado (<contexto> + <pregunta> + historial + few-shot)
                                                            ▼
                               LLM ─► Respuesta + fuentes citadas ─► Chat (Streamlit)
```

| Archivo | Función |
|---|---|
| `config.py` | Parámetros del pipeline, modelos y lectura segura de secretos |
| `ingest.py` | Ingesta multiformato, chunking, embeddings e índice Chroma persistente |
| `rag_pipeline.py` | Recuperación, reformulación de seguimiento, System Prompt, Few-Shot y generación |
| `llm.py` | Acceso al LLM con proveedor intercambiable (Groq o Gemini) |
| `app.py` | Interfaz de chat (Streamlit) con historial y fuentes |
| `evaluate_ragas.py` | Evaluación con Ragas, control de alucinaciones y comparación antes/después |
| `eval_dataset.json` | 20 preguntas con ground truth (16 en el corpus + 4 fuera del corpus) |
| `avance1/` | Código del Avance 1 (prompt + few-shot con recuperador por palabras clave) |

## Decisiones técnicas

### 1. Selección de documentos
Corpus: apuntes de la asignatura *Creación de Aplicaciones con IA* (`docs/knowledge_base.txt`,
15 temas: LLMs, Prompt Engineering, Few-Shot, delimitadores, RAG, embeddings, agentes,
function calling, fine-tuning, alucinaciones, temperatura, evaluación, multimodalidad,
despliegue y ética).
**Pertinencia:** es el material que un estudiante de la asignatura consultaría; es acotado,
verificable (permite construir un ground truth fiable) y coherente con el dominio del asistente
(Tutor Académico).
*(Si añades PDFs/guías de clase a `docs/`, lístalos aquí con una frase de justificación.)*

### 2. Ingesta
`PyPDFLoader` (PDF, conserva nº de página), `TextLoader` (TXT), `Docx2txtLoader` (DOCX) y
`BSHTMLLoader` (HTML). Cada sección conserva solo metadatos simples (`source`, `page`) para
poder citarla.

**Limpieza de encabezados y pies de página:** muchos PDFs repiten en cada página una plantilla
(dirección, teléfono, NIT…) que se cuela en los fragmentos y diluye la similitud. Se eliminan las
líneas de ≥25 caracteres que aparecen (ignorando dígitos) en ≥80 % de las páginas del documento.
El largo mínimo protege líneas cortas legítimas que se repiten, como los títulos "Artículo N:" de
un reglamento. Se puede desactivar con `LIMPIAR_ENCABEZADOS = False` en `config.py`.

### 3. Chunking
`RecursiveCharacterTextSplitter` con separadores jerárquicos `["\n\n", "\n", ". ", " ", ""]`:
corta primero por párrafo y solo baja de nivel si el fragmento es demasiado grande, respetando
los límites naturales del texto.
- **Línea base:** `chunk_size=500`, `chunk_overlap=50` (≈10 %). Fragmentos pequeños y precisos,
  compatibles con el límite de entrada del modelo de embeddings.
- **Iteración de mejora:** ver sección *Evaluación*.
- Cada fragmento recibe un `chunk_id` por documento para citar "documento + fragmento #n".

### 4. Vectorización
`paraphrase-multilingual-MiniLM-L12-v2` (Sentence Transformers, 384 dimensiones):
multilingüe (buen desempeño en español), ligero (~120 MB), corre en CPU y **local**, lo que
mantiene el corpus bajo nuestro control y evita costos por embeddings. Vectores normalizados.

### 5. Base de datos vectorial
**ChromaDB** persistente en `chroma_db/cs{chunk}_ov{overlap}/` con métrica coseno. Un directorio
por configuración permite comparar iteraciones sin pisar el índice. Metadatos almacenados:
`source`, `page`, `chunk_id` (más el texto del fragmento), usados para mostrar las fuentes.

### 6. Recuperación
Búsqueda por **similitud coseno** con `top_k` fragmentos (línea base `top_k=4`). Para preguntas de
seguimiento ("¿y para qué sirve?") se **reformula primero la pregunta** como una consulta
autónoma usando el historial, y esa consulta es la que se vectoriza.

### 7. Generación (refinamiento del Avance 1)
- **LLM:** proveedor intercambiable por `LLM_PROVIDER` (Groq `openai/gpt-oss-120b` por defecto si
  hay `GROQ_API_KEY`; Gemini en caso contrario), `temperature=0.2`.
- **System Prompt:** rol, tono, reglas numeradas, formato de salida (Explicación / Ejemplo / Fuente).
  Novedades: citas obligatorias con archivo y página, frase fija de rechazo, instrucción de tratar
  el contexto como *datos* (defensa ante prompt injection) y manejo de preguntas de seguimiento.
- **Few-Shot:** 2 ejemplos precargados: uno con respuesta en el contexto y otro **sin respuesta**
  (rechazo), para reforzar el control de alucinaciones.
- **Delimitadores:** etiquetas XML `<contexto>`, `<fragmento fuente=".." pagina="..">`, `<pregunta>`.
- **Conversación:** se reenvían los últimos 6 turnos; el contexto recuperado solo se inyecta en el
  turno actual (no se acumula) para ahorrar tokens.

## Evaluación con Ragas

Conjunto de evaluación: `eval_dataset.json` (20 preguntas, 16 dentro del corpus y 4 fuera para
medir alucinaciones). Métricas: `faithfulness`, `answer_relevancy`, `context_precision`,
`context_recall` + **tasa de rechazo correcto** (preguntas fuera del corpus respondidas con la
frase de rechazo) y **tasa de falso rechazo** (preguntas del corpus rechazadas por error).

```bash
pip install -r requirements-eval.txt
python evaluate_ragas.py --tag baseline --chunk-size 500  --overlap 50  --top-k 4
python evaluate_ragas.py --tag mejora   --chunk-size 1000 --overlap 150 --top-k 4
python evaluate_ragas.py --comparar baseline mejora
```

### Resultados *(completar con tus corridas)*

| Métrica | Baseline (500/50, k=4) | Mejora (…) | Variación |
|---|---|---|---|
| faithfulness | | | |
| answer_relevancy | | | |
| context_precision | | | |
| context_recall | | | |
| rechazo correcto (fuera de corpus) | | | |

### Análisis *(completar)*
- Métrica más baja y componente atribuido (chunking / recuperación / generación).
- Cambio realizado en la iteración de mejora, por qué se eligió y su efecto en las métricas.

## Ejecución local

```bash
python -m venv venv
.\venv\Scripts\Activate            # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env             # y edita tu clave (GROQ_API_KEY o GENAI_API_KEY)   (Mac/Linux: cp)
streamlit run app.py
```

## Despliegue (Streamlit Community Cloud)

1. Sube el proyecto a GitHub (el `.gitignore` ya excluye `.env`, `venv/` y `chroma_db/`).
2. En <https://share.streamlit.io> → **Create app** → selecciona el repo, rama y `app.py`.
3. En **Advanced settings**: Python 3.11 y, en **Secrets**, pega:
   ```toml
   GROQ_API_KEY = "tu_clave_de_groq_aqui"   # o GENAI_API_KEY si usas Gemini
   ```
4. **Deploy**. En el primer arranque se descarga el modelo de embeddings y se construye el índice
   desde `docs/` (tarda unos minutos); luego queda en caché.

## Estructura del repositorio

```
.
├── app.py  config.py  ingest.py  llm.py  rag_pipeline.py  evaluate_ragas.py
├── eval_dataset.json
├── docs/                      # corpus
├── results/                   # CSV, JSON y gráficos de las evaluaciones
├── avance1/                   # entrega anterior
├── requirements.txt  requirements-eval.txt
├── .env.example  .streamlit/secrets.toml.example  .gitignore
└── README.md
```
