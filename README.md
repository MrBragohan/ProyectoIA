# ProyectoIA

# Tutor Académico Personalizado (RAG) - Creación de Aplicaciones con IA

## Descripción
Asistente basado en IA que responde preguntas de estudiantes de Ingeniería de Sistemas
sobre la asignatura de Creación de Aplicaciones con IA, utilizando una base de conocimiento
local (apuntes de clase) y el modelo Gemini de Google.

El sistema sigue una arquitectura RAG (Retrieval-Augmented Generation):
1. **Retrieval:** se recuperan los fragmentos más relevantes del material de estudio
   según la pregunta del usuario (`retriever.py`).
2. **Augmented Generation:** los fragmentos recuperados se inyectan en el prompt como
   contexto, y el modelo genera la respuesta basándose únicamente en esa información.

## Enfoque elegido
Tutor Académico Personalizado, basado en apuntes de la asignatura de Creación de
Aplicaciones con IA. La base de conocimiento cubre 15 temas: Modelos de Lenguaje (LLMs),
Prompt Engineering, Few-Shot Prompting, Delimitadores en Prompts, RAG, Embeddings y Bases
de Datos Vectoriales, Agentes de IA, Function Calling / Tool Use, Fine-Tuning vs Prompt
Engineering, Alucinaciones en IA, Temperatura y Parámetros de Generación, Evaluación de
Aplicaciones de IA, Multimodalidad, Despliegue de Aplicaciones de IA, y Ética y Sesgos en IA.

## Estructura del proyecto
- `knowledge_base.txt`: material de estudio (base de conocimiento) sobre Creación de
  Aplicaciones con IA.
- `retriever.py`: función de recuperación de contexto por coincidencia de palabras clave.
- `main.py`: configuración del cliente, system prompt, few-shot y bucle de conversación.
- `.env`: variables de entorno (API Key), no se sube al repositorio.

## Técnicas de Prompt Engineering aplicadas
- **System Prompt:** define el rol (Tutor), el tono (paciente, motivador) y las reglas
  estrictas de uso del contexto.
- **Few-Shot Prompting:** se precarga un ejemplo de pregunta-respuesta en el historial
  del chat para fijar el formato de salida esperado (Markdown con secciones fijas).
- **Delimitadores:** se usan etiquetas XML (`<contexto>`, `<pregunta>`) para separar
  claramente el material recuperado de la pregunta del usuario.
- **Control de alucinaciones:** el prompt instruye explícitamente al modelo a responder
  "No tengo información suficiente..." si la respuesta no está en el contexto.

## Cómo ejecutar
1. Crear y activar un entorno virtual (opcional pero recomendado):
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate   # Windows
   source venv/bin/activate  # Mac/Linux
   ```
2. Instalar dependencias:
   ```bash
   pip install google-genai python-dotenv
   ```
3. Crear un archivo `.env` con tu clave:
   ```
   GENAI_API_KEY=tu_clave_aqui
   ```
4. Ejecutar:
   ```bash
   python main.py
   ```
5. Escribir preguntas relacionadas con el material de estudio (por ejemplo,
   "¿qué es RAG?" o "¿qué es el function calling?"). Escribir "salir" para terminar.

## Ejemplo de uso
```
Estudiante: ¿qué es RAG?

Tutor:
## Explicación
RAG (Retrieval-Augmented Generation) combina un sistema de recuperación de información
con un modelo generativo...

## Ejemplo
...

## Fuente
Tema: RAG - Retrieval-Augmented Generation
```

## Próximos pasos (Avance 2)
- Reemplazar la recuperación por palabras clave con embeddings y similitud semántica.
- Incorporar múltiples documentos (PDF) como fuente de conocimiento.
- Añadir capacidades de agente (por ejemplo, generar quizzes automáticamente sobre el
  tema consultado, usando Function Calling).dsadsa