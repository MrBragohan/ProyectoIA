"""Capa mínima de acceso al LLM: Groq o Gemini con la misma interfaz.

El resto del proyecto solo conoce `LLMClient.generar(...)`; así se puede cambiar de
proveedor con la variable de entorno LLM_PROVIDER ("groq" o "gemini") sin tocar el pipeline.

Los mensajes se pasan como lista de dicts: {"role": "user" | "assistant", "content": "..."}.
"""
import time

import config

# Esperas (segundos) entre reintentos: cubre errores 503 (alta demanda) y 429 (límite de tasa)
_ESPERAS = [2, 5, 10]


class LLMClient:
    def __init__(self, proveedor=None):
        self.proveedor = (proveedor or config.LLM_PROVIDER).lower()
        self.modelo = config.LLM_MODEL

        if self.proveedor == "groq":
            from groq import Groq
            self._cliente = Groq(api_key=config.GROQ_API_KEY)
        elif self.proveedor == "gemini":
            from google import genai
            self._cliente = genai.Client(api_key=config.GENAI_API_KEY)
        else:
            raise ValueError(
                f"LLM_PROVIDER='{self.proveedor}' no es válido. Usa 'groq' o 'gemini'."
            )

    def generar(self, mensajes, sistema=None, temperatura=0.2, max_tokens=4096):
        """Devuelve el texto de la respuesta, con reintentos ante fallos transitorios."""
        for intento in range(len(_ESPERAS) + 1):
            try:
                return self._generar_una_vez(mensajes, sistema, temperatura, max_tokens)
            except Exception:
                if intento == len(_ESPERAS):
                    raise
                time.sleep(_ESPERAS[intento])

    # ------------------------------------------------------------------
    def _generar_una_vez(self, mensajes, sistema, temperatura, max_tokens):
        if self.proveedor == "groq":
            msgs = ([{"role": "system", "content": sistema}] if sistema else []) + list(mensajes)
            resp = self._cliente.chat.completions.create(
                model=self.modelo,
                messages=msgs,
                temperature=temperatura,
                max_completion_tokens=max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()

        # Gemini
        from google.genai import types
        contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in mensajes
        ]
        cfg = types.GenerateContentConfig(
            system_instruction=sistema,
            temperature=temperatura,
            max_output_tokens=max_tokens,
        )
        resp = self._cliente.models.generate_content(
            model=self.modelo, contents=contents, config=cfg
        )
        return (resp.text or "").strip()


def crear_juez_langchain():
    """Chat model de LangChain que usa Ragas como LLM 'juez' (solo para evaluar)."""
    if config.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=config.GROQ_MODEL, temperature=0, api_key=config.GROQ_API_KEY)

    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL, google_api_key=config.GENAI_API_KEY, temperature=0
    )
