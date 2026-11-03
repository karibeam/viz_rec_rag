"""Configuracao: caminhos do projeto, chave de API e modelos (Gemini ou OpenAI)."""

import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"        # os 59 JSONs da base, intactos
CHROMA_DIR = ROOT / "data" / "chroma"  # o banco vetorial
COLLECTION = "achados"

load_dotenv(ROOT / ".env")


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "gemini").strip().lower()


def _require(var: str) -> str:
    val = os.getenv(var, "").strip()
    if not val:
        raise RuntimeError(f"{var} nao definida. Copie .env.example para .env e preencha a chave.")
    return val


GEMINI_PADRAO = "gemini-3.5-flash-lite"
OPENAI_PADRAO = "gpt-4o-mini"


def nome_do_chat(modelo: str | None = None) -> str:
    """O id do modelo que get_chat(modelo) vai usar de fato.

    A chave do cache depende desse nome, entao ele tem que ser resolvido aqui
    e nao adivinhado: um experimento rodado com outro modelo nao pode reusar a
    resposta guardada do anterior.
    """
    if modelo:
        return modelo
    if provider() == "openai":
        return os.getenv("OPENAI_CHAT_MODEL", OPENAI_PADRAO)
    return os.getenv("GEMINI_CHAT_MODEL", GEMINI_PADRAO)


def get_chat(modelo: str | None = None):
    """Modelo de chat, pedindo temperatura 0 (resposta mais estavel possivel).

    ATENCAO: o gemini-3.5-flash-lite usa amostragem fixa e IGNORA esse pedido
    ("uses fixed sampling defaults; the sampling parameter(s) temperature will
    be ignored"). Logo, o sistema NAO e deterministico: a mesma pergunta pode
    receber recomendacoes diferentes em execucoes diferentes. Veja
    "Limitacoes conhecidas" no README.
    """
    if provider() == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=nome_do_chat(modelo),
            temperature=0,
            api_key=_require("OPENAI_API_KEY"),
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=nome_do_chat(modelo),
        temperature=0,
        google_api_key=_require("GOOGLE_API_KEY"),
    )


def get_embeddings():
    """Modelo de embeddings (os dois provedores sao multilingues)."""
    if provider() == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            api_key=_require("OPENAI_API_KEY"),
        )

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001"),
        google_api_key=_require("GOOGLE_API_KEY"),
    )


def texto(resposta) -> str:
    """Texto de uma resposta do LLM (alguns modelos devolvem uma lista de partes)."""
    content = getattr(resposta, "content", resposta)
    if isinstance(content, list):
        return "".join(
            p if isinstance(p, str) else str(p.get("text", ""))
            for p in content
            if isinstance(p, str) or (isinstance(p, dict) and p.get("type") == "text")
        )
    return str(content)


def com_retentativa(funcao, tentativas: int = 4):
    """Executa `funcao()` de novo quando a API esta sobrecarregada ou no limite por minuto.

    O plano gratuito limita chamadas por minuto (erro 429) e as vezes o modelo
    fica sobrecarregado (erro 503). Nos dois casos basta esperar. Ja a cota
    DIARIA esgotada nao se resolve esperando, entao o erro sobe na hora.
    """
    for tentativa in range(1, tentativas + 1):
        try:
            return funcao()
        except Exception as exc:
            msg = str(exc)
            temporario = any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"))
            if not temporario or "PerDay" in msg or tentativa == tentativas:
                raise
            sugerido = re.search(r"retry in ([\d.]+)s", msg, re.IGNORECASE)
            time.sleep(float(sugerido.group(1)) + 2 if sugerido else 30 * tentativa)
