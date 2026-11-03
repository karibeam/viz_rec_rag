"""Testa o pipeline inteiro SEM chamar nenhuma API (modelos falsos).

Verifica que as etapas se encaixam: base -> documentos -> banco vetorial ->
traducao da pergunta -> busca -> recomendacao -> spec Vega-Lite.
Nao mede qualidade da resposta, so que nada quebra.

    .venv/bin/python tests/teste_offline.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from comum import confere

import config

config.CHROMA_DIR = Path(tempfile.mkdtemp())  # banco temporario, nao mexe no real

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import indexar
from condicao import Condicao
from recomendar import recomendar

TRADUCAO = {"tarefa": "sort", "tipos_de_dado": ["quantitative", "nominal"],
            "descricao": "compare quantitative values across nominal categories"}
RECOMENDACAO = {
    "grafico": "gráfico de barras",
    "justificativa": "O achado 1 mostrou que barras permitem comparar categorias com mais acerto.",
    "achados_usados": [1],
    "vegalite_spec": {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "Vendas por categoria",
        "data": {"values": [{"categoria": "A", "vendas": 10}, {"categoria": "B", "vendas": 20}]},
        "mark": "bar",
        "encoding": {"x": {"field": "categoria", "type": "nominal"},
                     "y": {"field": "vendas", "type": "quantitative"}},
    },
    "ressalva": "",
}


def llm_falso(*respostas):
    """Devolve as respostas na ordem pedida. Dicionario sai como JSON; texto, como texto.

    O passo do HyDE nao pede JSON: ele pede um achado hipotetico em texto
    corrido, entao o modelo falso precisa saber devolver os dois formatos.
    """
    textos = [r if isinstance(r, str) else json.dumps(r) for r in respostas]
    return GenericFakeChatModel(messages=iter(AIMessage(content=t) for t in textos))


print("1) base -> documentos")
docs = indexar.todos_os_documentos()
confere(f"{len(docs)} achados viraram documentos", len(docs) == 240)
confere("cada documento comeca pela tarefa", all(d.page_content.startswith("task: ") for d in docs))

print("2) documentos -> banco vetorial")
banco = indexar.abrir_banco(DeterministicFakeEmbedding(size=64))
banco.add_documents(docs[:20], ids=[d.metadata["fonte"] for d in docs[:20]])
confere("20 documentos indexados", len(banco.get(include=[])["ids"]) == 20)

print("3) condicao PADRAO: pergunta -> traducao -> busca -> recomendacao")
r = recomendar("quero comparar vendas de 5 categorias", Condicao(k=5),
               llm=llm_falso(TRADUCAO, RECOMENDACAO), banco=banco)
confere("pergunta traduzida para a tarefa 'sort'", r["traducao"]["tarefa"] == "sort")
confere("consulta no mesmo formato dos documentos", r["consulta"].startswith("task: sort"))
confere("5 achados recuperados (o k da condicao)", len(r["achados"]) == 5)
confere("um grafico recomendado", r["grafico"] == "gráfico de barras")
confere("spec Vega-Lite valida", r["spec_valida"])
confere("fonte citada", len(r["fontes"]) == 1)
confere("a condicao usada fica registrada no resultado",
        r["condicao"]["recuperacao"] == "traduzida" and r["condicao"]["k"] == 5)

print("4) pergunta fora do escopo")
r = recomendar("qual a cor da capa do meu relatorio?",
               llm=llm_falso({"tarefa": "nenhuma", "tipos_de_dado": [], "descricao": ""}), banco=banco)
confere("recusada sem buscar nada", r.get("fora_de_escopo") and "achados" not in r)

print("5) recuperacao 'crua': a pergunta vai ao banco como veio")
PERGUNTA = "quero comparar vendas de 5 categorias"
r = recomendar(PERGUNTA, Condicao(nome="crua", recuperacao="crua", k=3),
               llm=llm_falso(TRADUCAO, RECOMENDACAO), banco=banco)
confere("a consulta e a propria pergunta", r["consulta"] == PERGUNTA)
confere("a traducao ainda acontece (a recusa depende dela)", r["traducao"]["tarefa"] == "sort")
confere("3 achados recuperados", len(r["achados"]) == 3)

print("6) recuperacao 'hyde': um achado inventado pelo LLM vira a consulta")
HIPOTETICO = "task: sort\ndata types: quantitative, nominal\nmetric: accuracy (experimental)"
r = recomendar(PERGUNTA, Condicao(nome="hyde", recuperacao="hyde", k=2),
               llm=llm_falso(TRADUCAO, HIPOTETICO, RECOMENDACAO), banco=banco)
confere("a consulta e o achado hipotetico", r["consulta"] == HIPOTETICO)
confere("2 achados recuperados", len(r["achados"]) == 2)

print("7) recuperacao 'nenhuma': a ablacao responde sem evidencia")
SEM_FONTE = {**RECOMENDACAO, "achados_usados": []}
r = recomendar(PERGUNTA, Condicao(nome="ablacao", recuperacao="nenhuma"),
               llm=llm_falso(TRADUCAO, SEM_FONTE), banco=None)
confere("nao recuperou nada", r["achados"] == [] and r["consulta"] == "")
confere("nao citou fonte", r["fontes"] == [])
confere("ainda recomendou um grafico", r["grafico"] == "gráfico de barras")
confere("a spec continua sendo validada", r["spec_valida"])
confere("nem precisou do banco vetorial (banco=None)", True)

shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)
print("\nTUDO OK.")
