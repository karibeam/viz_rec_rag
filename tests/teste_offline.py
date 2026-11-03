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

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ / "experimentos"))

from comum import confere

import config

config.CHROMA_DIR = Path(tempfile.mkdtemp())  # banco temporario, nao mexe no real

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import diario as diario_mod
import indexar
import rodar
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

print("8) diario: uma pasta por execucao, com tudo o que foi feito")
PASTA = Path(tempfile.mkdtemp())
SEM_FONTE = {**RECOMENDACAO, "achados_usados": []}
HIPOTETICO = "task: sort\ndata types: quantitative, nominal"


def fabrica_falsa(cond, repeticao, d):
    """O modelo falso de cada execucao, envolvido pelo diario."""
    respostas = [TRADUCAO]
    if cond.recuperacao == "hyde":
        respostas.append(HIPOTETICO)
    respostas.append(SEM_FONTE if cond.recuperacao == "nenhuma" else RECOMENDACAO)
    return diario_mod.LlmComDiario(llm_falso(*respostas), d)


CONFIGURACAO = {
    "nome": "teste-offline",
    "repeticoes": 2,
    "condicoes": [{"nome": "traduzida", "recuperacao": "traduzida", "k": 3},
                  {"nome": "hyde", "recuperacao": "hyde", "k": 3},
                  {"nome": "sem-rag", "recuperacao": "nenhuma", "k": 0}],
    "perguntas": ["quero comparar vendas de 5 categorias", "quero ver a distribuicao das idades"],
}
linhas = rodar.rodar(CONFIGURACAO, raiz=PASTA, fabrica=fabrica_falsa, banco=banco)

confere("3 condicoes x 2 perguntas x 2 repeticoes = 12 execucoes", len(linhas) == 12)
confere("nenhuma deu erro", not any("erro" in l for l in linhas))
pastas = sorted(d for d in PASTA.iterdir() if d.is_dir())
confere("uma pasta por execucao, sem colisao", len(pastas) == 12)

d = Path(linhas[0]["pasta"])
confere("gravou o prompt e a resposta de cada chamada",
        (d / "chamadas/01-prompt.txt").exists() and (d / "chamadas/02-resposta.txt").exists())
confere("o prompt gravado e o da traducao de fato",
        "DECISAO 1" in (d / "chamadas/01-prompt.txt").read_text(encoding="utf-8"))
confere("gravou condicao, pergunta, achados e resultado",
        all((d / n).exists() for n in ("condicao.json", "pergunta.txt", "achados.json", "resultado.json")))
ambiente = json.loads((d / "ambiente.json").read_text(encoding="utf-8"))
confere("o ambiente registra o commit do codigo e se a arvore estava suja",
        "commit" in ambiente and "codigo_sujo" in ambiente)
confere("o ambiente registra os modelos resolvidos",
        ambiente["modelo_de_chat"] and ambiente["modelo_de_embedding"])

print("9) o resumo do experimento sai analisavel")
resumo = [json.loads(l) for l in (PASTA / "resumo.jsonl").read_text(encoding="utf-8").splitlines()]
confere("uma linha de resumo por execucao", len(resumo) == 12)
hyde = [l for l in resumo if l["condicao"] == "hyde"]
confere("a condicao hyde fez 3 chamadas ao modelo",
        len(list((Path(hyde[0]["pasta"]) / "chamadas").glob("*-prompt.txt"))) == 3)
sem_rag = [l for l in resumo if l["condicao"] == "sem-rag"]
confere("a ablacao nao recuperou nada e nao citou fonte",
        all(l["achados_recuperados"] == 0 and l["fontes_citadas"] == 0 for l in sem_rag))
confere("a ablacao gravou achados.json vazio",
        json.loads((Path(sem_rag[0]["pasta"]) / "achados.json").read_text(encoding="utf-8")) == [])
traduzida = [l for l in resumo if l["condicao"] == "traduzida"]
confere("as condicoes com RAG recuperaram k achados",
        all(l["achados_recuperados"] == 3 for l in traduzida + hyde))
confere("as repeticoes 1 e 2 existem e ficaram em pastas distintas",
        {l["repeticao"] for l in resumo} == {1, 2}
        and len({l["pasta"] for l in resumo}) == 12)

rodar.imprimir_resumo(resumo)
shutil.rmtree(PASTA, ignore_errors=True)

shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)
print("\nTUDO OK.")
