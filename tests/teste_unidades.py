"""Testa isoladamente as duas partes com mais ramificacoes de logica.

- src/validate_spec.py: confere e conserta a spec Vega-Lite gerada pelo LLM;
- src/indexar.py: os ajudantes que leem o formato da base (rankings
  aninhados, nome da tarefa, descricao de cada design).

Nao chama nenhuma API e nao toca no banco vetorial.

    .venv/bin/python tests/teste_unidades.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from comum import caminho_src, confere

caminho_src()

from langchain_core.messages import AIMessage

import cache
import condicao
import config
import indexar
import recomendar
import validate_spec

BARRAS = {
    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
    "data": {"values": [{"categoria": "A", "vendas": 10}, {"categoria": "B", "vendas": 20}]},
    "mark": "bar",
    "encoding": {"x": {"field": "categoria", "type": "nominal"},
                 "y": {"field": "vendas", "type": "quantitative"}},
}


def erro(funcao, excecao, *args, **kwargs):
    """True se chamar `funcao` levantar `excecao`."""
    try:
        funcao(*args, **kwargs)
    except excecao:
        return True
    return False


def trocar_k():
    """Tentar mudar uma Condicao depois de criada (ela e congelada)."""
    condicao.PADRAO.k = 15


def spec_barras(**mudancas):
    """Copia da spec valida, com os canais de `encoding` trocados."""
    import copy

    spec = copy.deepcopy(BARRAS)
    spec["encoding"].update(mudancas)
    return spec


# --- validate_spec.validate --------------------------------------------------

print("1) validate: o que e spec valida e o que nao e")
confere("spec de barras completa passa", validate_spec.validate(BARRAS)[0])
confere("texto no lugar da spec e recusado", validate_spec.validate("nao sou json")[0] is False)
confere("spec sem 'mark' nem 'layer' e recusada",
        validate_spec.validate({"encoding": {}})[0] is False)
confere("'encoding' que nao e objeto e recusado",
        validate_spec.validate({"mark": "bar", "encoding": []})[0] is False)
confere("canal sem 'type' e recusado",
        validate_spec.validate(spec_barras(y={"field": "vendas"}))[0] is False)
confere("canal com 'value' em vez de 'type' e aceito",
        validate_spec.validate(spec_barras(color={"value": "steelblue"}))[0])
_, erros = validate_spec.validate(spec_barras(y={"field": "vendas"}))
confere("o erro diz qual canal falhou", erros and "encoding.y" in erros[0])

print("2) validate: o schema do Vega-Lite e consultado de fato")
confere("marca inexistente ('barra') e recusada pelo schema",
        validate_spec.validate({**BARRAS, "mark": "barra"})[0] is False)


# --- validate_spec.repair ----------------------------------------------------

print("3) repair: completa o 'type' que o LLM esquece")
spec, corrigidos = validate_spec.repair(spec_barras(color={"field": "categoria"},
                                                    size={"field": "vendas"}))
confere("campo de texto vira 'nominal'", spec["encoding"]["color"]["type"] == "nominal")
confere("campo de numero vira 'quantitative'", spec["encoding"]["size"]["type"] == "quantitative")
confere("os dois canais sao relatados", sorted(corrigidos) == ["color=nominal", "size=quantitative"])
confere("a spec consertada passa na validacao", validate_spec.validate(spec)[0])

spec, corrigidos = validate_spec.repair(spec_barras(y={"field": "vendas", "type": "nominal"}))
confere("canal que ja tem 'type' nao e mexido", spec["encoding"]["y"]["type"] == "nominal")
confere("nada e relatado quando nada muda", corrigidos == [])

spec, _ = validate_spec.repair(spec_barras(color={"field": "ativo"}))
confere("campo ausente nos dados vira 'nominal'", spec["encoding"]["color"]["type"] == "nominal")

spec, _ = validate_spec.repair(spec_barras(tooltip={"aggregate": "count"}))
confere("canal sem 'field' e ignorado", "type" not in spec["encoding"]["tooltip"])

confere("texto no lugar da spec nao quebra o repair",
        validate_spec.repair("nao sou json") == ("nao sou json", []))

print("4) repair: categorias ordenadas mantem a ordem dos dados")
spec, corrigidos = validate_spec.repair(spec_barras(x={"field": "mes", "type": "ordinal"}))
confere("canal ordinal recebe sort=None", spec["encoding"]["x"]["sort"] is None)
confere("a correcao e relatada", corrigidos == ["x: ordem dos dados"])
spec, _ = validate_spec.repair(spec_barras(x={"field": "mes", "type": "ordinal", "sort": "ascending"}))
confere("sort escolhido pelo LLM e preservado", spec["encoding"]["x"]["sort"] == "ascending")


# --- validate_spec.ensure_inline_data ---------------------------------------

print("5) ensure_inline_data: a spec sempre tem dados")
confere("spec sem 'data' recebe lista vazia",
        validate_spec.ensure_inline_data({"mark": "bar"})["data"] == {"values": []})
confere("dados existentes sao mantidos",
        validate_spec.ensure_inline_data(BARRAS.copy())["data"]["values"][0]["vendas"] == 10)
confere("dados passados de fora substituem os da spec",
        validate_spec.ensure_inline_data({"mark": "bar"}, [{"a": 1}])["data"]["values"] == [{"a": 1}])


# --- indexar: leitura do formato da base ------------------------------------

print("6) ranking_em_texto: niveis alternam entre '>' (melhor) e '=' (empate)")
confere("sem empate", indexar.ranking_em_texto(["E-1", "E-2", "E-3"]) == "E-1 > E-2 > E-3")
confere("empate no meio",
        indexar.ranking_em_texto([["E-1", "E-2"], "E-3"]) == "(E-1 = E-2) > E-3")
confere("lista de um nivel so e desembrulhada",
        indexar.ranking_em_texto([["E-1", "E-2"]]) == "E-1 > E-2")
confere("um unico design", indexar.ranking_em_texto("E-1") == "E-1")

print("7) tarefa_do_achado: o nome do achado carrega a tarefa")
confere("'sort-2' -> 'sort'", indexar.tarefa_do_achado("sort-2") == "sort")
confere("'retrieve-value-1' -> 'retrieve-value'",
        indexar.tarefa_do_achado("retrieve-value-1") == "retrieve-value")
confere("espacos viram hifens",
        indexar.tarefa_do_achado("characterize distribution 3") == "characterize-distribution")
confere("tarefa fora da lista vira 'general'", indexar.tarefa_do_achado("tempo-de-reacao") == "general")

print("8) achatar: pega todos os ids, em qualquer aninhamento")
confere("tres niveis", indexar.achatar([["E-1", ["E-2"]], "E-3"]) == ["E-1", "E-2", "E-3"])

print("9) design_em_texto: marca e canais de cada design")
linha = indexar.design_em_texto("E-1", {
    "layers": [{"mark": "area-rect",
                "encodings": [{"data-type": "quantitative", "channel": "length"},
                              {"data-type": "nominal", "channel": "positionX"}]}],
    "note": "Two bars are next to each other.",
})
confere("marca e codificacoes na linha",
        linha.startswith("- E-1: mark area-rect; quantitative -> length; nominal -> positionX"))
confere("a nota do artigo e preservada", 'note: "Two bars are next to each other."' in linha)
confere("design sem camadas nao quebra", indexar.design_em_texto("E-9", {}) == "- E-9: ")


# --- indexar: um achado -> um documento -------------------------------------

print("10) documentos_do_artigo: o template do documento indexado")
ARTIGO = {
    "Title": "Graphical Perception",
    "Designs": {
        "E-1": {"layers": [{"mark": "area-rect",
                            "encodings": [{"data-type": "quantitative", "channel": "length"},
                                           {"data-type": "nominal-2", "channel": "positionX"}]}]},
        "E-2": {"layers": [{"mark": "area-arc",
                            "encodings": [{"data-type": "quantitative", "channel": "angle"}]}]},
    },
    "Results": {"Experimental": {"accuracy": {
        "sort-1": {"rank": [["E-1", "E-2"]], "significance": {"pairs": [["E-1", "E-2"]]}},
        "cluster-1": {"rank": ["E-2", "E-1"], "significance": []},
    }}},
}
docs = list(indexar.documentos_do_artigo("Cleveland1984graphical.json", ARTIGO))
confere("dois achados viraram dois documentos", len(docs) == 2)

d = docs[0]
confere("a fonte identifica o achado dentro do artigo",
        d.metadata["fonte"] == "Cleveland1984graphical.json#Experimental/accuracy/sort-1")
confere("a tarefa vai para os metadados (permite filtrar)", d.metadata["tarefa"] == "sort")
linhas = d.page_content.splitlines()
confere("o documento comeca pela tarefa", linhas[0] == "task: sort")
confere("o sufixo numerico do tipo de dado e removido ('nominal-2' -> 'nominal')",
        linhas[1] == "data types: nominal, quantitative")
confere("a metrica e o grupo aparecem", linhas[2] == "metric: accuracy (experimental)")
confere("o titulo do estudo aparece", linhas[3] == "study: Graphical Perception")
confere("o ranking vem em texto", linhas[4] == "result, best to worst: E-1 > E-2")
confere("significancia com pares -> 'yes'",
        linhas[5] == "statistically significant differences: yes")
confere("os designs comparados sao descritos", linhas[6] == "designs compared:" and len(linhas) == 9)

confere("significancia como lista vazia -> 'not reported'",
        "statistically significant differences: not reported" in docs[1].page_content)

# --- condicao: o que pode variar num experimento ----------------------------

print("11) Condicao: valida os parametros e se identifica")
confere("a condicao PADRAO e o sistema como ele e",
        (condicao.PADRAO.recuperacao, condicao.PADRAO.k) == ("traduzida", 9))
confere("estrategia inexistente e recusada na criacao",
        erro(condicao.Condicao, ValueError, recuperacao="magica"))
confere("k abaixo de 1 e recusado", erro(condicao.Condicao, ValueError, k=0))
confere("k=0 e aceito quando nao se recupera nada",
        condicao.Condicao(recuperacao="nenhuma", k=0).k == 0)
confere("a condicao e imutavel", erro(trocar_k, Exception))

print("12) Condicao.identidade: mesmos parametros, mesma identidade")
confere("o rotulo nao entra na identidade",
        condicao.Condicao(nome="a").identidade() == condicao.Condicao(nome="b").identidade())
confere("k diferente -> identidade diferente",
        condicao.Condicao(k=9).identidade() != condicao.Condicao(k=15).identidade())
confere("estrategia diferente -> identidade diferente",
        condicao.Condicao(recuperacao="crua").identidade()
        != condicao.Condicao(recuperacao="hyde").identidade())
confere("modelo diferente -> identidade diferente",
        condicao.Condicao(modelo="gpt-4o-mini").identidade() != condicao.PADRAO.identidade())
confere("a identidade cabe num nome de pasta", len(condicao.PADRAO.identidade()) == 8)

print("13) montar_consulta: cada estrategia monta a consulta a seu modo")
TRADUCAO = {"tarefa": "sort", "tipos_de_dado": ["quantitative", "nominal"],
            "descricao": "compare values across categories"}
PERGUNTA = "quero comparar as vendas de 5 lojas"
consulta = recomendar.montar_consulta(condicao.PADRAO, PERGUNTA, TRADUCAO, llm=None)
confere("'traduzida' usa o formato dos documentos", consulta.startswith("task: sort\ndata types: "))
confere("'crua' usa a pergunta como ela veio",
        recomendar.montar_consulta(condicao.Condicao(recuperacao="crua"), PERGUNTA, TRADUCAO, None)
        == PERGUNTA)
confere("nem 'traduzida' nem 'crua' chamam o LLM (llm=None nao quebrou)", True)

print("14) os dois prompts de geracao pedem o mesmo formato de resposta")
CAMPOS = ('"grafico"', '"justificativa"', '"achados_usados"', '"vegalite_spec"', '"ressalva"')
confere("a ablacao pede exatamente os mesmos campos da condicao padrao",
        all(recomendar.PROMPT_RECOMENDACAO.count(c) == recomendar.PROMPT_SEM_ACHADOS.count(c) == 1
            for c in CAMPOS))
confere("o prompt com achados explica como ler os achados",
        "result, best to worst" in recomendar.PROMPT_RECOMENDACAO)
confere("o prompt da ablacao nao menciona achado nenhum",
        "ACHADOS DE ESTUDOS" not in recomendar.PROMPT_SEM_ACHADOS)
confere("os dois formatam sem sobrar chave de template",
        "{{" not in recomendar.PROMPT_SEM_ACHADOS.format(pergunta="x")
        and "{{" not in recomendar.PROMPT_RECOMENDACAO.format(pergunta="x", achados="y"))

# --- cache: reproduzir uma execucao sem chamar a API ------------------------

print("15) cache: a chave distingue modelo, prompt e repeticao")
confere("prompts diferentes -> chaves diferentes",
        cache.chave("m", "a") != cache.chave("m", "b"))
confere("modelos diferentes -> chaves diferentes",
        cache.chave("m1", "a") != cache.chave("m2", "a"))
confere("REPETICOES diferentes -> chaves diferentes (senao a variacao medida seria zero)",
        cache.chave("m", "a", 1) != cache.chave("m", "a", 2))
confere("a mesma chamada -> sempre a mesma chave",
        cache.chave("m", "a", 3) == cache.chave("m", "a", 3))

PASTA = Path(tempfile.mkdtemp())

print("16) cache: grava, le e guarda o prompt junto")
confere("chamada que nunca aconteceu devolve None",
        cache.ler("m", "nunca perguntei", pasta=PASTA) is None)
arquivo = cache.gravar("m", "PROMPT", "RESPOSTA", pasta=PASTA)
confere("a resposta volta igual", cache.ler("m", "PROMPT", pasta=PASTA) == "RESPOSTA")
guardado = json.loads(arquivo.read_text(encoding="utf-8"))
confere("o prompt inteiro fica no arquivo (sem ele nao da para auditar)",
        guardado["prompt"] == "PROMPT" and guardado["modelo"] == "m")
confere("nao sobra arquivo parcial", list(PASTA.glob("*.parcial")) == [])
arquivo.write_text("{isso nao e json", encoding="utf-8")
confere("arquivo corrompido e tratado como ausente, nao quebra",
        cache.ler("m", "PROMPT", pasta=PASTA) is None)

print("17) ChatComCache: a segunda vez nao chama o modelo")
chamadas = []


class LlmQueConta:
    """Modelo falso que registra cada chamada recebida."""

    def invoke(self, prompt):
        chamadas.append(prompt)
        return AIMessage(content=f"resposta {len(chamadas)}")


com_cache = cache.ChatComCache(LlmQueConta(), "modelo-de-teste", pasta=PASTA)
primeira = config.texto(com_cache.invoke("PERGUNTA"))
segunda = config.texto(com_cache.invoke("PERGUNTA"))
confere("o modelo foi chamado uma vez so", len(chamadas) == 1)
confere("as duas respostas sao iguais", primeira == segunda == "resposta 1")
confere("contou 1 falta e 1 acerto", (com_cache.faltas, com_cache.acertos) == (1, 1))

outra_repeticao = cache.ChatComCache(LlmQueConta(), "modelo-de-teste", repeticao=2, pasta=PASTA)
outra_repeticao.invoke("PERGUNTA")
confere("a repeticao 2 da MESMA pergunta chama o modelo de novo", len(chamadas) == 2)
confere("e a repeticao 1 continua guardada separada",
        cache.ler("modelo-de-teste", "PERGUNTA", 1, PASTA) == "resposta 1")

print("18) EmbeddingsComCache: vetoriza so o que falta")
lotes, consultas = [], []


class EmbeddingsQueContam:
    """Modelo falso que registra o que recebeu por cada caminho.

    Consulta e documento devolvem vetores DIFERENTES de proposito: e assim que
    o provedor real se comporta (task_type RETRIEVAL_QUERY x RETRIEVAL_DOCUMENT),
    e e o que o teste abaixo precisa distinguir.
    """

    def embed_documents(self, textos):
        lotes.append(list(textos))
        return [[float(len(t)), 1.0] for t in textos]

    def embed_query(self, texto):
        consultas.append(texto)
        return [float(len(texto)), 2.0]


emb = cache.EmbeddingsComCache(EmbeddingsQueContam(), "emb-de-teste", pasta=PASTA)

confere("consulta passa pelo embed_query do provedor, nao pelo embed_documents",
        emb.embed_query("abc") == [3.0, 2.0] and consultas == ["abc"] and lotes == [])
confere("a segunda consulta igual nao chama a API",
        emb.embed_query("abc") == [3.0, 2.0] and len(consultas) == 1)
confere("o MESMO texto como documento tem outro vetor, e nao reusa o da consulta",
        emb.embed_documents(["abc"]) == [[3.0, 1.0]] and lotes == [["abc"]])

emb.embed_documents(["abc", "de", "f"])
confere("num lote, so os textos que faltam sao enviados", lotes[-1] == ["de", "f"])
confere("e o lote volta na ordem certa e completo",
        emb.embed_documents(["abc", "de", "f"]) == [[3.0, 1.0], [2.0, 1.0], [1.0, 1.0]])
confere("contou acertos e faltas", (emb.acertos, emb.faltas) == (1 + 1 + 3, 1 + 1 + 2))
confere("modelo diferente nao reusa o vetor do outro",
        cache.EmbeddingsComCache(EmbeddingsQueContam(), "outro-emb", pasta=PASTA)
        ._guardado("abc", "documento") is None)

print("19) cache.resumo: conta o que esta guardado")
por_modelo, bytes_totais = cache.resumo(PASTA)
confere("conta as chamadas do modelo de teste", por_modelo.get("modelo-de-teste") == 2)
confere("soma algum tamanho", bytes_totais > 0)
confere("pasta inexistente nao quebra o resumo", cache.resumo(PASTA / "nao-existe") == ({}, 0))

shutil.rmtree(PASTA, ignore_errors=True)

print("\nTUDO OK.")
