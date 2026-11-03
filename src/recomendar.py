"""ETAPA ONLINE (roda a cada pergunta): pergunta -> recomendacao de um grafico.

Passo 1. TRADUZIR a pergunta leiga para o vocabulario tecnico da base
         (tarefa analitica + tipos de dado). E aqui que se fecha o gap
         semantico: a pergunta passa a "falar a lingua" dos documentos.
Passo 2. BUSCAR os k achados mais parecidos (similaridade de cosseno).
Passo 3. GERAR: o LLM le os achados e recomenda UM grafico, citando as fontes.

COMO a consulta do passo 2 e montada, QUANTOS achados ela traz e QUAL modelo
responde sao parametros de uma Condicao (src/condicao.py), nao constantes:
e isso que permite comparar estrategias em vez de so defender a atual.
Chamar sem passar condicao roda a PADRAO, que e o sistema como ele e.

    python src/recomendar.py "quero comparar as vendas de 5 categorias"
    python src/recomendar.py --recuperacao crua "quero comparar as vendas"
    python src/recomendar.py --recuperacao nenhuma "..."   # ablacao: sem RAG
"""

import argparse
import difflib
import json
import re
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

import cache as cache_mod
import condicao as cond_mod
import config
import validate_spec
from indexar import TAREFAS, abrir_banco

K = cond_mod.Condicao.k  # 9 achados: abaixo disso, um unico artigo tende a dominar


# --- Passo 1: traduzir a pergunta -------------------------------------------

PROMPT_TRADUCAO = """Traduza a pergunta de um usuario leigo para o vocabulario tecnico
de uma base de estudos sobre percepcao de graficos.

Pergunta: "{pergunta}"

DECISAO 1 -- a pergunta e sobre mostrar ou analisar DADOS num grafico?
Se NAO (ex.: escolher software, cores ou layout de um documento, duvidas
gerais), responda com tarefa "nenhuma" e pare aqui.

DECISAO 2 -- se SIM, escolha SEMPRE a tarefa mais proxima da lista abaixo,
mesmo que nenhuma encaixe perfeitamente (ex.: "ver a evolucao ao longo do
tempo" nao tem tarefa propria: escolha a mais proxima). Nunca use "nenhuma"
para uma pergunta sobre dados.
- aggregate: estimar media, total ou valor agregado
- characterize-distribution: ver como os valores se distribuem
- cluster: identificar grupos de itens parecidos
- correlate: ver a relacao entre duas variaveis
- determine-range: ver o intervalo (minimo e maximo)
- filter: selecionar itens que atendem a um criterio
- find-anomalies: achar valores fora do padrao
- find-extremum: achar o maior ou o menor valor
- retrieve-value: ler o valor exato de um item
- sort: ordenar ou comparar valores entre categorias

Tipos de dado envolvidos, entre: quantitative (numeros), nominal (categorias
sem ordem), ordinal (categorias com ordem, inclusive tempo: meses, anos, dias).

NAO sugira nenhum grafico: apenas classifique.
Responda so com JSON: {{"tarefa": "...", "tipos_de_dado": ["..."], "descricao": "uma frase em ingles descrevendo a analise em termos tecnicos, sem citar tipos de grafico"}}"""


def traduzir_pergunta(pergunta, llm):
    resposta = config.com_retentativa(lambda: llm.invoke(PROMPT_TRADUCAO.format(pergunta=pergunta)))
    traducao = ler_json(config.texto(resposta))
    # O LLM as vezes escreve um nome quase certo ("retrieving-value"):
    # troca pela tarefa valida mais parecida.
    tarefa = traducao.get("tarefa", "")
    if tarefa != "nenhuma" and tarefa not in TAREFAS:
        parecida = difflib.get_close_matches(tarefa, TAREFAS, n=1, cutoff=0)
        traducao["tarefa"] = parecida[0]
    return traducao


def consulta_tecnica(traducao):
    """Monta a consulta no MESMO formato dos documentos (ver indexar.py)."""
    return (
        f"task: {traducao['tarefa']}\n"
        f"data types: {', '.join(traducao.get('tipos_de_dado') or [])}\n"
        f"analysis: {traducao.get('descricao', '')}"
    )


# --- Passo 2: montar a consulta e buscar ------------------------------------

PROMPT_HYDE = """Escreva o resumo de um achado HIPOTETICO de um estudo experimental de
percepcao de graficos que responderia a esta pergunta:

"{pergunta}"

Imite exatamente este formato, em ingles, sem comentar nada:
task: <uma tarefa analitica>
data types: <tipos de dado separados por virgula>
metric: <accuracy ou time> (experimental)
result, best to worst: <designs do melhor para o pior>
designs compared:
- E-1: mark <marca>; <tipo de dado> -> <canal visual>"""


def achado_hipotetico(pergunta, llm):
    """HyDE: o LLM inventa um achado plausivel, e ELE -- nao a pergunta -- vira a consulta.

    A aposta do HyDE e que um texto falso no formato dos documentos cai mais
    perto deles no espaco vetorial do que a pergunta verdadeira. E a
    alternativa obvia a traducao, e por isso precisa ser medida, nao suposta.
    """
    resposta = config.com_retentativa(lambda: llm.invoke(PROMPT_HYDE.format(pergunta=pergunta)))
    return config.texto(resposta).strip()


def montar_consulta(cond, pergunta, traducao, llm):
    """O texto que vai ao banco vetorial, conforme a estrategia da condicao."""
    if cond.recuperacao == "crua":
        return pergunta
    if cond.recuperacao == "hyde":
        return achado_hipotetico(pergunta, llm)
    return consulta_tecnica(traducao)  # "traduzida"


def buscar(consulta, banco, k=K):
    """Os k achados mais parecidos. No Chroma (cosseno): similaridade = 1 - distancia."""
    return [(doc, 1 - dist) for doc, dist in banco.similarity_search_with_score(consulta, k=k)]


# --- Passo 3: gerar a recomendacao ------------------------------------------

COMO_LER_ACHADOS = """Como ler os achados: "result, best to worst" ordena os designs do melhor para
o pior no estudo ('=' e empate). Cada design e descrito pela marca e pelo canal
visual: mark area-rect = barras; line = linhas; point = pontos (dispersao);
area-arc = pizza/rosca; area-circle = bolhas; text = tabela de texto;
length/position = comprimento/posicao; angle = angulo; color-hue = cor."""

# Os dois prompts de geracao compartilham o contrato de saida: numa ablacao, a
# unica coisa que pode diferir entre as condicoes e a evidencia, nunca o
# formato pedido. Os marcadores sao __MAIUSCULAS__ e nao chaves, para nao
# colidirem com o str.format que insere a pergunta depois.
FORMATO_RESPOSTA = """Responda so com JSON:
{{
  "grafico": "nome popular do grafico, em portugues",
  "justificativa": "__JUSTIFICATIVA__",
  "achados_usados": [__ACHADOS_USADOS__],
  "vegalite_spec": {{ spec Vega-Lite v5 completa, com "$schema", "title", "data": {{"values": [4 a 8 linhas de exemplo coerentes com a pergunta]}}, "mark" e "encoding" com "type" em cada canal }},
  "ressalva": "limitacao importante, ou \\"\\" se nao houver"
}}"""


def formato_resposta(justificativa, achados_usados=""):
    return (FORMATO_RESPOSTA
            .replace("__JUSTIFICATIVA__", justificativa)
            .replace("__ACHADOS_USADOS__", achados_usados))

PROMPT_RECOMENDACAO = """Voce ajuda pessoas que NAO sao especialistas a escolher um grafico.

PERGUNTA: {pergunta}

ACHADOS DE ESTUDOS CIENTIFICOS (sua unica fonte de conhecimento):
{achados}

""" + COMO_LER_ACHADOS + """

Recomende UM unico grafico, com base apenas nesses achados:
- de preferencia os achados cuja "task" combina com a pergunta e com
  "statistically significant differences: yes";
- se os achados discordarem, escolha o mais bem sustentado e diga por que.

""" + formato_resposta(
    "2 a 4 frases simples, citando os achados pelo numero (ex: 'o achado 2 mostrou que...'); "
    "nao use codigos de design como E-1 ou T-2, diga o nome do grafico",
    "numeros dos achados que sustentam a escolha",
)

# Ablacao (recuperacao="nenhuma"): mesmo pedido, sem evidencia nenhuma. Isola o
# efeito do RAG -- o que sobra e o que o LLM sabe por conta propria.
PROMPT_SEM_ACHADOS = """Voce ajuda pessoas que NAO sao especialistas a escolher um grafico.

PERGUNTA: {pergunta}

Recomende UM unico grafico.

""" + formato_resposta("2 a 4 frases simples explicando a escolha")


def referencia(fonte):
    """'Aigner2011bertin.json#...' -> 'Aigner (2011). Bertin was Right: ...'

    Autor e ano vem do nome do arquivo; o titulo, do proprio JSON da base.
    Nao precisa reindexar: e so para exibir.
    """
    arquivo = fonte.split("#")[0]
    autor, ano = re.match(r"([A-Za-z-]+?)(\d{4})", arquivo).groups()
    titulo = json.loads((config.RAW_DIR / arquivo).read_text(encoding="utf-8")).get("Title", "")
    return f"{autor} ({ano}). {titulo}"


def formatar_achados(resultados):
    return "\n\n".join(
        f"[ACHADO {i}] (fonte: {doc.metadata['fonte']})\n{doc.page_content}"
        for i, (doc, _) in enumerate(resultados, 1)
    )


def recomendar(pergunta, cond=cond_mod.PADRAO, llm=None, banco=None):
    # Sem llm explicito, usa o modelo da condicao COM cache em disco: a mesma
    # pergunta na mesma condicao devolve a mesma resposta e nao gasta chamada.
    llm = llm or cache_mod.chat(cond)

    # Passo 1
    traducao = traduzir_pergunta(pergunta, llm)
    if traducao.get("tarefa") == "nenhuma":
        return {
            "pergunta": pergunta,
            "condicao": cond.como_dict(),
            "traducao": traducao,
            "fora_de_escopo": True,
            "mensagem": "Essa pergunta nao parece ser sobre qual grafico usar. "
                        "Descreva a analise que voce quer fazer com seus dados.",
        }

    # Passo 2 -- a ablacao salta a busca inteira (nem abre o banco vetorial)
    if cond.recuperacao == "nenhuma":
        consulta, resultados = "", []
        prompt = PROMPT_SEM_ACHADOS.format(pergunta=pergunta)
    else:
        banco = banco or abrir_banco(cache_mod.embeddings())
        consulta = montar_consulta(cond, pergunta, traducao, llm)
        resultados = buscar(consulta, banco, cond.k)
        prompt = PROMPT_RECOMENDACAO.format(pergunta=pergunta, achados=formatar_achados(resultados))

    # Passo 3
    resposta = ler_json(config.texto(config.com_retentativa(lambda: llm.invoke(prompt))))

    spec = resposta.get("vegalite_spec")
    validate_spec.ensure_inline_data(spec)
    spec, _ = validate_spec.repair(spec)
    spec_ok, _ = validate_spec.validate(spec)

    usados = [n for n in resposta.get("achados_usados") or [] if isinstance(n, int) and 1 <= n <= len(resultados)]
    return {
        "pergunta": pergunta,
        "condicao": cond.como_dict(),
        "traducao": traducao,
        "consulta": consulta,
        "grafico": resposta.get("grafico", ""),
        "justificativa": resposta.get("justificativa", ""),
        "ressalva": resposta.get("ressalva", ""),
        "vegalite_spec": spec,
        "spec_valida": spec_ok,
        "fontes": [resultados[n - 1][0].metadata["fonte"] for n in usados],
        "achados_usados": usados,
        "achados": [
            {"n": i, "fonte": d.metadata["fonte"], "artigo": referencia(d.metadata["fonte"]),
             "similaridade": round(s, 3), "texto": d.page_content}
            for i, (d, s) in enumerate(resultados, 1)
        ],
    }


def ler_json(texto):
    """Extrai o objeto JSON da resposta (o modelo as vezes o envolve em ```json)."""
    inicio, fim = texto.find("{"), texto.rfind("}")
    if inicio == -1:
        raise ValueError(f"resposta sem JSON: {texto[:200]}")
    return json.loads(texto[inicio : fim + 1])


def argumentos(argv=None):
    p = argparse.ArgumentParser(description="Recomenda um grafico para uma pergunta em linguagem comum.")
    p.add_argument("pergunta", nargs="*", help="a pergunta, em linguagem comum")
    p.add_argument("--recuperacao", default="traduzida", choices=cond_mod.RECUPERACAO,
                   help="como montar a consulta ao banco vetorial (padrao: traduzida)")
    p.add_argument("--k", type=int, default=K, help=f"quantos achados recuperar (padrao: {K})")
    p.add_argument("--modelo", default=None, help="id do modelo de chat (padrao: o do .env)")
    p.add_argument("--sem-cache", action="store_true",
                   help="ignora o cache em disco e chama o modelo de novo")
    p.add_argument("--repeticao", type=int, default=1,
                   help="numero desta repeticao; entra na chave do cache (padrao: 1)")
    return p.parse_args(argv)


def main(argv=None):
    a = argumentos(argv)
    cond = cond_mod.Condicao(nome="linha-de-comando", recuperacao=a.recuperacao, k=a.k, modelo=a.modelo)
    pergunta = " ".join(a.pergunta) or "quero comparar as vendas de 5 categorias de produto"

    llm = cache_mod.chat(cond, repeticao=a.repeticao, ativo=not a.sem_cache)
    r = recomendar(pergunta, cond, llm=llm)
    print(f"\nCONDICAO: {cond}")
    if isinstance(llm, cache_mod.ChatComCache):
        print(f"({llm}, repeticao {a.repeticao})")
    print(f"PERGUNTA: {r['pergunta']}")
    print(f"TRADUCAO: tarefa={r['traducao'].get('tarefa')} | dados={r['traducao'].get('tipos_de_dado')}")
    if r.get("fora_de_escopo"):
        print(f"\n{r['mensagem']}")
        return
    print(f"\nRECOMENDACAO: {r['grafico']}")
    print(f"  {r['justificativa']}")
    if r["ressalva"]:
        print(f"  ressalva: {r['ressalva']}")
    print(f"  fontes: {', '.join(r['fontes']) or '(nenhuma: condicao sem recuperacao)'}")
    if r["achados"]:
        print("\nACHADOS RECUPERADOS:")
        for a in r["achados"]:
            print(f"  [{a['n']}] {a['similaridade']:.3f}  {a['fonte']}")


if __name__ == "__main__":
    main()
