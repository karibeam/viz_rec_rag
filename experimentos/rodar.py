"""Executa um experimento: condicoes x perguntas x repeticoes, com diario.

O experimento e um arquivo JSON, nao codigo: e isso que permite re-rodar o
mesmo experimento depois, e dizer na dissertacao exatamente o que foi rodado.

    .venv/bin/python experimentos/rodar.py experimentos/exemplo.json
    .venv/bin/python experimentos/rodar.py experimentos/exemplo.json --repeticoes 3

Cada execucao escreve uma pasta em runs/<nome do experimento>/, e o resumo de
todas vai em runs/<nome>/resumo.jsonl -- uma linha por execucao, para analise.
"""

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

import cache
import condicao as cond_mod
import config
import diario as diario_mod
from indexar import abrir_banco
from recomendar import recomendar


def carregar(caminho) -> dict:
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def perguntas_de(spec) -> list[str]:
    """Lista de perguntas, escrita no proprio JSON ou num arquivo de texto."""
    if isinstance(spec, str):
        linhas = (RAIZ / spec).read_text(encoding="utf-8").splitlines()
        return [l.strip() for l in linhas if l.strip() and not l.startswith("#")]
    return list(spec)


def condicoes_de(spec) -> list[cond_mod.Condicao]:
    return [cond_mod.Condicao(**c) for c in spec]


def precisa_de_banco(condicoes) -> bool:
    return any(c.recuperacao != "nenhuma" for c in condicoes)


def llm_padrao(cond, repeticao, diario):
    """Modelo da condicao, com cache em disco por fora e diario por dentro.

    A ordem importa: o diario envolve o cache, entao um acerto de cache tambem
    e registrado no diario -- a execucao fica completa mesmo sem chamar a API.
    """
    return diario_mod.LlmComDiario(cache.chat(cond, repeticao), diario)


def rodar_uma(cond, pergunta, repeticao=1, raiz=None, banco=None, fabrica=llm_padrao):
    """Uma execucao: devolve (resultado, pasta do diario)."""
    d = diario_mod.Diario(cond, pergunta, repeticao, raiz)
    resultado = recomendar(pergunta, cond, llm=fabrica(cond, repeticao, d), banco=banco)
    d.fechar(resultado)
    return resultado, d.pasta


def linha_de_resumo(cond, pergunta, repeticao, resultado, pasta):
    achados = resultado.get("achados") or []
    artigos = {a["fonte"].split("#")[0] for a in achados}
    return {
        "condicao": cond.nome,
        "identidade": cond.identidade(),
        "recuperacao": cond.recuperacao,
        "k": cond.k,
        "pergunta": pergunta,
        "repeticao": repeticao,
        "fora_de_escopo": bool(resultado.get("fora_de_escopo")),
        "grafico": resultado.get("grafico", ""),
        "spec_valida": resultado.get("spec_valida"),
        "melhor_similaridade": achados[0]["similaridade"] if achados else None,
        "achados_recuperados": len(achados),
        "artigos_distintos": len(artigos),
        "fontes_citadas": len(resultado.get("fontes") or []),
        "pasta": str(pasta) if pasta else None,
    }


def rodar(configuracao, raiz=None, fabrica=llm_padrao, banco=None):
    condicoes = condicoes_de(configuracao["condicoes"])
    perguntas = perguntas_de(configuracao["perguntas"])
    repeticoes = int(configuracao.get("repeticoes", 1))
    nome = configuracao.get("nome", "experimento")

    raiz = Path(raiz) if raiz else config.RUNS_DIR / nome
    raiz.mkdir(parents=True, exist_ok=True)

    if banco is None and precisa_de_banco(condicoes):
        banco = abrir_banco(cache.embeddings())

    total = len(condicoes) * len(perguntas) * repeticoes
    print(f"{nome}: {len(condicoes)} condicoes x {len(perguntas)} perguntas "
          f"x {repeticoes} repeticoes = {total} execucoes")
    print(f"diarios em {raiz}\n")

    linhas, feitas = [], 0
    with (raiz / "resumo.jsonl").open("a", encoding="utf-8") as resumo:
        for cond in condicoes:
            for pergunta in perguntas:
                for repeticao in range(1, repeticoes + 1):
                    feitas += 1
                    rotulo = f"[{feitas}/{total}] {cond.nome} r{repeticao} {pergunta[:44]}"
                    try:
                        resultado, pasta = rodar_uma(cond, pergunta, repeticao, raiz, banco, fabrica)
                        linha = linha_de_resumo(cond, pergunta, repeticao, resultado, pasta)
                        print(f"{rotulo} -> {linha['grafico'] or '(recusada)'}", flush=True)
                    except Exception as exc:  # uma falha nao pode derrubar o experimento
                        linha = {"condicao": cond.nome, "identidade": cond.identidade(),
                                 "pergunta": pergunta, "repeticao": repeticao,
                                 "erro": f"{type(exc).__name__}: {exc}"}
                        print(f"{rotulo} -> ERRO {type(exc).__name__}: {exc}", flush=True)
                    linhas.append(linha)
                    resumo.write(json.dumps(linha, ensure_ascii=False) + "\n")
                    resumo.flush()  # nao perde o que ja rodou se for interrompido
    return linhas


def imprimir_resumo(linhas):
    print(f"\n{'condicao':<16} {'exec':>5} {'erros':>6} {'recusas':>8} "
          f"{'melhor sim.':>12} {'artigos':>8}  grafico mais comum")
    print("-" * 94)
    for nome in dict.fromkeys(l["condicao"] for l in linhas):
        suas = [l for l in linhas if l["condicao"] == nome]
        erros = [l for l in suas if "erro" in l]
        ok = [l for l in suas if "erro" not in l]
        recusas = sum(l["fora_de_escopo"] for l in ok)
        sims = [l["melhor_similaridade"] for l in ok if l["melhor_similaridade"] is not None]
        artigos = [l["artigos_distintos"] for l in ok if l["artigos_distintos"]]
        graficos = [l["grafico"] for l in ok if l["grafico"]]
        comum = max(set(graficos), key=graficos.count) if graficos else "-"
        print(f"{nome[:15]:<16} {len(suas):>5} {len(erros):>6} {recusas:>8} "
              f"{(sum(sims) / len(sims) if sims else 0):>12.3f} "
              f"{(sum(artigos) / len(artigos) if artigos else 0):>8.1f}  {comum[:30]}")
    print("\n'melhor sim.' e 'artigos' sao medias das execucoes sem erro.")
    print("Numeros de uma rodada so: ainda nao sao resultado, sao instrumento funcionando.")


def main(argv=None):
    p = argparse.ArgumentParser(description="Roda um experimento descrito num JSON.")
    p.add_argument("configuracao", help="caminho do JSON do experimento")
    p.add_argument("--repeticoes", type=int, default=None,
                   help="sobrepoe o numero de repeticoes do JSON")
    p.add_argument("--raiz", default=None, help="onde gravar os diarios")
    a = p.parse_args(argv)

    configuracao = carregar(a.configuracao)
    if a.repeticoes:
        configuracao["repeticoes"] = a.repeticoes
    imprimir_resumo(rodar(configuracao, a.raiz))


if __name__ == "__main__":
    main()
