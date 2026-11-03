"""Diario de uma execucao: uma pasta com tudo o que foi feito e por que.

Um resultado que nao diz como foi produzido nao serve para pesquisa. Cada
execucao escreve em runs/<quando>-<identidade da condicao>-r<repeticao>/:

    ambiente.json   versao do Python, commit do codigo, modelos resolvidos
    condicao.json   os parametros usados
    pergunta.txt    a pergunta, como ela veio
    chamadas/       prompt e resposta de CADA chamada ao LLM, em ordem
    achados.json    os achados recuperados, com similaridade e fonte
    resultado.json  o resultado inteiro

O diario registra as chamadas envolvendo o modelo (LlmComDiario), do mesmo jeito
que o cache: recomendar.py nao precisa saber que esta sendo registrado.
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config


def commit_do_codigo():
    """O commit em que o codigo esta, e se havia alteracao nao commitada.

    Um resultado produzido com a arvore suja nao e reproduzivel a partir do
    commit, e o diario tem que dizer isso em vez de deixar parecer que e.
    """
    def git(*args):
        pronto = subprocess.run(["git", "-C", str(config.ROOT), *args],
                                capture_output=True, text=True)
        return pronto.stdout.strip() if pronto.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "codigo_sujo": None}
    return {"commit": commit, "codigo_sujo": bool(git("status", "--porcelain"))}


def pasta_livre(raiz: Path, cond, pergunta: str, repeticao: int) -> Path:
    """Uma pasta por execucao, sem nunca escrever em cima de outra.

    O nome leva tambem um hash da pergunta: com segundo de resolucao, duas
    perguntas da mesma condicao rodadas no mesmo segundo cairiam na mesma pasta
    e a segunda sobrescreveria a primeira.
    """
    quando = datetime.now().strftime("%Y%m%d-%H%M%S")
    marca = hashlib.sha256(pergunta.encode("utf-8")).hexdigest()[:6]
    base = f"{quando}-{cond.identidade()}-{marca}-r{repeticao}"
    pasta, n = raiz / base, 1
    while pasta.exists():
        n += 1
        pasta = raiz / f"{base}-{n}"
    return pasta


class Diario:
    """Grava uma execucao numa pasta propria."""

    def __init__(self, cond, pergunta, repeticao=1, raiz=None):
        self.cond = cond
        self.pergunta = pergunta
        self.repeticao = repeticao
        raiz = Path(raiz) if raiz else config.RUNS_DIR
        self.pasta = pasta_livre(raiz, cond, pergunta, repeticao)
        self.pasta.mkdir(parents=True, exist_ok=True)

    def registrar(self, nome, conteudo):
        """Escreve um arquivo na pasta da execucao. Objeto vira JSON; texto, texto."""
        arquivo = self.pasta / nome
        arquivo.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(conteudo, str):
            arquivo.write_text(conteudo, encoding="utf-8")
        else:
            arquivo.write_text(json.dumps(conteudo, ensure_ascii=False, indent=1), encoding="utf-8")
        return arquivo

    def fechar(self, resultado):
        """Grava o resultado e o contexto em que ele foi produzido."""
        self.registrar("ambiente.json", {
            "quando": datetime.now().isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "modelo_de_chat": config.nome_do_chat(self.cond.modelo),
            "modelo_de_embedding": config.nome_do_embedding(),
            "provedor": config.provider(),
            "repeticao": self.repeticao,
            **commit_do_codigo(),
        })
        self.registrar("condicao.json", self.cond.como_dict())
        self.registrar("pergunta.txt", self.pergunta)
        self.registrar("achados.json", resultado.get("achados") or [])
        self.registrar("resultado.json", resultado)
        return self.pasta


class DiarioNulo:
    """Diario que nao grava nada. Deixa o codigo chamador sem `if diario:`."""

    pasta = None

    def registrar(self, nome, conteudo):
        return None

    def fechar(self, resultado):
        return None


NULO = DiarioNulo()


class LlmComDiario:
    """Envolve um modelo de chat registrando prompt e resposta de cada chamada.

    As chamadas sao numeradas na ordem em que acontecem, que e sempre a mesma:
    1 traducao, 2 geracao -- ou 1 traducao, 2 achado hipotetico, 3 geracao na
    condicao hyde. Numerar em vez de rotular evita rotulo errado se o pipeline
    mudar de ordem.
    """

    def __init__(self, llm, diario=NULO):
        self.llm = llm
        self.diario = diario
        self.chamadas = 0

    def invoke(self, prompt):
        resposta = self.llm.invoke(prompt)
        self.chamadas += 1
        self.diario.registrar(f"chamadas/{self.chamadas:02d}-prompt.txt", prompt)
        self.diario.registrar(f"chamadas/{self.chamadas:02d}-resposta.txt", config.texto(resposta))
        return resposta

    def __str__(self):
        return f"{self.chamadas} chamadas ao modelo"
