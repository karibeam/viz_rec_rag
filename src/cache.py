"""Cache em disco das respostas do LLM: um experimento vira re-executavel.

Por que isso e necessario, e nao so uma economia de cota: o modelo de chat
usado por padrao ignora `temperature` e nao aceita semente (ver config.get_chat).
Nao da para pedir determinismo ao modelo -- entao ele se obtem aqui, guardando
a resposta de cada prompt. Um resultado obtido hoje pode ser reproduzido em
setembro a partir do mesmo cache, sem gastar chamada e sem variacao nova.

A CHAVE INCLUI O NUMERO DA REPETICAO, e isso e deliberado. Medir a variacao do
sistema exige rodar a mesma pergunta n vezes; se a chave fosse so (modelo,
prompt), a repeticao 2 leria a resposta da repeticao 1 e a variacao medida
seria zero -- um experimento silenciosamente errado. Com a repeticao na chave,
as n respostas sao distintas E ficam todas guardadas: a variacao e medida de
verdade na primeira vez e reproduzida igual nas vezes seguintes.

    python src/cache.py          # o que esta guardado
"""

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from langchain_core.messages import AIMessage


def pasta_padrao() -> Path:
    return config.CACHE_DIR


def chave(modelo: str, prompt: str, repeticao: int = 1) -> str:
    """Identificador de uma chamada. Muda se o modelo, o prompt ou a repeticao mudarem."""
    bruto = f"{modelo}\x00{repeticao}\x00{prompt}".encode("utf-8")
    return hashlib.sha256(bruto).hexdigest()


def caminho(modelo, prompt, repeticao=1, pasta=None) -> Path:
    return (pasta or pasta_padrao()) / f"{chave(modelo, prompt, repeticao)}.json"


def ler(modelo, prompt, repeticao=1, pasta=None):
    """A resposta guardada, ou None se esta chamada nunca foi feita."""
    arquivo = caminho(modelo, prompt, repeticao, pasta)
    if not arquivo.exists():
        return None
    try:
        return json.loads(arquivo.read_text(encoding="utf-8"))["resposta"]
    except (ValueError, KeyError, OSError):
        return None  # arquivo truncado ou corrompido: trata como ausente


def gravar(modelo, prompt, resposta, repeticao=1, pasta=None) -> Path:
    """Guarda a resposta junto do prompt que a produziu.

    O prompt vai inteiro no arquivo de proposito: um cache que nao diz o que
    foi perguntado nao serve para auditar um experimento depois.
    """
    arquivo = caminho(modelo, prompt, repeticao, pasta)
    arquivo.parent.mkdir(parents=True, exist_ok=True)
    conteudo = {
        "modelo": modelo,
        "repeticao": repeticao,
        "quando": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt": prompt,
        "resposta": resposta,
    }
    # escreve ao lado e renomeia: uma interrupcao no meio nao deixa cache meio gravado
    temporario = arquivo.with_suffix(".parcial")
    temporario.write_text(json.dumps(conteudo, ensure_ascii=False, indent=1), encoding="utf-8")
    temporario.replace(arquivo)
    return arquivo


class ChatComCache:
    """Envolve um modelo de chat mantendo a interface `.invoke(prompt)`.

    Quem chama nao precisa saber que existe cache: recomendar.py continua
    fazendo llm.invoke(prompt) e config.com_retentativa continua funcionando
    (um acerto no cache nao chama a API, logo nao pode falhar por limite).
    """

    def __init__(self, llm, modelo: str, repeticao: int = 1, pasta=None):
        self.llm = llm
        self.modelo = modelo
        self.repeticao = repeticao
        self.pasta = pasta
        self.acertos = 0
        self.faltas = 0

    def invoke(self, prompt):
        guardada = ler(self.modelo, prompt, self.repeticao, self.pasta)
        if guardada is not None:
            self.acertos += 1
            return AIMessage(content=guardada)
        self.faltas += 1
        texto = config.texto(self.llm.invoke(prompt))
        gravar(self.modelo, prompt, texto, self.repeticao, self.pasta)
        return AIMessage(content=texto)

    def __str__(self):
        return f"cache: {self.acertos} acertos, {self.faltas} chamadas novas"


def chat(cond, repeticao: int = 1, ativo: bool = True, pasta=None):
    """O modelo de chat de uma condicao, com cache (ou sem, se ativo=False)."""
    llm = config.get_chat(cond.modelo)
    if not ativo:
        return llm
    return ChatComCache(llm, config.nome_do_chat(cond.modelo), repeticao, pasta)


# --- embeddings -------------------------------------------------------------
#
# O vetor de um texto nao muda nunca para o mesmo modelo, entao cachear
# embedding e puro ganho: a varredura de k (6 valores) vetoriza a MESMA
# consulta 6 vezes, e o plano gratuito aceita ~100 embeddings por minuto.


def pasta_de_embeddings(pasta=None) -> Path:
    return (pasta or pasta_padrao()) / "emb"


class EmbeddingsComCache:
    """Envolve um modelo de embeddings, mantendo a interface que o Chroma usa."""

    def __init__(self, embeddings, modelo: str, pasta=None):
        self.embeddings = embeddings
        self.modelo = modelo
        self.pasta = pasta_de_embeddings(pasta)
        self.acertos = 0
        self.faltas = 0

    # O PAPEL entra na chave porque o mesmo texto tem DOIS vetores diferentes:
    # o provedor vetoriza consulta e documento com task_type diferente
    # (RETRIEVAL_QUERY x RETRIEVAL_DOCUMENT). Confundir os dois muda os achados
    # recuperados -- um erro silencioso, que nao levanta excecao nenhuma.
    def _arquivo(self, texto, papel) -> Path:
        return self.pasta / f"{chave(f'{self.modelo}:{papel}', texto)}.json"

    def _guardado(self, texto, papel):
        arquivo = self._arquivo(texto, papel)
        if not arquivo.exists():
            return None
        try:
            return json.loads(arquivo.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None

    def _guardar(self, texto, papel, vetor):
        self.pasta.mkdir(parents=True, exist_ok=True)
        temporario = self._arquivo(texto, papel).with_suffix(".parcial")
        temporario.write_text(json.dumps(vetor), encoding="utf-8")
        temporario.replace(self._arquivo(texto, papel))

    def embed_query(self, texto):
        """Vetor de uma CONSULTA. Tem que passar pelo embed_query do provedor."""
        guardado = self._guardado(texto, "consulta")
        if guardado is not None:
            self.acertos += 1
            return guardado
        self.faltas += 1
        vetor = self.embeddings.embed_query(texto)
        self._guardar(texto, "consulta", vetor)
        return vetor

    def embed_documents(self, textos):
        """Vetores de DOCUMENTOS, chamando a API so para os que faltam."""
        vetores = [self._guardado(texto, "documento") for texto in textos]
        faltando = [i for i, v in enumerate(vetores) if v is None]
        self.acertos += len(textos) - len(faltando)
        self.faltas += len(faltando)
        if faltando:
            novos = self.embeddings.embed_documents([textos[i] for i in faltando])
            for i, vetor in zip(faltando, novos):
                self._guardar(textos[i], "documento", vetor)
                vetores[i] = vetor
        return vetores

    def __str__(self):
        return f"embeddings: {self.acertos} acertos, {self.faltas} novos"


def embeddings(ativo: bool = True, pasta=None):
    """O modelo de embeddings, com cache em disco (ou sem, se ativo=False)."""
    emb = config.get_embeddings()
    if not ativo:
        return emb
    return EmbeddingsComCache(emb, config.nome_do_embedding(), pasta)


def resumo(pasta=None):
    """Quantas chamadas estao guardadas, por modelo."""
    pasta = pasta or pasta_padrao()
    por_modelo, bytes_totais = {}, 0
    for arquivo in sorted(pasta.glob("*.json")) if pasta.exists() else []:
        bytes_totais += arquivo.stat().st_size
        try:
            modelo = json.loads(arquivo.read_text(encoding="utf-8")).get("modelo", "?")
        except (ValueError, OSError):
            modelo = "(arquivo ilegivel)"
        por_modelo[modelo] = por_modelo.get(modelo, 0) + 1
    return por_modelo, bytes_totais


if __name__ == "__main__":
    por_modelo, bytes_totais = resumo()
    print(f"cache em {pasta_padrao()}")
    if not por_modelo:
        print("  (nenhuma chamada de chat guardada)")
    for modelo, n in sorted(por_modelo.items(), key=lambda x: -x[1]):
        print(f"  {n:5d} chamadas de chat  {modelo}")
    emb = pasta_de_embeddings()
    vetores = list(emb.glob("*.json")) if emb.exists() else []
    bytes_emb = sum(a.stat().st_size for a in vetores)
    print(f"  {len(vetores):5d} embeddings guardados")
    print(f"  {(bytes_totais + bytes_emb) / 1024 / 1024:.1f} MB no total")
