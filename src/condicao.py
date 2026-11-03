"""A condicao de um experimento: tudo o que pode variar de uma execucao para outra.

Cada decisao de projeto que antes era constante no codigo entra aqui como
parametro, para poder ser variada, registrada e reproduzida. Rodar o sistema
"como ele e hoje" e rodar com a condicao PADRAO.

Nao existe semente aleatoria aqui, de proposito: o modelo de chat usado por
padrao (gemini-3.5-flash-lite) usa amostragem fixa, ignora `temperature` e nao
aceita semente. Determinismo nao se obtem pedindo ao modelo -- se obtem pelo
cache em disco (src/cache.py), que guarda a resposta de cada prompt.
"""

import hashlib
import json
from dataclasses import asdict, dataclass

# Como a consulta ao banco vetorial e montada. A hipotese central do trabalho
# e que "traduzida" recupera evidencia melhor que "crua"; "hyde" e a
# alternativa da literatura; "nenhuma" e a ablacao que isola o RAG.
RECUPERACAO = (
    "traduzida",  # o LLM traduz a pergunta para o vocabulario da base (atual)
    "crua",       # a pergunta do usuario, como ela veio, vira a consulta
    "hyde",       # o LLM escreve um achado hipotetico, e ELE vira a consulta
    "nenhuma",    # nao busca nada: o LLM responde sem evidencia nenhuma
)


@dataclass(frozen=True)
class Condicao:
    """Uma condicao experimental. Imutavel: serve de rotulo do que foi rodado."""

    nome: str = "padrao"             # rotulo livre, so para leitura humana
    recuperacao: str = "traduzida"
    k: int = 9                       # quantos achados recuperar
    modelo: str | None = None        # None = o modelo do .env

    def __post_init__(self):
        if self.recuperacao not in RECUPERACAO:
            raise ValueError(
                f"recuperacao {self.recuperacao!r} nao existe; use uma de {RECUPERACAO}"
            )
        if self.recuperacao != "nenhuma" and self.k < 1:
            raise ValueError(f"k deve ser >= 1 (recebi {self.k})")

    def como_dict(self):
        return asdict(self)

    def identidade(self) -> str:
        """Hash curto dos PARAMETROS, para nomear a pasta da execucao.

        O `nome` fica fora: trocar o rotulo nao muda o que foi rodado, e duas
        condicoes com os mesmos parametros devem cair na mesma identidade.
        """
        parametros = {c: v for c, v in sorted(self.como_dict().items()) if c != "nome"}
        bruto = json.dumps(parametros, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(bruto).hexdigest()[:8]

    def __str__(self):
        modelo = self.modelo or "modelo-do-.env"
        return f"{self.nome} [recuperacao={self.recuperacao} k={self.k} {modelo}]"


PADRAO = Condicao()  # o sistema como ele e hoje
