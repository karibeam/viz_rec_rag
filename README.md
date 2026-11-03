# Recomendador de Visualizações via RAG

Um sistema que recomenda **um gráfico** para quem **não é especialista**, a partir de
uma pergunta em linguagem do dia a dia. A recomendação vem de estudos científicos
sobre percepção de gráficos ([Zeng et al., CHI 2023](https://github.com/Zehua-Zeng/graphical-perception-knowledge)),
recuperados por RAG — não de regras fixas.

## O problema: a base e o usuário falam línguas diferentes

A base descreve os estudos em notação técnica:

```
task: characterize-distribution
result, best to worst: (E-4 = E-5 = E-6) > (E-1 = E-2 = E-3) > ...
- E-4: mark area-rect; quantitative -> length; nominal -> positionX
```

O usuário pergunta *"quero mostrar a distribuição das idades dos meus clientes"*.
Os dois textos não têm palavras nem conceitos em comum na superfície, então a
similaridade de cosseno entre eles não significa muita coisa.

## A solução: traduzir a pergunta, não a base

Em vez de reescrever os 240 achados da base, o sistema traduz **a pergunta** para o
vocabulário técnico da base, e só então busca:

```
pergunta leiga:     "quero mostrar a distribuição das idades dos meus clientes"
        ↓ LLM classifica (não recomenda nada)
consulta técnica:   task: characterize-distribution
                    data types: quantitative
                    analysis: show how a quantitative variable is distributed
```

Agora pergunta e documentos usam os mesmos termos, e o cosseno volta a medir o que
interessa. A base fica **intacta**: nenhum artigo é reescrito por LLM.

## Pipeline

![Pipeline do sistema](docs/pipeline.svg)

O mesmo pipeline em texto, para quem for editar este arquivo:

```mermaid
flowchart TD
    subgraph OFF["ETAPA OFFLINE · roda uma vez · src/indexar.py"]
        A["59 artigos (JSON)"] --> B["240 achados<br/>1 achado = 1 documento"]
        B --> C["template fixo, sem LLM<br/>task / data types / result / designs"]
        C --> D["embeddings"]
        D --> E[("Chroma<br/>banco vetorial")]
    end

    subgraph ON["ETAPA ONLINE · a cada pergunta · src/recomendar.py"]
        P["pergunta do usuário"] --> T["1 · TRADUZIR<br/>LLM classifica: tarefa + tipos de dado"]
        T -->|"não é sobre dados"| X["recusa<br/>fora do escopo"]
        T --> Q["consulta técnica<br/>mesmo formato dos documentos"]
        Q --> S["2 · BUSCAR<br/>similaridade de cosseno · top 9"]
        E --> S
        S --> G["3 · GERAR<br/>LLM lê os 9 achados e recomenda UM gráfico"]
        G --> V["valida a spec Vega-Lite"]
        V --> R["gráfico + justificativa + artigos citados"]
    end
```

**Custo por pergunta:** 2 chamadas de LLM (traduzir + gerar) e 1 embedding.
Pergunta fora do escopo: 1 chamada de LLM, sem busca.

### Quando o sistema recusa

A tradução decide duas coisas, nesta ordem:

1. **A pergunta é sobre mostrar ou analisar dados?** Se não for (ex.: "qual a cor da
   capa do meu relatório?", "qual biblioteca JavaScript usar?"), o sistema recusa
   sem buscar nada.
2. **Se for, qual das 10 tarefas da base é a mais próxima?** O LLM escolhe sempre a
   mais próxima, mesmo sem encaixe perfeito. Uma pergunta sobre dados nunca é
   recusada só porque a base não tem uma tarefa exata para ela.

## Interface

A tela segue as heurísticas de usabilidade de Nielsen, sem enfeites:

- **linguagem do usuário:** a tarefa e os tipos de dado aparecem traduzidos
  ("ver como os valores se distribuem", "números"), nunca o termo técnico;
- **reconhecer em vez de lembrar:** exemplos clicáveis logo abaixo da pergunta;
- **minimalismo:** a recomendação fica num cartão (gráfico de exemplo,
  justificativa, dica); os detalhes técnicos ficam recolhidos em
  "Como o sistema chegou a essa recomendação";
- **fontes verificáveis:** cada achado mostra autor, ano e título do artigo, com
  link de busca no Google Scholar;
- **erros compreensíveis:** mensagens em português (ex.: limite de uso da API),
  com o erro técnico recolhido.

## Arquivos

```
src/
  config.py         chave de API, modelos, nova tentativa em caso de erro temporário
  indexar.py        etapa offline: base → documentos → embeddings → Chroma
  recomendar.py     etapa online: traduzir → buscar → gerar
  validate_spec.py  confere e conserta a spec Vega-Lite (tipos faltantes;
                    categorias ordenadas, como meses, mantêm a ordem dos dados)
app.py              interface (Streamlit)
mostrar_traducao.py log passo a passo da tradução (para documentar o sistema)
docs/pipeline.svg   diagrama do pipeline
docs/historico/     plano da v1, superado — documenta por que a v2 é assim
.streamlit/         tema da interface (cor principal, barra de ferramentas mínima)
data/raw/           os 59 JSONs da base, intactos
requirements.txt      dependências do projeto (faixas abertas)
requirements.lock.txt versões exatas do ambiente em que o sistema foi validado
tests/
  teste_unidades.py     testa isoladamente validate_spec.py e os ajudantes de indexar.py
  teste_offline.py      testa o pipeline inteiro sem gastar API
  comum.py              ajudante compartilhado pelos testes (OK/FALHA)
  comparar_v1_v2.py     compara esta versão com a v1 nas 18 perguntas
  test_questions.md     perguntas de avaliação + gabarito aproximado
  resultados_v1.json    respostas da v1, para a comparação
```

## Como rodar

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Preencha `GOOGLE_API_KEY` no `.env` (chave em https://aistudio.google.com/apikey).
Uma chave só cobre chat e embeddings.

```bash
.venv/bin/python src/indexar.py
```

Roda uma vez, leva uns 6 minutos (o plano gratuito aceita ~100 embeddings por minuto).

```bash
.venv/bin/streamlit run app.py
```

Ou pelo terminal:

```bash
.venv/bin/python src/recomendar.py "quero comparar as vendas de 5 categorias"
```

## Testes

Nenhum dos dois gasta chamada de API — dá para conferir o sistema sem chave:

```bash
.venv/bin/python tests/teste_unidades.py
```

```bash
.venv/bin/python tests/teste_offline.py
```

O primeiro cobre as partes com mais lógica: a validação e o reparo da spec
Vega-Lite (`type` faltante, categorias ordenadas) e a leitura do formato da base
(rankings com empate aninhado, nome da tarefa, template do documento indexado).
O segundo encaixa o pipeline inteiro com modelos falsos: base → documentos →
banco vetorial → tradução → busca → recomendação → spec, incluindo a recusa de
pergunta fora do escopo.

Para reproduzir o ambiente exato em que foram validados:

```bash
.venv/bin/pip install -r requirements.lock.txt
```

## Limitações conhecidas

- **Séries temporais.** A base não tem o tipo de dado "temporal" nem uma tarefa de
  tendência, e os estudos com séries temporais comparam variações de gráficos de
  linhas entre si, não linhas contra barras. Por isso, perguntas como "evolução
  da temperatura em 12 meses" às vezes recebem barras. Estudos que comparam linhas
  e barras (ex.: Zacks & Tversky, 1999) foram excluídos da revisão de Zeng et al.;
  estendê-la com eles é o próximo passo planejado.
- **Variação do LLM — o sistema não é determinístico.** `src/config.py` pede
  `temperature=0`, mas o `gemini-3.5-flash-lite` usa amostragem fixa e **ignora
  esse pedido**; ele mesmo avisa, ao ser chamado: *"uses fixed sampling defaults;
  the sampling parameter(s) temperature will be ignored"*. Por isso a mesma
  pergunta pode receber recomendações diferentes em execuções diferentes,
  principalmente quando a base cobre mal o caso. Trocar para um provedor que
  respeite a temperatura (`LLM_PROVIDER=openai`) reduz a variação, mas o passo de
  geração continua sendo amostragem — não um cálculo.
- **Referências.** Os JSONs da base trazem só o título; autor e ano vêm do nome do
  arquivo (`Saket2018task.json` → "Saket (2018)"), sem os coautores.

## Versão anterior (v1)

A primeira versão está guardada na branch
[`v1-cards-placar`](https://github.com/karibeam/viz_rec_rag/tree/v1-cards-placar).
Ela resolvia o gap pelo outro lado: um LLM reescrevia os 240 achados em "cards" com
linguagem leiga, e um placar em Python decidia o gráfico. Funcionava, mas exigia
várias camadas extras (pesos, amortecimento por artigo, piso de escopo). A v2 troca
isso por um passo só: traduzir a pergunta.
