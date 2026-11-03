> **DOCUMENTO HISTORICO — SUPERADO PELA v2. NAO DESCREVE O SISTEMA ATUAL.**
>
> Este foi o plano da primeira versao (v1). Ele propoe fechar o gap semantico
> **reescrevendo a base**: um LLM transformaria cada um dos 240 achados em um
> "card" em linguagem leiga, e um placar em Python decidiria o grafico.
>
> Essa arquitetura foi construida, avaliada e **abandonada**. Ela exigia varias
> camadas de ajuste (pesos, amortecimento por artigo, piso de escopo) e
> deixava a base dependente de uma reescrita por LLM.
>
> A v2 — o sistema que esta neste repositorio — faz o contrario: traduz **a
> pergunta** para o vocabulario da base, em um passo, e mantem os artigos
> intactos. A arquitetura atual esta no [README](../../README.md); a v1 esta
> guardada na tag `v1-cards-placar`.
>
> Guardado aqui porque documenta por que a arquitetura atual e assim.

---

# Plano MVP — Sistema de Recomendação de Visualizações (RAG)

## 0. Diagnóstico do problema de similaridade (respondendo sua pergunta principal)

Fui checar a base real (`Zehua-Zeng/graphical-perception-knowledge`). Exemplo de registro (`Cleveland1984graphical.json`):

```json
{
  "Title": "Graphical Perception: Theory, Experimentation, and Application...",
  "Designs": {
    "E-1": {
      "layers": [{
        "encodings": [
          {"data-type": "quantitative", "channel": "positionY", "scale": "linear"},
          {"data-type": "nominal", "channel": "positionX", "scale": "nominal"}
        ],
        "mark": "area-rect"
      }],
      "note": "Two bars are next to each other."
    }
  },
  "Tasks": ["sort"],
  "Results": {"Experimental": {"accuracy": {"sort-1": {"rank": ["E-1","E-2","E-3"], ...}}}}
}
```

**O problema não é o idioma — é a forma.** A base é dado estruturado de paper acadêmico (chaves como `channel: positionX`, `mark: area-rect`, `Tasks: ["sort"]`, tabelas de significância estatística), não prosa explicando "quando usar gráfico de barras". Uma pergunta de não-especialista tipo *"quero comparar vendas de 5 categorias ao longo do ano"* não tem quase nenhuma sobreposição lexical/semântica com `"channel": "positionX"` — não importa quão bom seja o embedding, cosine similarity vai ficar baixa porque os dois lados do espaço vetorial simplesmente não falam a mesma língua conceitual.

**Traduzir para PT-BR sozinho não resolve isso.** Resolve (parcialmente) o problema de cross-lingual (usuário pergunta em PT, base em EN), mas não resolve o problema de formato/densidade semântica.

### O que realmente resolve: enriquecimento estruturado → narrativo (antes de indexar)

Um passo de pré-processamento, uma vez, offline, usando um LLM para transformar cada registro técnico em um **"card" narrativo** + metadados extraídos:

```json
{
  "text": "Estudo de Cleveland (1984) sobre percepção gráfica. Tarefa: ordenar valores (sort).
   Comparou gráfico de barras agrupadas lado a lado (E-1) vs. barras empilhadas na base (E-2) vs.
   barras empilhadas em dois grupos (E-3). Resultado: E-1 (barras lado a lado) foi
   significativamente mais fácil de ordenar corretamente que as demais variações.
   Indicado para: comparar valores quantitativos entre categorias nominais.",
  "metadata": {
    "task": "sort",
    "data_types": ["quantitative", "nominal"],
    "chart_types": ["grouped-bar", "stacked-bar"],
    "winner": "grouped-bar",
    "source": "Cleveland1984graphical.json#E-1"
  }
}
```

Isso ataca o problema na raiz: agora o texto indexado descreve **o que o gráfico faz e para que serve**, na mesma "linguagem" que um usuário leigo usaria pra perguntar. Ganhos adicionais, em ordem de impacto:

1. **Chunking semântico, não por caracteres** — 1 chunk = 1 "Design"/finding do JSON, não um corte arbitrário de texto.
2. **Metadata filtering + busca vetorial (hybrid search)** — filtrar por `data_types`/`task` antes ou depois da busca por similaridade, reduzindo ruído.
3. **Embedding multilíngue** (`text-embedding-004` do Gemini ou `text-embedding-3-small/large` da OpenAI) — cobre o cross-lingual sem precisar traduzir manualmente cada registro.
4. **HyDE (Hypothetical Document Embeddings)** — antes de buscar, o LLM expande a pergunta do usuário gerando uma resposta técnica hipotética, e é essa resposta hipotética que vira o vetor de busca. Aproxima o espaço vetorial da pergunta ao espaço vetorial dos documentos técnicos.
5. **Re-ranking** (cross-encoder, ex. Cohere Rerank ou `ms-marco-MiniLM` local) sobre o top-k inicial, se a precisão do passo 1-4 ainda não for suficiente.

Priorize nessa ordem: **(1) e (2) resolvem ~80% do problema pelo custo mais baixo**; HyDE e re-rank são otimizações de segunda rodada, só valem a pena se depois de medir a qualidade ainda faltar precisão.

---

## 1. Arquitetura do pipeline (MVP)

**Fase A — Preparação da base (offline, roda 1x ou quando a base mudar)**

```
JSONs brutos (59 arquivos) 
  → enrich.py (LLM reescreve cada Design/finding em card narrativo + metadata)
  → chunks narrativos (1 por finding/design, não por tamanho de texto)
  → embeddings (multilíngue)
  → Chroma (vector store local)
```

**Fase B — Retrieval & Generation (runtime, por pergunta)**

```
Pergunta do usuário (PT ou EN)
  → [opcional] HyDE: LLM expande a pergunta em termos técnicos
  → embedding da pergunta (mesmo modelo)
  → busca por similaridade de cosseno + metadata filter opcional
  → top-k chunks
  → [opcional] re-rank
  → LLM gera JSON estruturado: {principal, alternativa} com spec Vega-Lite
  → validação do schema Vega-Lite (rejeita/repara specs inválidas)
  → resposta final + fontes citadas
```

Isso é exatamente os dois diagramas que você mandou, com um bloco novo entre "Documentos" e "Split into chunks": **enriquecimento via LLM**.

---

## 2. Stack recomendada (grátis / baixo custo, adequada a MVP)

| Componente | Recomendação | Por quê |
|---|---|---|
| Orquestração | LangChain (`langchain`, `langchain-community`) | Você já pediu, e tem integração pronta pra Chroma/Gemini/OpenAI |
| Embeddings | Gemini `text-embedding-004` (free tier generoso) **ou** OpenAI `text-embedding-3-small` | Ambos multilíngues, resolvem PT↔EN sem tradução manual |
| Vector store | **Chroma** (local, embutido, sem infra) | Ideal pra MVP; zero setup de servidor |
| LLM generativo | Gemini 2.0/2.5 Flash (free tier) ou GPT-4o-mini | Barato/grátis, latência baixa, suficiente pra gerar JSON estruturado |
| Enriquecimento da base | Mesmo LLM generativo, rodando 1x sobre os 59 JSONs (~59 chamadas, custo desprezível) | |
| Validação do output | `altair` ou schema JSON do Vega-Lite | Garante que a spec gerada é renderizável |
| Interface | **Streamlit** | Ver seção 4 |

Alternativa 100% gratuita sem API key: `sentence-transformers` multilíngue local (ex. `paraphrase-multilingual-mpnet-base-v2`) — funciona, mas a qualidade de retrieval em texto técnico é sensivelmente pior que Gemini/OpenAI embeddings. Para MVP eu recomendaria começar com Gemini (free tier cobre tranquilamente o volume de 59 documentos + testes).

---

## 3. Estrutura de projeto sugerida

```
viz_rec_rag/
├── data/
│   ├── raw/                    # os 59 JSONs originais do repo
│   └── processed/              # cards.jsonl — saída do enriquecimento, pronta p/ indexar
├── src/
│   ├── ingest.py                # baixa/lê os JSONs brutos
│   ├── enrich.py                # LLM: JSON técnico -> card narrativo + metadata (roda 1x)
│   ├── build_index.py           # embeddings + persiste no Chroma
│   ├── retrieve.py              # busca vetorial (+ HyDE/rerank opcionais)
│   ├── recommend.py             # prompt final -> {principal, alternativa} em Vega-Lite
│   └── validate_spec.py         # valida a spec Vega-Lite gerada
├── app.py                       # interface Streamlit
├── tests/
│   └── test_questions.md        # ~15 perguntas reais de não-especialista p/ avaliação manual
├── .env.example                 # GOOGLE_API_KEY= / OPENAI_API_KEY=
├── requirements.txt
└── README.md
```

Simples, linear, sem microserviços — dá pra rodar tudo localmente com `python src/build_index.py` (indexação) e depois `streamlit run app.py` (uso).

---

## 4. Markdown ou interface?

Para o **plano** (este documento): markdown é o formato certo — é o que você está lendo.

Para o **MVP em si**: recomendo **Streamlit**, não markdown puro. Motivo direto:
- Você quer testar "prompt + recomendação principal + alternativa" de forma repetível — um app com campo de texto e output formatado é mais rápido de iterar visualmente do que reler um arquivo `.md` gerado a cada rodada.
- Streamlit renderiza o Vega-Lite spec nativamente (`st.vega_lite_chart`) — você vê o gráfico recomendado de verdade, não só o JSON da spec. Isso é validação visual imediata de que a recomendação faz sentido.
- Ainda é MVP: um único arquivo `app.py`, sem build step, sem deploy complexo.

Um notebook Jupyter/Colab é aceitável como *primeiro rascunho* (validar o pipeline de enriquecimento + retrieval antes de ter interface), mas eu não entregaria o MVP final assim — o Streamlit custa pouco a mais e already é o "produto" que dá pra mostrar pra alguém de fora.

---

## 5. Formato de saída esperado

```json
{
  "pergunta": "Quero comparar as vendas de 5 categorias de produto ao longo do ano",
  "principal": {
    "grafico": "Gráfico de barras agrupadas",
    "justificativa": "Estudos de percepção gráfica (Cleveland 1984) mostram que comparar valores quantitativos entre categorias nominais é feito com mais precisão com posição/comprimento (barras) do que com outros canais.",
    "vegalite_spec": { "...": "..." }
  },
  "alternativa": {
    "grafico": "Gráfico de linhas",
    "justificativa": "Se o eixo temporal (ano) for o foco principal em vez da comparação entre categorias, linhas comunicam melhor a tendência.",
    "vegalite_spec": { "...": "..." }
  },
  "fontes": ["Cleveland1984graphical.json#E-1", "Cleveland1984graphical.json#sort-1"]
}
```

Citar a fonte (`fontes`) é importante para um sistema baseado em RAG (não em regras) — dá rastreabilidade e permite auditar se a recomendação realmente veio da base ou foi alucinada.

---

## 6. Roteiro de execução (MVP em ~1–2 semanas)

1. Baixar os 59 JSONs (`data/raw/`)
2. Escrever o prompt de enriquecimento e rodar 1x sobre toda a base → `data/processed/cards.jsonl`
3. Indexar no Chroma (`build_index.py`)
4. Montar o RAG básico via script/CLI primeiro (sem interface) — mais rápido de depurar
5. Rodar 10–15 perguntas reais de não-especialista, avaliar manualmente a qualidade da recuperação e da recomendação
6. Ajustar: tamanho do chunk, k (nº de chunks recuperados), necessidade de HyDE/rerank, prompt de geração
7. Envolver em Streamlit
8. (Opcional, pós-MVP) avaliação automatizada com RAGAS (métricas de faithfulness/context relevance)

---

## 7. O que faltava no seu pipeline original

- **Etapa de enriquecimento/normalização da base antes dos embeddings** — é o ponto crítico que resolve o problema de similaridade que você descreveu.
- **Estratégia de chunking específica** — por finding/design, não por corte de caracteres.
- **Validação de schema** da spec Vega-Lite gerada pelo LLM (evita devolver JSON quebrado/não renderizável).
- **Metadata filtering / hybrid search**, não só busca vetorial pura.
- **Conjunto de perguntas de teste** para avaliar a qualidade do sistema de forma repetível.
- **Citação de fontes** na resposta final, para rastreabilidade (importante ao afirmar "é RAG, não regras").
