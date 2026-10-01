# Adaptações do TransferNet neste projeto

Este documento registra o que foi efetivamente modificado em relação ao repositório
original ([shijx12/TransferNet](https://github.com/shijx12/TransferNet), disponível
localmente em `methods/_original_reference/TransferNet/` para comparação direta).
Gerado por análise de diff real (ignorando diferenças de fim de linha CRLF/LF,
que por si só não representam mudança de conteúdo).

## Resumo executivo

O objetivo da adaptação foi permitir que o TransferNet, além de prever a entidade-resposta,
**exponha o caminho de raciocínio (triplas do KG) que sustentou essa previsão**, para que esse
caminho pudesse ser passado a um LLM como módulo final de interpretação (ver `Modulo2/`).
Isso exigiu primeiro unificar a arquitetura dos três datasets (MetaQA, WebQSP, CompWebQ) num
único encoder baseado em BERT, e só depois acoplar a extração de caminho.

Em paralelo, `RelationSupervision/` explora um objetivo de treino diferente: em vez de só
extrair o caminho que já sustentou uma previsão de entidade, supervisiona diretamente a
classificação de relação em cada hop contra um caminho-gabarito conhecido (ver tabela abaixo).

## Pastas e o que cada uma é

| Pasta | Status | Descrição |
|---|---|---|
| `CompWebQ/` | **Não modificada** — cópia idêntica do original | O TransferNet original já usa BERT + ensemble de "ways" (`num_ways`) nesse dataset. Serviu de base arquitetural para a generalização. |
| `WebQSP/` | **Não modificada** — cópia idêntica do original | Já usava BERT (`AutoModel`) no repositório original, mas sem o ensemble `num_ways` que o CompWebQ tem. |
| `MetaQA-KB/` (só existe no original) | — | Versão original usa GloVe + BiGRU (sem BERT) e `torch.softmax` no classificador de relação — arquitetura mais antiga, específica para MetaQA. |
| `AtualizacaoMetaQA/` | **Modificada — estágio intermediário** | Porta a arquitetura do `WebQSP/model.py` (BERT + `torch.sigmoid`) para os dados do MetaQA, substituindo o pipeline de vocabulário/GloVe por `AutoTokenizer` (`data.py` reescrito). Mudança real no `model.py` é mínima: `num_steps` default 2→3 (MetaQA tem perguntas de até 3 hops) — o resto é idêntico ao `WebQSP/model.py`. Contém 4 variantes de `predict_paths*.py` (`_inicial`, principal, `_analisar`, `_dando_errado`) que são iterações de debug da extração de caminho, não versões alternativas em uso — candidatas a arquivar/remover quando o `Geral/` for consolidado como a versão definitiva. |
| `AtualizacaoWebQSP/` | **Não modificada** | Idêntica byte-a-byte ao `WebQSP/`, aparentemente só reorganizada/copiada durante o processo, sem alteração de conteúdo. Candidata a remover (redundante com `WebQSP/`). |
| `Geral/` | **Modificada — versão atual/recomendada** | Parte da arquitetura do `CompWebQ/` (BERT + `num_ways`, já existente no original) e adiciona a extração de caminho (`return_paths=True`): a cada hop, filtra as top-5 relações candidatas, cruza com as entidades ativas (`last_e > 0.1`) e reconstrói as triplas `(head, relation, tail)` do KG que sustentam aquele hop, mantendo as top-5 por score. Essa é a única mudança de arquitetura genuinamente nova em relação a qualquer versão do repositório original. `predict.py` foi reescrito para chamar `model(..., return_paths=True)` e salvar `predicted_paths.jsonl` (`{questions, paths}`) em vez de só calcular acurácia. |
| `Modulo2/rog_llama_predict.py` | **Novo — sem equivalente no original** | Módulo final de interpretação via LLM. Carrega um checkpoint causal genérico (`AutoModelForCausalLM.from_pretrained(args.model_path)`), monta um prompt a partir dos `hop_N` extraídos pelo `Geral/`, gera a resposta, faz parsing (JSON ou fallback por linha/vírgula) e calcula Hits@1/Precision/Recall/F1 + tempo de inferência. |
| `Dataset/` | Dados, não código | Dumps do MetaQA (`kb.tsv`, `entities.dict`, `relations.dict`, splits por hop). |
| `utils/`, `pickle_glove.py` | **Não modificados** | Cópias do original (usados pelo `AtualizacaoMetaQA/` que ainda depende de parte da infra antiga). |
| `RelationSupervision/` | **Nova — baseada no `AtualizacaoMetaQA/`** | Abandona completamente a predição de entidade: remove `Msubj/Mobj/Mrel`, `follow()` e `hop_selector` do `model.py`, e supervisiona só a classificação de relação por hop via `CrossEntropyLoss` contra um caminho-gabarito externo (extraído do BeamQA para o MetaQA — ver `src/kgqa/data/metaqa/build_relpaths_dataset.py` no repo principal). O vocabulário de relação ganha uma classe extra `<stop>` (caminhos mais curtos aprendem a prever `<stop>` nos hops restantes, até um teto fixo `--max_hops`). Reintroduz o histórico entre hops que existia no `Geral/` mas não no `WebQSP/AtualizacaoMetaQA/`: `last_h` (o `ctx_h` do hop anterior) é concatenado na entrada do hop seguinte, então cada hop agora depende do anterior. O `rel2id` é construído uma vez por fora (`build_rel2id.py`, escaneando a coluna `paths` dos CSVs) e passado como arquivo (`--rel2id`) tanto pro treino quanto pra inferência, em vez de reconstruído a cada execução — evita desalinhamento entre os ids usados no checkpoint e os usados depois. `predict.py` passou a validar relação por hop (exact-path-match + acurácia por hop, não mais Hits@1 de entidade) e o `train.py` persiste essas métricas incrementalmente em `save_dir/metrics.json` a cada validação. `predict_paths.py` virou o módulo de inferência de fato: recebe pergunta crua (sem gabarito), devolve até 3 relações mais prováveis por hop acima de 15% (`REL_PROB_THRESHOLD`/`TOP_K`, constantes no arquivo); quando rodado sobre um conjunto com gabarito (o uso esperado, sempre o conjunto de teste), também gera um `eval_report.json` com a taxa de "relação-gabarito está entre os candidatos retornados" por hop e a cobertura do caminho inteiro. |

## Linha do tempo reconstruída

1. **Original**: 3 arquiteturas diferentes por dataset — MetaQA-KB (GRU/GloVe), WebQSP (BERT, 1 "way"), CompWebQ (BERT, `num_ways`).
2. **`AtualizacaoMetaQA/`**: migra o MetaQA para a arquitetura do WebQSP (BERT + sigmoid), trocando o pipeline de dados para `AutoTokenizer`.
3. **`Geral/`**: unifica tudo sobre a base do CompWebQ (que já tinha `num_ways`) e adiciona a extração de caminho — essa é a versão que roda para os três datasets.
4. **`Modulo2/`**: consome o `predicted_paths.jsonl` do `Geral/` e interpreta a resposta final via LLM.
5. **`RelationSupervision/`**: linha de exploração paralela, a partir da `AtualizacaoMetaQA/` (não do `Geral/`). Em vez de aprender a apontar pra entidade-resposta certa com supervisão fraca (só o resultado final), aprende a prever a sequência de relações do caminho certo, hop a hop, com supervisão direta vinda do caminho-gabarito extraído do BeamQA para o MetaQA.

## Pontos de atenção para limpeza futura

- `AtualizacaoWebQSP/` pode ser removida (idêntica a `WebQSP/`).
- Os 3 variantes de debug em `AtualizacaoMetaQA/predict_paths_*.py` podem ser arquivados fora do `methods/` ou removidos, já que `Geral/predict.py` é a versão estável.
- `CompWebQ/` e `WebQSP/` continuam como estavam no fork original — só entram na comparação como referência arquitetural, não foram adaptadas para o pipeline de caminhos/LLM.
- `RelationSupervision/` depende de arquivos gerados no repositório principal (KGQA), fora deste submodule: os CSVs de caminho-gabarito (`src/kgqa/data/metaqa/outputs/metaqa_*.csv`) e o `rel2id.json` gerado por `build_rel2id.py` a partir deles. Rodar o treino/inferência exige apontar `--input_train`/`--input_valid`/`--rel2id` pra esses caminhos.
