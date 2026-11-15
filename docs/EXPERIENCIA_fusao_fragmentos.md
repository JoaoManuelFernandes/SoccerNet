# Fusão de fragmentos do seguimento para o reconhecimento do número

**Data:** 2026-09-23 · **Veredicto: NÃO ADOTAR** · Ganho na métrica oficial: +0.04 pontos (ruído)

---

## Pergunta

O seguimento parte a trajetória de um jogador em vários *tracklets*. Cada fragmento decide o número
de camisola isoladamente, com poucas imagens. Se os fragmentos do mesmo jogador partilhassem a
decisão, quanto melhorava o reconhecimento do número?

## Teto teórico

Agrupando os *tracklets* pela anotação (isto é, sabendo quais são o mesmo jogador) e propagando o
número do fragmento com maior confiança:

| | corretos / avaliados | exatidão |
|---|---|---|
| Sistema atual | 2131 / 5638 | 37.80% |
| Com fusão perfeita | 3434 / 5638 | **60.91%** |

**+23.1 pontos percentuais.** Há 1702 grupos com mais de um fragmento, 1143 dos quais discordam no
número. Para contexto, o ganho de MMOCR para Qwen2.5-VL reportado na dissertação foi +22.5 pp.

## Como identificar os fragmentos sem a anotação

Sinais testados, sobre 5250 pares do mesmo jogador contra 52 812 de jogadores diferentes:

| Sinal | Precisão | Conclusão |
|---|---|---|
| Ausência de sobreposição temporal | 9.0% | inútil; jogadores diferentes também não se sobrepõem |
| Continuidade de movimento (`bbox_pitch`, portão de velocidade) | 9–13% | inútil; num clipe de 30 s a posição não identifica ninguém |
| Cor média do equipamento | — | sinal errado: identifica a equipa, não o jogador |
| *Embeddings* de reidentificação | — | **não existem nos estados**, descartados ao gravar |
| **Distâncias ReID em `costs['R']`** | **94.9%** a 0.15 | **funciona** |

O estado guardado descarta os vetores de reidentificação, mas conserva, por deteção, a distância a
cada *track* candidato que o StrongSORT calculou para decidir as associações. A afinidade entre dois
*tracklets* reconstrói-se pela média dessas distâncias. Cobertura: 81.8% dos pares do mesmo jogador
têm pelo menos uma distância registada.

## Resultado na exatidão por *tracklet*

Limiar de distância escolhido em teste+validação (0.20), com o treino a servir de confirmação.
Critério de fusão: sem sobreposição de *frames*, mesma equipa prevista, distância ReID ≤ 0.20.
Dentro de cada grupo, o número do fragmento com maior confiança propaga-se aos restantes.

| Conjunto | Antes | Depois | Δ |
|---|---|---|---|
| Teste | 41.7% | 45.9% | +4.2 pp |
| Validação | 37.1% | 41.1% | +4.0 pp |
| Treino (retenção) | 35.3% | 43.0% | **+7.7 pp** |
| **Total** | **37.80%** | **43.30%** | **+5.50 pp** |

Por vídeo: melhoram 108, empatam 41, pioram 10, de 159. As perdas são todas de um ou dois
*tracklets*; o maior ganho é de doze. O treino, que não entrou na escolha do limiar, ganha mais do
que o teste — não há indício de sobreajuste.

## Resultado no GS-HOTA — e é aqui que a experiência falha

Aplicado ao conjunto de teste completo (876 *tracklets* fundidos, 211 mudam de número, 16 342
deteções alteradas em 44 dos 49 vídeos), avaliado com o TrackEval oficial:

| | GS-HOTA | DetA | AssA | LocA |
|---|---|---|---|---|
| Sistema da dissertação | 43.764 | 33.093 | 57.890 | 93.077 |
| Com fusão | 43.800 | 33.445 | 57.377 | 93.091 |
| **Δ** | **+0.036** | +0.352 | **−0.513** | +0.014 |

Melhoram 22 vídeos, ficam iguais 16, pioram 11.

## Porque é que 5.5 pontos numa métrica valem 0.04 na outra

**A exatidão por *tracklet* conta *tracklets*; o GS-HOTA conta deteções.** Os fragmentos que a fusão
corrige são precisamente os curtos — é por terem poucas imagens que liam mal. Corrigir dezenas de
fragmentos curtos desloca muito uma métrica em que cada *tracklet* vale um voto e quase nada uma
métrica ponderada pelo número de deteções: apenas 3% das deteções mudaram de número.

Acresce um efeito de retorno não previsto. O GS-HOTA valida uma deteção pela correspondência
integral de posição, função, equipa e número, pelo que o número faz parte da identidade. Atribuir o
mesmo número a vários *tracklets* torna-os indistinguíveis para o emparelhamento da métrica e
degrada a componente de associação: o AssA desce 0.513 e anula quase todo o ganho de 0.352 em DetA.

## Decisão

Não adotar. O custo — reescrever os estados, refazer a exatidão do número, a decomposição por
atributo, o GS-HOTA dos três conjuntos e o anexo vídeo a vídeo — não se justifica por +0.04 pontos
na métrica oficial. Adotar obrigaria ainda a explicar na dissertação porque é que um ganho de cinco
pontos numa tabela não aparece na seguinte, o que enfraqueceria o texto.

## O que se aproveita

O diagnóstico, que sustenta e quantifica uma afirmação já presente na conclusão: **a fragmentação do
seguimento limita o reconhecimento do número**. O teto dessa correção é de +23 pp na exatidão por
*tracklet*, e mesmo assim não se traduz no GS-HOTA. É argumento direto a favor da tese de que o
próximo avanço está nas etapas que alimentam o módulo — deteção e seguimento — e não na leitura.

## Reprodução

Todos os guiões em `scratchpad/` da sessão; os dados intermédios não são versionados.

| Ficheiro | Função |
|---|---|
| `cache_jnr.py` → `cache_jnr.pkl` | emparelha os *tracklets* com a anotação (importa `load_gt`/`match_to_gt` de `sn-gamestate/generate_report.py`). Reproduz a dissertação ao dígito: 2131/5638 = 37.80% |
| `reid.py` → `reid.pkl` | afinidade ReID entre *tracklets*, reconstruída de `costs['R']` |
| `traj.py` → `traj.pkl` | trajetórias no campo (sinal descartado) |
| `aplica.py` → `mapa_<split>.json` | calcula a fusão e o novo número por `track_id` |

Avaliação: alterar `attributes.jersey` nos JSON de `nosso_out/gshota/preds/nosso/` e correr
`nosso_out/gshota/eval_trackeval.py <pasta> test <saida.json>` com o `venv_tl`. Não requer GPU.
