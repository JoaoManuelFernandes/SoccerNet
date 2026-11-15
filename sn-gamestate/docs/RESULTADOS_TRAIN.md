# Resultados — Conjunto de TREINO (parcial: 33 de 57 vídeos)

Extraído em 2026-09-07 a partir dos estados dos lotes 1–11. Os lotes 12–19 (24 vídeos) não
chegaram a correr por falta de espaço em disco.

## Resumo global

| Métrica | Corretos | Total limpo | **Exatidão** |
|---|---|---|---|
| Reconhecimento do número | 418 | 1264 | **33.1%** |
| Atribuição de equipa e lado | 1408 | 1516 | **92.9%** |

## Comparação com os restantes conjuntos

| Conjunto | JNR | Equipa/lado |
|---|---|---|
| Treino (33 vídeos) | **33.1%** | **92.9%** |
| Validação (54 vídeos) | 35.7% | 82.4% |
| Teste (39 vídeos) | 39.5% | 97.8% |

Os valores do treino são coerentes com os já reportados no Capítulo 3, confirmando que ambos
os módulos generalizam de forma estável entre os três subconjuntos.

## Por lote

| Lote | JNR | Equipa/lado |
|---|---|---|
| batch01 | 40/111 (36.0%) | 144/144 (100.0%) |
| batch02 | 36/112 (32.1%) | 132/135 (97.8%) |
| batch03 | 29/114 (25.4%) | 148/150 (98.7%) |
| batch04 | 52/112 (46.4%) | 104/118 (88.1%) |
| batch05 | 25/109 (22.9%) | 131/132 (99.2%) |
| batch06 | 27/80 (33.8%) | 111/113 (98.2%) |
| batch07 | 50/100 (50.0%) | 110/110 (100.0%) |
| batch08 | 37/116 (31.9%) | 147/147 (100.0%) |
| batch09 | 55/152 (36.2%) | 160/160 (100.0%) |
| batch10 | 27/124 (21.8%) | 107/152 (70.4%) |
| batch11 | 40/134 (29.9%) | 114/155 (73.5%) |

## Melhores vídeos (candidatos a figura/vídeo final)

| Vídeo | JNR | Equipa/lado |
|---|---|---|
| SNGS-071 | 59.5% (42 tracklets) | 100.0% |
| SNGS-098 | 56.2% (32 tracklets) | 100.0% |
| SNGS-097 | 50.0% (34 tracklets) | 100.0% |
| SNGS-111 | 48.3% (29 tracklets) | 0.0% |
| SNGS-099 | 44.1% (34 tracklets) | 100.0% |
| SNGS-065 | 42.9% (35 tracklets) | 100.0% |
| SNGS-062 | 42.4% (33 tracklets) | 100.0% |
| SNGS-072 | 41.9% (43 tracklets) | 97.7% |
| SNGS-061 | 40.5% (42 tracklets) | 100.0% |
| SNGS-101 | 39.5% (38 tracklets) | 100.0% |

## Nota metodológica

As métricas foram extraídas com `generate_report.py` (correspondência por IoU ≥ 0.30, filtro de
pureza ≥ 0.80 para excluir trocas de identidade do *tracker*), sobre os estados guardados de cada
lote. Os relatórios detalhados por lote estão em `docs/train_reports/`.
