# Resumo final dos dados gerados (2026-09-20, madrugada)

Tudo o que a tese precisa para a comparação baseline original vs nosso desenvolvimento, nos 160 vídeos do SoccerNet-GSR
(159 comparáveis: o SNGS-096 falha na calibração original no ambiente atual, nos dois sistemas).

## 1. GS-HOTA oficial (TrackEval, agrupado por conjunto) — `nosso_out/gshota/RESUMO_GSHOTA.md`, por vídeo em `gshota/*_<split>.json`

| Conjunto | vídeos | Baseline original | Nosso | Ganho |
|---|---|---|---|---|
| test | 49 | 28.95 | **44.00** | +15.05 |
| valid | 53 | 21.89 | **33.17** | +11.29 |
| train | 57 | 29.06 | **42.79** | +13.74 |

Validação: a baseline original no teste dá 28.95, contra 29.01 publicado pelo SoccerNet para a baseline (reprodução ao décimo).
Por vídeo: ganho mediano +13 a +14 pontos; nenhum vídeo piora no teste; pioram 5 no valid (025, 039, 042, 048, 049) e 2 no train (069, 113).

## 2. Equipa e número por tracklet (mesmos tracklets nos dois sistemas) — `nosso_out/COMPARACAO_FINAL.md`

| Conjunto | Equipa original | Equipa nossa | Número OCR original | Número JNR v2 |
|---|---|---|---|---|
| test | 1781/2148 = 82.9% | **2088/2148 = 97.2%** | 333/1731 = 19.2% | **722/1731 = 41.7%** |
| valid | 1378/1964 = 70.2% | **1609/1964 = 81.9%** | 267/1691 = 15.8% | **627/1691 = 37.1%** |
| train | 2515/2716 = 92.6% | **2561/2716 = 94.3%** | 265/2216 = 12.0% | **782/2216 = 35.3%** |

Nota: as deteções e os tracklets são idênticos nos dois sistemas (mesmo detetor e seguimento; o nosso ReID mantém a coluna original), por isso os denominadores coincidem.

## 3. Veredictos visuais do João (categorias) — `baseline_out/VEREDICTOS_VISUAIS_BASELINE.md` e `docs/FALHAS_VALIDADAS_VISUALMENTE.md`

| Conjunto | Sistema | OK | NOK-lado | NOK-grupo mesma equipa | NOK-grupo misturado | NOK-função | anotação invertida |
|---|---|---|---|---|---|---|---|
| test | original | 42 | 5 | 1 | 0 | 0 | 3 |
| test | nosso* | 47 | 1 | 0 | 0 | 0 | 1 |
| valid | original | 16 | 1 | 5 | 11 | 20 | 0 |
| valid | nosso* | 29 | 7 | 0 | 0 | 18 | 0 |
| train | original | 50 | 3 | 0 | 4 | 0 | 1 |
| train | nosso* | 55 | 1 | 1 | 0 | 0 | 0 |

\* nosso: passagem anterior do João (lista de NOK, categorias mais grossas, 192 não avaliável); a passagem sistemática com as 5 categorias está por fazer.
Anotação invertida confirmada em 4 vídeos (126, 131, 197 no teste; 111 no train): o sistema certo recebe 0% na métrica.
Ponto cego da métrica de equipa: só conta tracklets classificados como jogador (ex.: 042, 075, 088, 089, 156: visual NOK, métrica ≥95%).

## 4. Tempos por módulo — `nosso_out/TEMPOS_COMPARACAO.md`, por vídeo em `baseline_out/TEMPOS_POR_VIDEO.md`

Total dos módulos por clip de 30 s: original 21.3 min, nosso 22.0 min (RTX 3060 portátil). Equipa+lado < 1.5 s nos dois; número: MMOCR 487 s vs JNR v2 400-478 s.
Atenção: os tempos do nosso foram medidos com 2 workers de dados e memória mais apertada (swap) do que os da baseline (3 workers); a diferença ReID 449 vs 484 s e JNR 400 vs 478 s vem em parte daí, não dos módulos.

## 5. Vídeos para a apresentação — `baseline_out/apresentacao/README.md`
Pares baseline 100% original (render do estado com MMOCR) vs nosso: 148, 124, 146, 195, 200 (teste), 083, 056 (valid), 109, 160 (train).

## 6. Ficheiros
- estados encolhidos: `baseline_out/states` (original + JNR v2), `baseline_out/states_ocr` (100% original), `nosso_out/states` (nosso)
- predições no formato oficial: `nosso_out/gshota/preds/<original|nosso>/SoccerNetGS-<split>/tracklab/*.json`
- MP4 guardados da baseline (NOK e anotações invertidas): `baseline_out/videos`; nossos MP4: `docs/final_report/videos`, `docs/full_report/videos/train`
- pesquisa para trabalho futuro/produto: `docs/pesquisa_modulos_atuais_2026.md`
