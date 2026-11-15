# Vídeos para a apresentação: baseline 100% original vs nosso desenvolvimento (teste)

`<V>_baseline_original_100pct.mp4`: render do estado `states_ocr` (pipeline original inteira, incluindo o MMOCR).
`<V>_nosso.mp4`: o nosso desenvolvimento (docs/final_report/videos/test).
Números: exatidão por tracklet, mesmos tracklets (OCR original vs JNR v2). Equipa: veredicto visual do João + métrica.

## Equipa (mau no original, bom no nosso)
| Vídeo | Baseline original | Nosso | Número OCR original -> JNR v2 |
|---|---|---|---|
| SNGS-148 | NOK-lado (0/48) | OK | 18.2% -> 56.8% |
| SNGS-124 | NOK-grupo, tudo a mesma equipa (28/52) | OK | 17.9% -> 46.4% |

## Número (mau no original, bom no nosso)
| Vídeo | OCR original | JNR v2 | Equipa |
|---|---|---|---|
| SNGS-146 | 31.6% | 73.7% (14 certos, 1 errado, 19 tracklets) | 100% nos dois |
| SNGS-195 | 4.2% | 50.0% | 100% nos dois (o nosso pelo critério 4, ver §3.3) |
| SNGS-200 | 5.3% | 47.4% | 100% nos dois |

Outros pares "equipa NOK no original / OK no nosso" disponíveis (MP4 da baseline com números do JNR v2, não do MMOCR):
137, 140, 143, 190. Para os usar na apresentação, renderizar com tools/render_ocr.sh test SNGS-xxx.

## Pares adicionais do valid e do train (2026-09-20, com GS-HOTA por vídeo)
| Vídeo | Split | Baseline original | Nosso | Número OCR original -> JNR v2 | GS-HOTA original -> nosso |
|---|---|---|---|---|---|
| SNGS-083 | valid | NOK-grupo misturado (equipa 26%) | OK (98%) | 16% -> 37% | 24.0 -> 50.9 |
| SNGS-109 | train | NOK-lado (equipa 0%) | OK (100%) | 2% -> 29% | 13.6 -> 38.3 |
| SNGS-056 | valid | (equipa amarela como árbitro nos dois) | — | 46% -> 91% | número: o melhor ganho do valid |
| SNGS-160 | train | OK nos dois | OK | 15% -> 65% | número: o melhor ganho do train |
Sugestão final para a apresentação: equipa = SNGS-148 (teste) ou SNGS-109 (train); número = SNGS-146 (teste) ou SNGS-160 (train).
