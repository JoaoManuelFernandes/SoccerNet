# Veredictos visuais do João sobre os vídeos da BASELINE ORIGINAL (módulos team/team_side originais)

Data: 2026-09-14. Conjunto de teste, vídeos 116 a 150 e 187 a 199 (48). O SNGS-200 fica para o lote seguinte.
Vídeo = `baseline_out/videos/<V>.mp4`. Métrica = exatidão por tracklet da equipa original (COMPARACAO_POR_VIDEO.md).

| Vídeo | Veredicto visual | Métrica por tracklet | Concordam? |
|---|---|---|---|
| SNGS-116 | OK | 47/47 = 100.0% | sim |
| SNGS-117 | OK | 38/38 = 100.0% | sim |
| SNGS-118 | OK | 40/40 = 100.0% | sim |
| SNGS-119 | OK | 26/26 = 100.0% | sim |
| SNGS-120 | OK | 48/48 = 100.0% | sim |
| SNGS-121 | OK | 38/38 = 100.0% | sim |
| SNGS-122 | OK | 57/57 = 100.0% | sim |
| SNGS-123 | OK | 40/40 = 100.0% | sim |
| SNGS-124 | NOK-grupo (tudo a mesma equipa) | 28/52 = 53.8% | sim |
| SNGS-125 | OK | 40/40 = 100.0% | sim |
| SNGS-126 | OK (anotação errada, confirmado 2.ª visualização) | 0/36 = 0.0% | anotação invertida |
| SNGS-127 | OK | 22/22 = 100.0% | sim |
| SNGS-128 | OK | 37/37 = 100.0% | sim |
| SNGS-129 | OK | 44/45 = 97.8% | sim |
| SNGS-130 | OK | 36/36 = 100.0% | sim |
| SNGS-131 | OK (anotação errada, confirmado 2.ª visualização) | 0/39 = 0.0% | anotação invertida |
| SNGS-132 | OK | 44/44 = 100.0% | sim |
| SNGS-133 | OK | 47/47 = 100.0% | sim |
| SNGS-134 | OK | 44/44 = 100.0% | sim |
| SNGS-135 | OK | 63/63 = 100.0% | sim |
| SNGS-136 | OK | 45/46 = 97.8% | sim |
| SNGS-137 | NOK-lado (trocado) | 0/38 = 0.0% | sim |
| SNGS-138 | OK | 61/61 = 100.0% | sim |
| SNGS-139 | OK | 45/45 = 100.0% | sim |
| SNGS-140 | OK [revisto após recalibração] | 0/53 = 0.0% | sim |
| SNGS-141 | OK | 68/68 = 100.0% | sim |
| SNGS-142 | OK | 56/56 = 100.0% | sim |
| SNGS-143 | NOK-lado (trocado) | 0/49 = 0.0% | sim |
| SNGS-144 | OK | 47/47 = 100.0% | sim |
| SNGS-145 | OK | 59/59 = 100.0% | sim |
| SNGS-146 | OK | 27/27 = 100.0% | sim |
| SNGS-147 | OK | 51/51 = 100.0% | sim |
| SNGS-148 | NOK-lado (trocado; corrigido na 2.ª visualização) | 0/48 = 0.0% | sim |
| SNGS-149 | OK | 38/39 = 97.4% | sim |
| SNGS-150 | OK | 48/48 = 100.0% | sim |
| SNGS-187 | OK | 34/34 = 100.0% | sim |
| SNGS-188 | OK | 27/27 = 100.0% | sim |
| SNGS-189 | OK | 42/42 = 100.0% | sim |
| SNGS-190 | NOK-lado (trocado) | 0/38 = 0.0% | sim |
| SNGS-191 | OK | 38/38 = 100.0% | sim |
| SNGS-192 | OK | 42/42 = 100.0% | sim |
| SNGS-193 | OK | 48/48 = 100.0% | sim |
| SNGS-194 | OK | 35/35 = 100.0% | sim |
| SNGS-195 | OK | 31/31 = 100.0% | sim |
| SNGS-196 | OK | 47/47 = 100.0% | sim |
| SNGS-197 | OK (anotação errada, confirmado 2.ª visualização) | 0/38 = 0.0% | anotação invertida |
| SNGS-198 | OK | 54/54 = 100.0% | sim |
| SNGS-199 | OK | ver COMPARACAO_POR_VIDEO.md | sim |

## Reverificação (2026-09-14, 2.ª visualização do João)

- 126, 131, 197: **a anotação de referência é que está invertida**; a baseline original está visualmente certa.
  Coincide com Oo et al. (Broadcast2Pitch, WACV 2026), que assinalam exatamente estes três vídeos.
  Consequência: a métrica oficial penaliza a resposta certa nestes 3 vídeos de teste (113 tracklets).
- 148: **baseline original errada** (lado trocado); anotação certa.

## Resumo do teste (48 vídeos vistos; 200 por ver)
- OK visual: 42 (dos quais 3 com anotação invertida: 126, 131, 197)
- NOK-grupo: 1 (124)
- NOK-lado: 5 (137, 140, 143, 148, 190)

## MP4 mantidos
Os 6 NOK (para a apresentação e para a comparação com o nosso desenvolvimento) e os 3 de anotação
invertida (prova para a tese). Os restantes 39 foram apagados. Nada de states/ ou states_ocr/ foi apagado.

## VALID (2026-09-15, 1.ª parte: 021 a 058, 081, 082; 083 por confirmar)

| Vídeo | Veredicto visual | Métrica por tracklet (equipa original) | Nota |
|---|---|---|---|
| SNGS-021 | OK | 51/51 = 100.0% |  |
| SNGS-022 | OK | 50/50 = 100.0% |  |
| SNGS-023 | OK | 39/39 = 100.0% |  |
| SNGS-024 | NOK-grupo (misturado: alguns tracklets certos, outros errados) | 25/41 = 61.0% |  |
| SNGS-025 | OK | 44/44 = 100.0% |  |
| SNGS-026 | NOK-grupo (tudo a mesma equipa) | 22/39 = 56.4% |  |
| SNGS-027 | NOK-grupo (tudo a mesma equipa) | 12/23 = 52.2% |  |
| SNGS-028 | NOK-grupo (tudo a mesma equipa) | 18/33 = 54.5% |  |
| SNGS-029 | NOK-lado (trocado) [revisto após recalibração] | 57/57 = 100.0% |  |
| SNGS-030 | NOK-grupo (tudo a mesma equipa) | 8/20 = 40.0% |  |
| SNGS-031 | NOK-grupo (mesma equipa) [revisto após recalibração] | 35/52 = 67.3% |  |
| SNGS-032 | NOK-grupo (misturado: alguns tracklets certos, outros errados) | 35/55 = 63.6% |  |
| SNGS-033 | OK | 48/53 = 90.6% | métrica baixa com visual OK |
| SNGS-034 | NOK-grupo (misturado: alguns tracklets certos, outros errados) | 25/38 = 65.8% |  |
| SNGS-035 | OK | 50/57 = 87.7% | métrica baixa com visual OK |
| SNGS-036 | OK | 30/31 = 96.8% |  |
| SNGS-037 | OK | 34/34 = 100.0% |  |
| SNGS-038 | OK | 29/29 = 100.0% |  |
| SNGS-039 | NOK-função (equipa de amarelo tratada como árbitro; equipa/lado mal à partida) | 7/32 = 21.9% |  |
| SNGS-040 | NOK-função (equipa de amarelo tratada como árbitro; equipa/lado mal à partida) | 28/35 = 80.0% |  |
| SNGS-041 | NOK-função (equipa de amarelo tratada como árbitro; equipa/lado mal à partida) | 3/17 = 17.6% |  |
| SNGS-042 | NOK-função (equipa de amarelo tratada como árbitro; equipa/lado mal à partida) | 26/27 = 96.3% | **métrica cega**: visual NOK com métrica alta |
| SNGS-043 | NOK-função (equipa de amarelo tratada como árbitro; equipa/lado mal à partida) | 4/32 = 12.5% |  |
| SNGS-044 | NOK-função (equipa de amarelo tratada como árbitro; equipa/lado mal à partida) | 21/32 = 65.6% |  |
| SNGS-045 | NOK-função (equipa de amarelo tratada como árbitro; equipa/lado mal à partida) | 13/38 = 34.2% |  |
| SNGS-046 | NOK-função (equipa de amarelo tratada como árbitro; equipa/lado mal à partida) | 21/28 = 75.0% |  |
| SNGS-047 | NOK-função (equipa de amarelo tratada como árbitro; equipa/lado mal à partida) | 3/26 = 11.5% |  |
| SNGS-048 | NOK-função (equipa de amarelo tratada como árbitro; equipa/lado mal à partida) | 4/15 = 26.7% |  |
| SNGS-049 | NOK-função (equipa de amarelo tratada como árbitro; equipa/lado mal à partida) | 2/20 = 10.0% |  |
| SNGS-050 | NOK-função (equipa de amarelo tratada como árbitro; equipa/lado mal à partida) | 14/23 = 60.9% |  |
| SNGS-051 | NOK-função (equipa de amarelo tratada como árbitro; equipa/lado mal à partida) | 10/16 = 62.5% |  |
| SNGS-052 | NOK-função (equipa de amarelo tratada como árbitro; equipa/lado mal à partida) | 15/21 = 71.4% |  |
| SNGS-053 | NOK-função (equipa de amarelo tratada como árbitro; equipa/lado mal à partida) | 2/13 = 15.4% |  |
| SNGS-054 | NOK-função (equipa de amarelo tratada como árbitro; equipa/lado mal à partida) | 18/28 = 64.3% |  |
| SNGS-055 | NOK-função (equipa de amarelo tratada como árbitro; equipa/lado mal à partida) | 5/26 = 19.2% |  |
| SNGS-056 | NOK-função (equipa de amarelo tratada como árbitro; equipa/lado mal à partida) | 0/14 = 0.0% |  |
| SNGS-057 | NOK-função (equipa de amarelo tratada como árbitro; equipa/lado mal à partida) | 25/33 = 75.8% |  |
| SNGS-058 | NOK-função (equipa de amarelo tratada como árbitro; equipa/lado mal à partida) | 17/30 = 56.7% |  |
| SNGS-081 | NOK-grupo (tudo a mesma equipa) | 29/54 = 53.7% |  |
| SNGS-082 | NOK-grupo (misturado: alguns tracklets certos, outros errados) | 37/52 = 71.2% |  |

Resumo: OK 10 (021, 022, 023, 025, 029, 033, 035, 036, 037, 038); NOK-grupo misturado 5 (024, 031, 032, 034, 082);
NOK-grupo colapsado 5 (026, 027, 028, 030, 081); NOK-função 20 (039 a 058, equipa amarela classificada como árbitro).
Teste: SNGS-200 OK (fecha o teste: 49/49 vistos).
Modo novo face ao teste: 'misturado' (parte dos tracklets trocados, ex. 082 tracklets 4 e 13 de branco dados como right, sendo o right azul).
MP4 mantidos: todos os NOK do valid. Apagados: os 10 OK e o 200 fica (render para apresentação).

## VALID (2.ª parte, 2026-09-16: 083 a 095) e TRAIN (1.ª parte: 060 a 077, 097 a 113)

| Vídeo | Split | Veredicto visual | Métrica por tracklet (equipa original) | Nota |
|---|---|---|---|---|
| SNGS-083 | valid | NOK-grupo (misturado) | 13/49 = 26.5% |  |
| SNGS-084 | valid | NOK-grupo (misturado) | 44/63 = 69.8% |  |
| SNGS-085 | valid | NOK-grupo (misturado) | 35/43 = 81.4% |  |
| SNGS-086 | valid | OK | 43/45 = 95.6% |  |
| SNGS-087 | valid | OK | 33/34 = 97.1% |  |
| SNGS-088 | valid | NOK-grupo (misturado) (tracklet 14) | 38/40 = 95.0% | **métrica cega** (visual NOK, métrica alta) |
| SNGS-089 | valid | NOK-grupo (misturado) (tracklet 4) | 49/49 = 100.0% | **métrica cega** (visual NOK, métrica alta) |
| SNGS-090 | valid | OK | 44/45 = 97.8% |  |
| SNGS-091 | valid | NOK-grupo (misturado: alguns tracklets trocados; corrigido na 2.ª visualização) | 39/45 = 86.7% | concordam |
| SNGS-092 | valid | OK (anotação errada; 2.ª visualização do nosso deu OK e os dois sistemas atribuem os mesmos lados, 0% nos dois) | 0/55 = 0.0% | anotação invertida |
| SNGS-093 | valid | OK | 48/48 = 100.0% |  |
| SNGS-094 | valid | NOK-grupo (misturado) | 33/46 = 71.7% |  |
| SNGS-095 | valid | OK | 43/44 = 97.7% |  |
| SNGS-060 | train | OK | 54/54 = 100.0% |  |
| SNGS-061 | train | OK | 47/47 = 100.0% |  |
| SNGS-062 | train | OK | 42/42 = 100.0% |  |
| SNGS-063 | train | OK | 50/50 = 100.0% |  |
| SNGS-064 | train | OK | 43/43 = 100.0% |  |
| SNGS-065 | train | OK | 42/42 = 100.0% |  |
| SNGS-066 | train | OK | 59/59 = 100.0% |  |
| SNGS-067 | train | OK | 47/47 = 100.0% |  |
| SNGS-068 | train | NOK-grupo (misturado) | 31/45 = 68.9% |  |
| SNGS-069 | train | OK | 30/30 = 100.0% |  |
| SNGS-070 | train | OK | 46/46 = 100.0% |  |
| SNGS-071 | train | OK | 42/42 = 100.0% |  |
| SNGS-072 | train | OK [revisto após recalibração] | 0/43 = 0.0% |  |
| SNGS-073 | train | OK | 52/52 = 100.0% |  |
| SNGS-074 | train | OK | 37/37 = 100.0% |  |
| SNGS-075 | train | NOK-grupo (misturado) | 44/44 = 100.0% | **métrica cega** (visual NOK, métrica alta) |
| SNGS-076 | train | OK | 30/30 = 100.0% |  |
| SNGS-077 | train | OK | 39/39 = 100.0% |  |
| SNGS-097 | train | OK | 34/34 = 100.0% |  |
| SNGS-098 | train | OK | 34/34 = 100.0% |  |
| SNGS-099 | train | OK | 42/42 = 100.0% |  |
| SNGS-100 | train | OK | 54/54 = 100.0% |  |
| SNGS-101 | train | OK | 47/47 = 100.0% |  |
| SNGS-102 | train | OK | 46/46 = 100.0% |  |
| SNGS-103 | train | OK | 59/59 = 100.0% |  |
| SNGS-104 | train | OK | 41/41 = 100.0% |  |
| SNGS-105 | train | OK | 59/59 = 100.0% |  |
| SNGS-106 | train | NOK-lado (trocado; agrupamento OK; corrigido na 2.ª visualização) | 0/45 = 0.0% | concordam |
| SNGS-107 | train | OK | 54/54 = 100.0% |  |
| SNGS-108 | train | OK | 53/53 = 100.0% |  |
| SNGS-109 | train | OK [revisto após recalibração] | 0/57 = 0.0% |  |
| SNGS-110 | train | OK | 57/57 = 100.0% |  |
| SNGS-111 | train | OK (anotação errada, confirmado 2.ª visualização) | 0/42 = 0.0% | anotação invertida |
| SNGS-112 | train | OK | 41/41 = 100.0% |  |
| SNGS-113 | train | OK | 46/46 = 100.0% |  |

Resumo valid 083-095 (13): OK 6; NOK-grupo misturado 6 (083, 084, 085, 088, 089, 091); NOK-lado 1 (092).
Resumo train 060-077, 097-113 (35): OK 29 (dos quais 1 com anotação invertida: 111); NOK-lado 3 (072, 106, 109); NOK-grupo misturado 3 (068, 075, 094).
NOK-função e anotação errada: nenhum neste lote. 081 (mesma equipa) e 082 (misturado) já estavam registados na 1.ª parte.
MP4 apagados: os OK deste lote. Mantidos: todos os NOK.

## TRAIN (2.ª parte, 2026-09-18: 114, 115, 151 a 170) — fecha a baseline original

| Vídeo | Veredicto visual | Métrica por tracklet (equipa original) | Nota |
|---|---|---|---|
| SNGS-114 | OK | 39/39 = 100.0% |  |
| SNGS-115 | OK | 42/42 = 100.0% |  |
| SNGS-151 | OK | 40/40 = 100.0% |  |
| SNGS-152 | OK | 46/46 = 100.0% |  |
| SNGS-153 | NOK-lado (trocado) [revisto após recalibração] | 49/49 = 100.0% |  |
| SNGS-154 | OK | 72/72 = 100.0% |  |
| SNGS-155 | OK | 40/40 = 100.0% |  |
| SNGS-156 | NOK-grupo (misturado, tracklet 4) | 43/43 = 100.0% | métrica cega |
| SNGS-157 | OK | 51/51 = 100.0% |  |
| SNGS-158 | OK | 52/52 = 100.0% |  |
| SNGS-159 | OK | 74/74 = 100.0% |  |
| SNGS-160 | OK | 36/36 = 100.0% |  |
| SNGS-161 | OK | 48/48 = 100.0% |  |
| SNGS-162 | OK | 80/80 = 100.0% |  |
| SNGS-163 | OK | 51/51 = 100.0% |  |
| SNGS-164 | OK | 57/57 = 100.0% |  |
| SNGS-165 | OK | 37/37 = 100.0% |  |
| SNGS-166 | OK | 36/36 = 100.0% |  |
| SNGS-167 | OK | 52/52 = 100.0% |  |
| SNGS-168 | OK | 50/50 = 100.0% |  |
| SNGS-169 | OK | 61/61 = 100.0% |  |
| SNGS-170 | OK | 57/57 = 100.0% |  |

Resumo: OK 21; NOK-grupo misturado 1 (156). SNGS-096 (valid) sem vídeo: a calibração original falhou.

## TOTAIS DA BASELINE ORIGINAL (159 vídeos vistos)
- Teste (49): OK 42 (3 com anotação invertida: 126, 131, 197); NOK-grupo 1 (124); NOK-lado 5 (137, 140, 143, 148, 190).
- Valid (53): OK 16; NOK-grupo misturado 11 (024, 031, 032, 034, 082, 083, 084, 085, 088, 089, 091); NOK-grupo mesma equipa 5 (026, 027, 028, 030, 081); NOK-lado 1 (092); NOK-função 20 (039 a 058).
- Train (57): OK 50 (1 com anotação invertida: 111); NOK-lado 3 (072, 106, 109); NOK-grupo misturado 4 (068, 075, 094, 156).
- Total: OK 108 (4 anotações invertidas), NOK 51 (lado 9, grupo mesma equipa 6, grupo misturado 16, função 20).
