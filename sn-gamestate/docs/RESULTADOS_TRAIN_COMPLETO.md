# Conjunto de treino — resultados completos (57 vídeos)

**Data:** 2026-09-09. **Execução:** `run_train_full.sh` e `run_train_resume.sh`,
19 lotes de 3 vídeos, com GS-HOTA oficial (`trackeval.SoccerNetGS`, EVAL_SPACE=pitch).
Metodologia idêntica à usada em teste e validação, ou seja, média das avaliações por lote.

| Lote | Vídeos | GS-HOTA | Número | Equipa |
|---|---|---|---|---|
| 01 | 060, 061, 062 | 46.484% | 35.5% | 100.0% |
| 02 | 063, 064, 065 | 35.255% | 29.5% | 97.8% |
| 03 | 066, 067, 068 | 39.959% | 23.7% | 98.7% |
| 04 | 069, 070, 071 | 40.473% | 45.5% | 88.1% |
| 05 | 072, 073, 074 | 45.323% | 33.9% | 99.2% |
| 06 | 075, 076, 077 | 52.189% | 37.5% | 98.2% |
| 07 | 097, 098, 099 | 55.237% | 51.0% | 100.0% |
| 08 | 100, 101, 102 | 40.205% | 31.9% | 100.0% |
| 09 | 103, 104, 105 | 45.277% | 37.7% | 100.0% |
| 10 | 106, 107, 108 | 33.007% | 24.2% | 70.4% |
| 11 | 109, 110, 111 | 29.488% | 29.6% | 73.1% |
| 12 | 112, 113, 114 | 35.112% | 38.4% | 63.5% |
| 13 | 115, 151, 152 | 40.665% | 36.3% | 100.0% |
| 14 | 153, 154, 155 | 44.558% | 41.9% | 100.0% |
| 15 | 156, 157, 158 | 41.042% | 37.0% | 100.0% |
| 16 | 159, 160, 161 | 47.357% | 32.9% | 100.0% |
| 17 | 162, 163, 164 | 42.842% | 37.9% | 100.0% |
| 18 | 165, 166, 167 | 47.022% | 43.8% | 100.0% |
| 19 | 168, 169, 170 | 47.376% | 27.7% | 100.0% |

Lotes concluídos: 19  (57 vídeos de 57)
GS-HOTA médio por lote : 42.57%   (min 29.49 / max 55.24)
Número  acumulado      : 781/2216 = 35.2%
Equipa  acumulada      : 2561/2716 = 94.3%

## Comparação entre os três conjuntos

| Conjunto | Vídeos | GS-HOTA | Número | Equipa |
|---|---|---|---|---|
| Treino | 57 | **42.57%** | 35.2% (781/2216) | 94.3% (2561/2716) |
| Teste | 49 | 44.26% | 39.5% | 97.8% |
| Validação | 54 | 31.95% | 35.7% | 82.4% |
| **Total** | **160** | | | |

## Leitura

O treino situa-se muito perto do teste (42.57% contra 44.26%) e bem acima da
validação. Como o sistema nunca foi afinado sobre o conjunto de treino, esta
proximidade sustenta que não há sobreajuste ao conjunto de teste, e reforça a
explicação já dada para a validação, isto é, que a sua queda decorre da falha de
deteção em cerca de vinte vídeos e não dos módulos intervencionados.

Três observações por lote:

1. **Melhor resultado global do trabalho**, lote 07 (SNGS-097/098/099), com 55.24%
   de GS-HOTA e 51.0% de leitura de número. Candidato natural a figura de demonstração.
2. **Bloco contíguo de quebra na atribuição de equipa**, lotes 10, 11 e 12
   (SNGS-106 a SNGS-114), com 70.4%, 73.1% e 63.5% contra 98 a 100% em todos os
   restantes. Por serem vídeos consecutivos, é padrão e não ruído, e explica a
   diferença entre os 94.3% do treino e os 97.8% do teste. Merece diagnóstico.
3. **Dispersão elevada**, de 29.5% a 55.2%, o que justifica ter corrido os 57
   vídeos em vez de uma amostra.

## Ressalva sobre a comparabilidade

O conjunto de treino foi processado depois da reconstrução do ambiente de 2026-09-07,
que passou de Python 3.10 sob conda para Python 3.12 com PyTorch 2.6. Os conjuntos de
teste e de validação tinham sido processados no ambiente anterior. O código dos módulos
é o mesmo e as versões das bibliotecas foram fixadas para minimizar diferenças, mas não
se pode excluir que parte da diferença de 1.7 pontos entre o treino e o teste tenha
origem aí, e não no conjunto de dados. Se essa distinção vier a ser relevante para
alguma conclusão, a forma de a resolver é reprocessar um lote do teste no ambiente novo
e comparar com o valor histórico.

## Artefactos

- GS-HOTA por lote: `docs/train_gshota_valores.txt` (extraído por `extract_gshota.py`)
- Registo cronológico: `docs/train_gshota.txt`
- Métricas por lote: `docs/train_reports/train-b01.md` a `train-b19.md`
- Logs completos: `docs/train_logs/batch01.log` a `batch19.log`
- Agregação: `agrega_train.py`
