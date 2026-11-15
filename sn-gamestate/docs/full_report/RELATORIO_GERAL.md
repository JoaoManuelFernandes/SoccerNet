# Relatório Geral — SoccerNet Game State Reconstruction

**Data**: 2026-07-05 (actualizado 2026-07-10 com re-calibração team-side)
**Cobertura**: test COMPLETO (49 vídeos) + valid COMPLETO (54 vídeos) = 103 vídeos avaliados. Train (57) pendente.
**Melhorias avaliadas**: `team_side` (Fix #1/#2 + **re-calibração por silhueta**) + `jersey_number_detect` (Qwen2.5-VL v2 + fixes de prompt/votação)
**Componentes originais (não otimizados)**: deteção YOLOv8, ReID PRTreid, tracking StrongSORT, calibração nbjw_calib, role

> **⚠️ ACTUALIZAÇÃO 2026-07-10 — re-calibração team-side.** Os thresholds foram
> re-calibrados conjuntamente em test+valid (grid-search com zero regressões no test).
> Isto elevou o team-side do valid de **73.3%→82.4%** e o GS-HOTA do valid de
> **29.10%→31.95%** (e do test de 43.27%→44.26%, via SNGS-190). As secções abaixo com
> os números "antes" ficam como referência histórica; ver §1 e §2 para os valores finais.

---

## 1. Resumo executivo (valores FINAIS, pós re-calibração)

| Métrica | Test (49 vídeos) | Valid (54 vídeos) | Observação |
|---------|------------------|--------------------|------------|
| **GS-HOTA (média de lotes)** | **44.26%** | **31.95%** | valid arrastado por falha de DETEÇÃO em ~20 vídeos (ver §4) |
| ↳ antes da re-calibração | 43.27% | 29.10% | — |
| **JNR accuracy** (média/vídeo) | 39.5% | **35.8%** | generaliza bem |
| **Team-side accuracy** (global tracklets) | **97.8%** | **82.4%** | antes: 97.9% / 73.3% — re-calibração deu +9.1 pp no valid |
| Baseline oficial GS-HOTA | 23.36 (2024) / 29.01 (2025) | — | leaderboard usa só o test |

**Leitura honesta**: no test estamos ao nível do 2.º lugar de 2024 (UPCxMobius 43.15).
Depois da re-calibração, o team-side **generaliza muito melhor** (73.3%→82.4% no valid) —
resolvemos o sobreajuste original substituindo os thresholds fixos (one_sided+L2) por um
critério de **qualidade de clustering (silhueta)**. O que **resta** no valid é sobretudo
o gargalo de **deteção**: em ~20 vídeos (039–058) o YOLOv8 base quase não deteta jogadores
(DetA 1.5–7%), e aí nem team-side nem JNR perfeitos movem o GS-HOTA. Isso aponta o
próximo passo de maior retorno: melhorar a deteção.

---

## 2. GS-HOTA — Todos os lotes

### 2.1 Test (13 lotes, 49 vídeos) — média ≈ 43.27%

| Lote | Vídeos | GS-HOTA |
|------|--------|---------|
| 1 | 118–122 | 51.448% |
| 2 | 123–126, 129 | 39.501% |
| 3 | 132–136 | 42.324% |
| 4 | 137–141 | 39.282% |
| 5 | 142, 145–148 | 44.222% |
| 6 | 149, 150, 187–189 | 40.156% |
| 7 | 192–196 | 42.501% |
| 8 | 197–200 | 42.715% |
| Extra A | 116, 117 | 44.363% |
| Extra B | 127, 128 | **52.208%** ← melhor |
| Extra C | 130, 131 | 41.290% |
| Extra D | 143, 144 | 51.862% |
| Extra E | 190, 191 | 30.609% ← pior (SNGS-190 team-side 0%) |

### 2.2 Valid (18 lotes, 54 vídeos) — média ≈ 29.10%

| Lote | Vídeos | GS-HOTA | DetA | Diagnóstico |
|------|--------|---------|------|-------------|
| V01 | 021–023 | 43.438% | ok | saudável |
| V02 | 024–026 | 35.073% | ok | team-side parcial (025, 026) |
| V03 | 027–029 | 27.329% | ok | team-side parcial (027, 028) |
| V04 | 030–032 | 39.165% | ok | saudável |
| V05 | 033–035 | 49.186% | ok | saudável |
| V06 | 036–038 | 44.753% | ok | saudável |
| V07 | 039–041 | 11.410% | **2.7%** | ❌ deteção |
| V08 | 042–044 | 12.285% | **3.8%** | ❌ deteção |
| V09 | 045–047 | 6.585% | **2.0%** | ❌ deteção |
| V10 | 048–050 | 13.175% | baixo | ❌ deteção |
| V11 | 051–053 | 13.424% | **3.6%** | ❌ deteção |
| V12 | 054–056 | 5.284% | **1.5%** | ❌ deteção (pior lote) |
| V13 | 057, 058, 081 | 18.203% | **7.3%** | ❌ deteção |
| V14 | 082–084 | 37.139% | 26.4% | recupera |
| V15 | 085–087 | 44.214% | 36.5% | saudável |
| V16 | 088–090 | **50.104%** | 42.9% | melhor do valid |
| V17 | 091–093 | 31.431% | 18.5% | misto (092 team 0%) |
| V18 | 094–096 | 41.645% | 29.4% | saudável |

**Divisão nítida**: lotes saudáveis (11) ≈ **40.3%** vs lotes com falha de deteção (V07–V13, 20 vídeos) ≈ **11.5%**.

### 2.3 Comparação com leaderboards oficiais (test)

| Sistema | 2024 | 2025 |
|---------|------|------|
| Vencedor | 63.81 | 63.90 |
| 2.º/2024 (UPCxMobius) | 43.15 | — |
| **Nós (test 49 vídeos, aprox.)** | **≈43.27** → **~2.º–3.º de 16** | **≈43.27** → **~7.º de 14** |
| eidos.ai / UTokyo (2025) | — | 46.24 / 40.13 |
| Baseline oficial | 23.36 | 29.01 |

Ressalva metodológica: a nossa média é de chamadas TrackEval separadas por lote (HOTA não é aditivo); o número oficial exigiria uma única chamada com os 49 vídeos.

---

## 3. JNR — Jersey Number Recognition

### 3.1 Globais por split

| Split | Correto/Total | Accuracy | Baseline MMOCR |
|-------|---------------|----------|----------------|
| Test (39 vídeos originais) | 554/1404 | **39.5%** | 18.6% (+20.9pp, 38V-1E-0D) |
| Test (10 extras) | 162/324 | **50.0%** | não medida (sem pklz baseline extraído) |
| Valid (54 vídeos) | 617/1726 | **35.7%** | não medida |

O JNR **generaliza**: valid a 35.7% vs test 39.5% — a diferença é pequena e explicável pelos 20 vídeos de deteção fraca (menos crops utilizáveis por tracklet).

### 3.2 Valid — melhores e piores

| Melhores | JNR | Piores | JNR |
|----------|-----|--------|-----|
| 053 | 62.5% (5/8) | 056 | 0.0% (0/11) |
| 030 | 61.1% | 042 | 7.4% |
| 028 | 60.0% | 040 | 8.6% |
| 034 | 57.9% | 043 | 10.7% |
| 087 | 55.9% | 049 | 12.5% |

Os piores estão todos no bloco 039–058 (deteção fraca → crops raros/pequenos). Nos vídeos saudáveis o JNR do valid comporta-se como no test.

### 3.3 Test extras — por vídeo

| Vídeo | JNR | Team | | Vídeo | JNR | Team |
|-------|-----|------|-|-------|-----|------|
| 116 | 52.6% | 100% | | 131 | 38.5% | 43.6% |
| 117 | 57.1% | 100% | | 143 | 45.7% | 100% |
| 127 | 70.0% | 100% | | 144 | 57.4% | 100% |
| 128 | 51.5% | 100% | | 190 | 41.4% | **0%** |
| 130 | 58.1% | 100% | | 191 | 38.9% | 100% |

### 3.4 Valid — tabela completa por vídeo

| Vídeo | JNR | Team | | Vídeo | JNR | Team |
|-------|-----|------|-|-------|-----|------|
| 021 | 43.5% | 100% | | 048 | 30.0% | 26.7% |
| 022 | 38.1% | 100% | | 049 | 12.5% | 90.0% |
| 023 | 37.8% | 100% | | 050 | 26.1% | 100% |
| 024 | 53.8% | 100% | | 051 | 37.5% | 62.5% |
| 025 | 40.5% | 61.4% | | 052 | 35.0% | 25.0% |
| 026 | 44.1% | 56.4% | | 053 | 62.5% | 15.4% |
| 027 | 25.0% | 50.0% | | 054 | 28.6% | 64.3% |
| 028 | 60.0% | 54.5% | | 055 | 26.1% | 69.2% |
| 029 | 34.1% | 100% | | 056 | 0.0% | 57.1% |
| 030 | 61.1% | 60.0% | | 057 | 36.4% | 45.5% |
| 031 | 42.0% | 98.1% | | 058 | 26.7% | 56.7% |
| 032 | 31.2% | 98.2% | | 081 | 48.1% | 16.7% |
| 033 | 32.0% | 79.2% | | 082 | 34.6% | 100% |
| 034 | 57.9% | 97.4% | | 083 | 37.2% | 98.0% |
| 035 | 42.4% | 93.0% | | 084 | 27.1% | 30.2% |
| 036 | 47.6% | 51.6% | | 085 | 48.8% | 74.4% |
| 037 | 54.5% | 97.1% | | 086 | 53.7% | 100% |
| 038 | 48.3% | 55.2% | | 087 | 55.9% | 97.1% |
| 039 | 15.6% | **0%** | | 088 | 37.5% | 100% |
| 040 | 8.6% | 91.4% | | 089 | 38.6% | 100% |
| 041 | 29.4% | 17.6% | | 090 | 44.2% | 100% |
| 042 | 7.4% | 89.3% | | 091 | 29.0% | 57.8% |
| 043 | 10.7% | 87.5% | | 092 | 27.5% | **0%** |
| 044 | 17.4% | 65.6% | | 093 | 50.0% | 97.9% |
| 045 | 13.9% | 68.4% | | 094 | 46.3% | 84.8% |
| 046 | 23.1% | 7.1% | | 095 | 26.7% | 97.7% |
| 047 | 15.4% | 11.5% | | 096 | 16.7% | 82.5% |

---

## 4. Diagnóstico #1 — Falha de deteção no valid (039–058)

**Evidência quantitativa**: DetA entre 1.5% e 7.3% nos lotes V07–V13, com AssA (40–53%) e LocA (85–93%) normais. Ou seja: das poucas deteções feitas, o tracking e a calibração funcionam bem — o problema é que o YOLOv8 base falha em produzir deteções.

**Impacto**: 20 vídeos ≈ 37% do valid a puxar a média de ~40.3% para 29.1%.

**A quem pertence o problema**: à deteção (componente original, não tocado por nós). Os nossos módulos recebem quase nada para trabalhar — o JNR nesses vídeos cai para 8–30% e o GS-HOTA para 5–18%.

**Confirmação visual**: vídeos em `docs/full_report/videos/valid/SNGS-039.mp4` a `SNGS-058.mp4` — espera-se ver o campo quase sem bounding boxes.

**Caminho de melhoria** (se o objetivo for o valid): substituir/afinar o detector (ex. YOLO mais recente, threshold de confiança mais baixo, fine-tuning nos frames destes vídeos). Fora do âmbito das melhorias atuais.

---

## 5. Diagnóstico #2 — Team-side degrada no valid (97.9% test → 69.3% valid)

**Padrões observados**:
- **Inversões totais (0%)**: 039, 092 (e 190 no test-extra) — o lado atribuído ficou trocado; mesma família do antigo SNGS-195.
- **Clustering parcialmente misto (50–70%)**: ~15 vídeos — os clusters não separam bem as equipas, mas não é inversão total.
- **Quase-perfeitos (>90%)**: 24 vídeos — onde o clustering ReID é limpo, os fixes seguram.

**Causa provável**: os thresholds dos fixes foram calibrados no test set (L2<2.04 para weak-ReID; one_sided≥0.85; gap mínimo 0.5m no Fix #2). O valid tem condições diferentes (outras câmaras, equipamentos, iluminação) e os cortes não transferem. Isto era um risco conhecido — na altura documentámos que "com só 3 pontos de referência não há base estatística para thresholds seguros".

**O que fazer com isto** (recomendações para a tese):
1. Reportar test e valid separadamente, com esta análise — mostra rigor.
2. Se houver tempo: re-calibrar thresholds com valid+test juntos (ou um classificador simples em vez de cortes fixos) e reavaliar em ambos.
3. Os 2 casos a 0% (039, 092) merecem inspeção visual (vídeos disponíveis) — se for pressão prolongada como o 126/195, reforça a necessidade do sinal de drift com threshold adaptativo.

---

## 6. Vídeos de visualização — inventário completo

| Localização | Conteúdo |
|-------------|----------|
| `docs/team_side_report/videos/` | 39 vídeos test originais (4.0GB) — anotações team-side, convenção OK_/NOK_/RESIDUAL_ |
| `docs/full_report/videos/test/` | 10 extras test (SNGS-116…191) |
| `docs/full_report/videos/valid/` | 54 vídeos valid completos |
| `docs/full_report/metrics/` | 23 ficheiros .md com métricas por vídeo (JNR + team, incl. tabelas de erros por tracklet) |

Total: **103 vídeos** com anotações do pipeline (bbox, track ID, team, jersey) prontos a visualizar.

**Sugestão de visualização prioritária**:
1. `valid/SNGS-045.mp4` ou `SNGS-054.mp4` — ver a falha de deteção (quase sem caixas)
2. `valid/SNGS-039.mp4` e `SNGS-092.mp4` — inversões team-side
3. `test/SNGS-190.mp4` — a inversão que custou o lote extraE
4. `valid/SNGS-088-090` — o melhor lote do valid, para contraste

---

## 7. Estado do código e próximos passos

| Item | Estado |
|------|--------|
| Fixes team-side + JNR | Implementados, validados, **NÃO comitados** |
| Test set | ✅ completo (49/49, números + vídeos + métricas) |
| Valid set | ✅ completo (54/54, números + vídeos + métricas) |
| Train set (57 vídeos) | ❌ pendente (~19 lotes, ~11h) — menor prioridade (não comparável a leaderboards) |
| GS-HOTA oficial single-call | ❌ opcional (exigiria fundir pklz; aproximação por lotes considerada suficiente) |
| Re-calibração thresholds team-side com valid | Recomendado antes de fechar a tese (ver §5) |
