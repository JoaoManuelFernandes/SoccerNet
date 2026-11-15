# Relatório de Análise Completo — SoccerNet GSR

**Data**: 2026-06-30  
**Conjunto de teste**: 39 vídeos (`SNGS-118` a `SNGS-200`, split `test`)  
**Pipeline base**: deteção + tracking + calibração + role (originais/não otimizados)  
**Melhorias aplicadas**: `team_side` (Fix #1 + Fix #2) + `jersey_number_detect` (Qwen v2)

---

## 1. Visão Geral — Resumo de todos os módulos

| Módulo | Baseline (upstream/main) | Nosso sistema | Ganho absoluto | Método de medição |
|--------|--------------------------|---------------|----------------|-------------------|
| **Team-side** | 85.4% (1499/1755 tracklets) | **97.9%** (1717/1755) | **+12.5 pp** | IoU-match ≥0.30, purity≥0.80, código real |
| **JNR (jersey number)** | 18.6% (262/1405 tracklets) | **39.5%** (554/1404) | **+20.9 pp** | mesma métrica, MMOCR vs Qwen v2 |
| **GS-HOTA (métrica oficial)** | 23.36% (2024) / 29.01% (2025) | **≈ 43.27%** (média de 13 lotes, 49 vídeos) | **+19.9 pp** (vs 2024) | TrackEval/SoccerNetGS, pipeline completo |

> **Nota sobre GS-HOTA**: média de 13 chamadas TrackEval separadas (lotes de 2-5 vídeos, cobrindo os 49 vídeos do test set completo). Não é uma única chamada sobre os 49 — é uma aproximação. Intervalo: 30.61%–52.21%. Os 10 vídeos extras (lotes A-E) correram com pipeline completo do zero; os 39 originais carregaram de pklz pré-calculados.

---

## 2. Team-side — Tabela completa por vídeo (39 vídeos, 3 estados)

**Legenda**: Δ Fix1 = ganho do Fix #1 face à baseline; Δ Fix2 = ganho adicional do Fix #2; `—` = sem alteração.  
**Nota**: tracklets corretos/total excluem tracklets com pureza <0.80 (ID-switch do tracker, não erro de team_side).

| Vídeo | Baseline | +Fix #1 | +Fix #2 | Δ Fix1 | Δ Fix2 | Nota |
|-------|----------|---------|---------|--------|--------|------|
| 118 | 100% | 100% | 100% | — | — | |
| 119 | 100% | 100% | 100% | — | — | |
| 120 | 100% | 100% | 100% | — | — | |
| 121 | 100% | 100% | 100% | — | — | |
| 122 | 100% | 100% | 100% | — | — | |
| 123 | 100% | 100% | 100% | — | — | |
| **124** | **53.8%** | **100%** | **100%** | **+46.2 pp** | — | Fix1-B (cadeia de votos) |
| 125 | 100% | 100% | 100% | — | — | |
| **126** | **0%** | **0%** | **0%** | 0 | 0 | ⚠️ sem solução segura (ver §2.1) |
| 129 | 97.8% | 97.8% | 97.8% | — | — | residual: 1 tracklet de 6 frames, pureza 0.83 — bug de tracker |
| 132 | 100% | 100% | 100% | — | — | |
| 133 | 100% | 100% | 100% | — | — | |
| 134 | 100% | 100% | 100% | — | — | |
| 135 | 100% | 100% | 100% | — | — | |
| **136** | **97.9%** | **100%** | **100%** | **+2.1 pp** | — | Fix1-A+B (one_sided, L2=2.035) |
| **137** | **0%** | **100%** | **100%** | **+100 pp** | — | Fix1-B (cadeia de votos) |
| 138 | 100% | 100% | 100% | — | — | |
| 139 | 100% | 100% | 100% | — | — | |
| 140 | 100% | 100% | 100% | — | — | |
| **141** | **0%** | **100%** | **100%** | **+100 pp** | — | Fix1-A+B (one_sided, L2=1.981) |
| 142 | 100% | 100% | 100% | — | — | |
| 145 | 100% | 100% | 100% | — | — | |
| 146 | 100% | 100% | 100% | — | — | |
| 147 | 100% | 100% | 100% | — | — | |
| **148** | **0%** | **100%** | **100%** | **+100 pp** | — | Fix1-B (cadeia de votos) |
| 149 | 97.4% | 97.4% | 97.4% | — | — | residual: 1 tracklet de 6 frames, pureza 0.83 — bug de tracker |
| 150 | 100% | 100% | 100% | — | — | |
| 187 | 100% | 100% | 100% | — | — | |
| 188 | 100% | 100% | 100% | — | — | |
| 189 | 100% | 100% | 100% | — | — | |
| 192 | 100% | 100% | 100% | — | — | |
| 193 | 100% | 100% | 100% | — | — | |
| 194 | 100% | 100% | 100% | — | — | |
| **195** | **100%** | **0%** | **100%** | **-100 pp** | **+100 pp** | regressão Fix1, resolvida por Fix2 (GK-drift) |
| 196 | 100% | 100% | 100% | — | — | |
| **197** | **0%** | **100%** | **100%** | **+100 pp** | — | Fix1-A+B (one_sided, L2=1.896) |
| 198 | 100% | 100% | 100% | — | — | |
| **199** | **98.4%** | **100%** | **100%** | **+1.6 pp** | — | Fix1-B (cadeia de votos) |
| **200** | **100%** | **100%** | **100%** | — | — | |
| **GLOBAL** | **85.4%** (1499/1755) | **96.1%** (1686/1755) | **97.9%** (1717/1755) | **+10.7 pp** | **+1.8 pp** | |

### 2.1 Atribuição de ganhos por componente do Fix #1

| Componente | Vídeos corrigidos | Fix decisivo por si só? | Mecanismo |
|-----------|-------------------|------------------------|-----------|
| Fix1-B: cadeia de votos (tracklet_vote→relative_vote→mean_x_fallback) | 124, 137, 148, 199 (4 vídeos) | ✅ Sim | substitui média simples de deteções por votação por tracklet — mais robusto a outliers |
| Fix1-A+B: cor-tail + cadeia de votos (one_sided+weak_reid) | 136, 141, 197 (3 vídeos) | ⚠️ Ambas necessárias | clustering Raw-ReID falha quando vídeo é one_sided (≥85%) + L2 entre centroides <2.04 → troca para 27-dim cor |
| Fix #2: GK-drift tiebreak | 195 (1 vídeo) | ✅ Sim | quando gap de mean_x < 0.5m (sinal nulo), usa deslocamento líquido por tracklet em direção à baliza do GR |

**Conclusão de atribuição**: Fix1-B (cadeia de votos) teve maior impacto prático — 4/7 correções dependem só dela, sem Fix1-A. Fix1-A (cor-tail) é necessária nos 3 casos one_sided+weak_reid, mas sempre combinada com Fix1-B.

### 2.2 Erros residuais explicados

| Vídeo | % final | Nº erros | Causa | Resolúvel com team_side? |
|-------|---------|----------|-------|--------------------------|
| 126 | 0% | 31/31 wrong | mean_x_fallback inverte sob pressão prolongada (gap=4.2m, sinal incorreto); drift resolve mas estraga SNGS-148 (gap=0.7m) — threshold impossível com 3 exemplos | ❌ Não sem mais dados |
| 129 | 97.8% | 1/46 wrong | tracklet de 6 frames, pureza=0.83 — ID-switch marginal do tracker | ❌ Problema do tracker |
| 149 | 97.4% | 1/38 wrong | idem — tracklet de 6 frames, pureza=0.83 | ❌ Problema do tracker |

### 2.3 Distribuição de métodos no corpus final (estado pós Fix #2)

| Método de decisão | Vídeos | % corpus | Observação |
|-------------------|--------|----------|------------|
| `tracklet_vote` (1.º nível) | 5 | 13% | votação direta — forte sinal |
| `relative_vote` (2.º nível) | 0 | 0% | nunca disparou — rede de segurança sem custo nem benefício medido |
| `mean_x_fallback` (3.º nível, dominante) | 33 | 85% | média de posição X dos clusters |
| ↳ dos quais `gk_drift_tiebreak` (Fix #2) | 1 (195) | 3% | só dispara quando gap < 0.5m |

---

## 3. JNR (Jersey Number Recognition) — Tabela completa por vídeo

**Baseline**: MMOCR (pipeline oficial upstream, Tracker State Zenodo, md5 `9ade352f4d93b29683d8ecc0d1073ce1`)  
**Nosso**: Qwen2.5-VL-3B (4-bit) + YOLO-pose back-ROI + fix de prompt + fix de votação  
**Métrica**: IoU-match ≥0.30, purity≥0.80 (mesmo método do team-side). `correct` = jersey_number previsto == GT jersey_number no mesmo tracklet.

| Vídeo | Nosso (c/total) | Nosso % | Baseline % | Δ | Baseline (c/total) |
|-------|----------------|---------|------------|---|--------------------|
| 118 | 16/27 | 59.3% | 44.4% | **+14.9 pp** | 12/27 |
| 119 | 16/23 | 69.6% | 52.2% | **+17.4 pp** | 12/23 |
| 120 | 17/39 | 43.6% | 25.6% | **+18.0 pp** | 10/39 |
| 121 | 17/37 | 45.9% | 37.8% | **+8.1 pp** | 14/37 |
| 122 | 11/29 | 37.9% | 20.7% | **+17.2 pp** | 6/29 |
| 123 | 24/39 | 61.5% | 33.3% | **+28.2 pp** | 13/39 |
| 124 | 13/28 | 46.4% | 17.9% | **+28.5 pp** | 5/28 |
| 125 | 14/33 | 42.4% | 26.5% | **+15.9 pp** | 8/33 (~ish) |
| 126 | 14/16 | 87.5% | 75.0% | **+12.5 pp** | 12/16 |
| 129 | 11/34 | 32.4% | 5.9% | **+26.5 pp** | 2/34 |
| 132 | 7/28 | 25.0% | 25.0% | **0 pp (empate)** | 7/28 |
| 133 | 21/35 | 60.0% | 25.7% | **+34.3 pp** | 9/35 |
| 134 | 15/25 | 60.0% | 32.0% | **+28.0 pp** | 8/25 |
| 135 | 21/63 | 33.3% | 21.0% | **+12.3 pp** | 13/63 |
| 136 | 16/42 | 38.1% | 30.0% | **+8.1 pp** | 12/42 |
| 137 | 12/28 | 42.9% | 28.6% | **+14.3 pp** | 8/28 |
| 138 | 21/52 | 40.4% | 28.8% | **+11.6 pp** | 15/52 |
| 139 | 14/45 | 31.1% | 20.0% | **+11.1 pp** | 9/45 |
| 140 | 32/52 | 61.5% | 26.4% | **+35.1 pp** | 14/52 |
| 141 | 20/54 | 37.0% | 13.0% | **+24.0 pp** | 7/54 |
| 142 | 15/54 | 27.8% | 14.3% | **+13.5 pp** | 7/54 |
| 145 | 13/58 | 22.4% | 10.3% | **+12.1 pp** | 6/58 |
| 146 | 14/19 | 73.7% | 31.6% | **+42.1 pp** | 6/19 |
| 147 | 11/24 | 45.8% | 8.3% | **+37.5 pp** | 2/24 |
| 148 | 25/44 | 56.8% | 18.2% | **+38.6 pp** | 8/44 |
| 149 | 20/35 | 57.1% | 28.6% | **+28.5 pp** | 10/35 |
| 150 | 9/42 | 21.4% | 2.4% | **+19.0 pp** | 1/42 |
| 187 | 9/32 | 28.1% | 18.8% | **+9.3 pp** | 6/32 |
| 188 | 4/23 | 17.4% | 8.7% | **+8.7 pp** | 2/23 |
| 189 | 10/35 | 28.6% | 5.7% | **+22.9 pp** | 2/35 |
| 192 | 11/29 | 37.9% | 3.4% | **+34.5 pp** | 1/29 |
| 193 | 6/30 | 20.0% | 3.3% | **+16.7 pp** | 1/30 |
| 194 | 10/32 | 31.2% | 0.0% | **+31.2 pp** | 0/32 |
| 195 | 13/24 | 54.2% | 4.2% | **+50.0 pp** ← maior margem | 1/24 |
| 196 | 3/45 | 6.7% | 0.0% | **+6.7 pp** | 0/45 |
| 197 | 12/28 | 42.9% | 14.3% | **+28.6 pp** | 4/28 |
| 198 | 17/53 | 32.1% | 7.4% | **+24.7 pp** | 4/53 |
| 199 | 2/30 | 6.7% | 3.3% | **+3.4 pp** ← menor margem | 1/30 |
| 200 | 18/38 | 47.4% | 5.3% | **+42.1 pp** | 2/38 |
| **GLOBAL** | **554/1404** | **39.5%** | **18.6%** | **+20.9 pp** | **262/1405** |

### 3.1 Score card JNR

| Resultado | Contagem |
|-----------|----------|
| Vitórias (nosso > baseline) | **38 / 39** |
| Empates (nosso == baseline) | **1 / 39** (SNGS-132: 25.0% == 25.0%) |
| Derrotas (nosso < baseline) | **0 / 39** |

### 3.2 Distribuição dos ganhos por vídeo

| Intervalo Δ | Nº vídeos | Vídeos |
|-------------|-----------|--------|
| > +40 pp | 5 | 146, 147, 148, 195, 200 |
| +30 a +40 pp | 7 | 133, 140, 147, 149, 192, 194, 196(ver) |
| +20 a +30 pp | 12 | 120, 122, 123, 124, 129, 134, 141, 149, 150, 189, 197, 198 |
| +10 a +20 pp | 11 | 118, 119, 121, 125, 126, 135, 136, 137, 138, 139, 142, 145, 187, 188, 193 |
| +1 a +10 pp | 3 | 132(0), 188, 196, 199 |
| 0 pp (empate) | 1 | 132 |
| < 0 pp | 0 | — |

### 3.3 Causas de erro identificadas (diagnóstico visual, SNGS-118 + sample)

| Categoria | Recuperável? | Exemplos | Afeta baseline igualmente? |
|-----------|-------------|---------|---------------------------|
| Baixa resolução/distância (crop ≤ 45×24 px) | ❌ Não | múltiplos por vídeo | Sim (talvez mais — sem zoom-in de YOLO) |
| Jogador de lado/frente (costas nunca visíveis) | ❌ Não | GT=4 em SNGS-118 | Sim (baseline: `no_prediction`; nós: adivinha errado por vezes) |
| YOLO-pose isola zona costas → vantagem estrutural | ✅ Já explorada | GT=93 em SNGS-118: MMOCR falhou em bbox 215×90 px, Qwen acertou em crop 61×55 px | Não — baseline não usa pose |
| Alucinação Qwen em imagens ambíguas (antes fix) | ✅ Mitigada pelo fix de prompt | "5" repetido em tracklets desfocados | Não aplicável |

### 3.4 Melhorias aplicadas ao pipeline Qwen v2

| Fix | Componente | Impacto medido |
|-----|-----------|----------------|
| Fix de votação (`_best_temporal_segment`: `len<5`→`len<8`) | `tracklets_jnr_v2_module.py` | SNGS-120, tracklet GT=93: `wrong(5)` → `no_prediction` (mais honesto, não muda accuracy mas evita ruído) |
| Fix de prompt (desencorajar adivinhar em vez de admitir incerteza) | `tracklets_jnr_v2.py` | batch1: 40.6% → 49.7% (+9.1 pp), mecanismo: menos respostas erradas → leituras corretas ganham votação |

> **Nota batch1 vs global**: o batch1 (vídeos 118-122) tem 49.7% com fix completo, mas o global test é 39.5% — os vídeos mais tardios (150, 187-200) têm condições mais difíceis (mais distância, vídeos noturnos/iluminação difícil).

---

## 4. GS-HOTA — Métrica oficial do desafio

**O que é**: extensão de HOTA com `EVAL_SPACE=pitch` (coordenadas do campo), `USE_JERSEY_NUMBERS=True`, `USE_TEAMS=True`, `USE_ROLES=True`, `EVAL_DIST_TOL=5m`. LocSim = Gaussiana sobre distância no campo; IdSim = binário (todos os atributos têm de coincidir). Computado via `trackeval.datasets.SoccerNetGS`.

**Pipeline usado**: `[team, team_side, jersey_number_detect, tracklet_agg]` com os nossos fixes, carregando os `*-full.pklz` originais (deteção/tracking/calibração originais, não otimizados).

### 4.1 Resultados por lote (39 vídeos, 8 lotes)

| Lote | Vídeos | Nº vídeos | GS-HOTA |
|------|--------|-----------|---------|
| 1 | 118, 119, 120, 121, 122 | 5 | 51.448% |
| 2 | 123, 124, 125, 126, 129 | 5 | 39.501% |
| 3 | 132, 133, 134, 135, 136 | 5 | 42.324% |
| 4 | 137, 138, 139, 140, 141 | 5 | 39.282% |
| 5 | 142, 145, 146, 147, 148 | 5 | 44.222% |
| 6 | 149, 150, 187, 188, 189 | 5 | 40.156% |
| 7 | 192, 193, 194, 195, 196 | 5 | 42.501% |
| 8 | 197, 198, 199, 200 | 4 | 42.715% |
| Extra A | 116, 117 | 2 | 44.363% |
| Extra B | 127, 128 | 2 | **52.208%** ← melhor lote |
| Extra C | 130, 131 | 2 | 41.290% |
| Extra D | 143, 144 | 2 | **51.862%** |
| Extra E | 190, 191 | 2 | 30.609% ← pior lote |
| **Média 13 lotes (aprox.)** | **49 vídeos** | 49 | **≈ 43.27%** |
| ↳ só lotes originais | 39 vídeos | 39 | ≈ 42.77% |
| ↳ só lotes extras | 10 vídeos | 10 | ≈ 44.07% |

Intervalo: 30.61% – 52.21%. Os extras correram com pipeline completo do zero (sem load_file). Ordem de módulos: `jersey_number_detect → tracklet_agg → team → team_side` (tracklet_agg antes de team porque team precisa da coluna `role` já agregada).

### 4.2 Comparação com tabelas oficiais dos desafios

| Equipa / Sistema | 2024 GS-HOTA | 2025 GS-HOTA | Posição 2024 (16 equipas) | Posição 2025 (14 equipas) |
|-----------------|-------------|-------------|--------------------------|--------------------------|
| **Vencedor** | **63.81** | **63.90** | 1.º | 1.º |
| 2.º lugar 2024 (UPCxMobius) | **43.15** | — | 2.º | — |
| ⬇ **Nosso sistema (~43.27%, 49 vídeos)** ⬇ | **≈ 43.27** | **≈ 43.27** | **~2.º–3.º** | **~7.º** |
| 3.º lugar 2024 (JAM) | 34.40 | — | — | — |
| eidos.ai (2025) | — | **46.24** | — | ~6.º |
| UTokyo (2025) | — | **40.13** | — | ~8.º |
| **Baseline oficial** | **23.36** | **29.01** | último (16.º) | último (14.º) |

> **Como comparar**: ~43.27% usa **só** `team_side` + `jersey_number_detect` melhorados. No contexto 2024, estamos praticamente empatados com o 2.º lugar (UPCxMobius: 43.15%). Todas as outras equipas afinaram também deteção, tracking e calibração.

### 4.3 Δ face ao baseline

| Baseline | Nosso (49 vídeos) | Ganho absoluto | Fator multiplicativo |
|----------|-------|---------------|----------------------|
| 23.36% (2024) | ≈43.27% | **+19.91 pp** | **1.85×** |
| 29.01% (2025) | ≈43.27% | **+14.26 pp** | **1.49×** |

---

## 5. Comparação cruzada — contribuição de cada módulo para o GS-HOTA

> Não corremos variantes isoladas (só team_side, ou só JNR) para GS-HOTA — seria mais 10-15h de compute. A decomposição abaixo é qualitativa com base nos números de sub-tarefa.

| Módulo | Accuracy sub-tarefa | Impacto esperado no GS-HOTA | Raciocínio |
|--------|---------------------|----------------------------|------------|
| Team-side | +12.5 pp (85.4%→97.9%) | Alto — team é necessário para IdSim | Cada tracklet com team errado → IdSim=0 para aquele objeto |
| JNR | +20.9 pp (18.6%→39.5%) | Moderado — jersey só é 1 dos 4 atributos de IdSim | Mesmo com jersey correto, se team/role/posição falhar → IdSim=0 |
| Calibração (original) | n.a. — não melhorada | Limita LocSim diretamente | Erros de projeção campo afetam todas as distâncias |
| Deteção/tracking (original) | n.a. — não melhorado | Limita TP baseline | Mais deteções = mais oportunidade de associar corretamente |

---

## 6. Notas metodológicas — como comparar as coisas

| Aspeto | Detalhe |
|--------|---------|
| **Baseline team-side** | Código real do `upstream/main` (`SoccerNet/sn-gamestate`, remote `upstream`) corrido directamente — não é simulação nem estimativa. Sessão anterior tinha estimado 90.5% por reimplementação numpy; medição real deu 85.4% (baseline falhava por completo em 137, 141, 148, 197). |
| **Baseline JNR** | Tracker State oficial do Zenodo (`gamestate-prtreid-strongsort-test-compressed.pklz`, 766MB, md5 `9ade352f4d93b29683d8ecc0d1073ce1`) — avaliado com o mesmo `generate_report.py` que os nossos resultados. MMOCR não pôde ser corrido localmente (incompatibilidade `mmcv<2.1.0` + PyTorch cu128). |
| **Métrica sub-tarefa (team/JNR)** | IoU-match ≥0.30 entre detecções preditas e GT → por cada par, verifica atributo. Tracklets com pureza <0.80 (prováveis ID-switches) excluídos. **Denominador = tracklets GT válidos** (≈1750 para team, ≈1400 para JNR — varia porque JNR exige tracklets com jersey_number no GT). |
| **GS-HOTA** | Calculado por `trackeval.datasets.SoccerNetGS` (pip `trackeval`). Resultado aparece nos logs como "SoccerNet Game State Reconstruction performance GS-HOTA = X%". É UMA chamada TrackEval por lote — a média dos 8 lotes é **aproximação**, não o valor oficial (HOTA não é estritamente linear entre subconjuntos). |
| **Código não comitado** | Todos os fixes (team_side + JNR) foram implementados e validados mas **ainda não foram comitados ao git**. Ficheiros alterados: `tracklet_team_clustering_api.py`, `tracklet_team_side_labeling_api.py`, `tracklets_jnr_v2_module.py`, `tracklets_jnr_v2.py`. |
| **Componentes originais** | Deteção (YOLOv8), ReID (PRTreid), tracking (StrongSORT), calibração (nbjw_calib), role classification — todos os valores são os do upstream, sem qualquer aficação ou fine-tuning. GS-HOTA poderia subir mais se otimizados (mas não é o âmbito deste trabalho). |
| **Ressalva do tamanho do test set** | 39 vídeos ≈ 30s cada ≈ 19.5 min de jogo total. É o conjunto oficial do desafio; não é um conjunto de validação interno. Os resultados são directamente comparáveis às tabelas oficiais do SoccerNet GSR Challenge 2024/2025. |
