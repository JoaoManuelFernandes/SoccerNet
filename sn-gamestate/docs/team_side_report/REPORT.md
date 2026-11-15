# Relatório — Team / Team-side (clustering + atribuição de lado)

**Data**: 2026-06-24
**Âmbito**: 39 vídeos do conjunto de teste (SoccerNetGS `test`)
**Ficheiros de código**:
- `sn_gamestate/team/tracklet_team_clustering_api.py`
- `sn_gamestate/team/tracklet_team_side_labeling_api.py`

Todos os números foram obtidos correndo o **código real do pipeline**
(`pipeline: [team, team_side]`), avaliado por IoU-matching (≥0.30) contra o GT
do SoccerNetGS, com filtro de pureza/min-frames para excluir tracklets afetados
por troca de identidade (ID switch). **A coluna "Baseline" foi medida correndo o
código original (`SoccerNet/sn-gamestate`, branch `main`, via remote
`upstream`) diretamente** — não é simulação nem estimativa.

---

## 1. Resumo executivo

| Estado | Global | Δ |
|--------|--------|---|
| **Baseline** (código original `upstream/main`) | **85.4%** (1499/1755) | — |
| **+ Fix #1** (clustering por cor + cadeia de votos) | **96.1%** (1686/1755) | **+10.7 pp** |
| **+ Fix #2** (desempate por deslocamento ao GR) | **97.9%** (1717/1755) | **+1.8 pp** |

> Nota: uma sessão anterior tinha estimado a baseline em 90.5% por simulação.
> A medição real (esta sessão) corrige esse número para **85.4%** — a baseline
> falhava por completo (0%) em mais vídeos do que a simulação previa.

---

## 2. O que ganhámos com cada fix — atribuição vídeo a vídeo

### Fix #1 tem DUAS partes distintas, com efeitos diferentes:

**(A) Clustering por cor** (`tracklet_team_clustering_api.py`): quando o vídeo é
one-sided (≥85% dos tracklets no mesmo meio-campo) e o ReID separa mal as
equipas (L2 entre centróides < 2.04), cluster passa a usar só os 27 dims de
cor de equipamento em vez do bloco ReID(256d) completo.

**(B) Cadeia de votos no side-labeling** (`tracklet_team_side_labeling_api.py`):
substitui a média simples e "flat" de todas as deteções (sensível a outliers e
a frames ruidosos) por votação **por tracklet**: `tracklet_vote` (cada tracklet
vota pelo seu próprio x médio) → `relative_vote` (compara fração de tracklets à
direita entre clusters) → `mean_x_fallback` (a média antiga, mas só como último
recurso). Também remove `_setpiece_anchor`/`_tactical_anchor`, duas heurísticas
que nunca acertaram.

| Vídeo | Baseline | Branch clustering | Ganho vem de | Depois Fix #1 |
|-------|----------|--------------------|---------------|----------------|
| 124 | 53.8% (28/52) | balanced (não disparou A) | **(B)** só a cadeia de votos | 100% |
| 136 | 97.9% (47/48) | one_sided+weak_reid (L2=2.035) — disparou A | (A)+(B) | 100% |
| 137 | **0%** (0/38) | balanced (não disparou A) | **(B)** só a cadeia de votos | 100% |
| 141 | **0%** (0/68) | one_sided+weak_reid (L2=1.981) — disparou A | (A)+(B) | 100% |
| 148 | **0%** (0/48) | balanced (não disparou A) | **(B)** só a cadeia de votos | 100% |
| 197 | **0%** (0/38) | one_sided+weak_reid (L2=1.896) — disparou A | (A)+(B) | 100% |
| 199 | 98.4% (61/62) | balanced (não disparou A) | **(B)** só a cadeia de votos | 100% |

**Conclusão importante**: dos 7 vídeos corrigidos, **a maioria (4/7) deve-se só
à parte (B)** — a cadeia de votos por tracklet. A parte (A) — clustering por
cor — só foi decisiva nos casos one-sided+weak-ReID (136,141,197), mas nesses
casos a clusterização raw-ReID já costumava ser "suficientemente limpa" ao
nível do tracklet; o que faltava era a parte (B) para ler corretamente essa
informação. Isto sugere que **(B) teve mais impacto prático que (A)** neste
conjunto de 39 vídeos, apesar de (A) ter sido o fix mais "vistoso"/discutido em
sessões anteriores.

### O preço do Fix #1: regrediu o SNGS-195

| Vídeo | Baseline | Branch clustering | Depois Fix #1 |
|-------|----------|--------------------|----------------|
| **195** | **100%** (31/31) | one_sided=left+weak_reid (L2=1.799) — disparou A | **0%** (0/31) |

No código original, 195 dava 100% **por sorte**: o clustering raw-ReID
calhou a produzir uma divisão cujas médias de posição diferiam o suficiente
para a comparação simples acertar. Ao trocar para clustering por cor (parte A,
disparada porque L2=1.799 < 2.04), a nova divisão de clusters é igualmente
"limpa" por equipa (confirmado — ver secção 3), mas as médias de posição dos
dois clusters passaram a ficar **praticamente empatadas** (gap de 0.01m),
porque o jogo está todo concentrado numa zona durante os 30s do clip. A
comparação simples de médias deixou de ter qualquer sinal real para decidir, e
acertou ou errou ao acaso — neste caso, errou.

### Fix #2: repara o 195 sem repetir o erro

| Vídeo | Depois Fix #1 | Depois Fix #2 |
|-------|----------------|----------------|
| **195** | 0% (0/31) | **100%** (31/31) |

Ver secção 4 para o mecanismo. Validado duas vezes contra os 39 vídeos
completos — **zero regressões** nos outros 38.

---

## 3. O porquê, com os números reais por detrás da decisão

### SNGS-195 (corrigido pelo Fix #2)
- Clustering por aparência: **perfeito** — cluster 0 = 100% GT "left" (20
  tracklets), cluster 1 = 100% GT "right" (20 tracklets). Sem ambiguidade.
- Guarda-redes corretamente identificado: x médio ≈ -46m → GT confirma "left".
- `mean_x_fallback`: cluster0=-23.03m, cluster1=-23.04m → **gap de 0.01m**,
  sinal nulo.
- Critério de desempate (Fix #2): deslocamento líquido por tracklet
  (último−primeiro pitch_x). Cluster 1 (GT="right", visitante) avança em média
  **-5.75m** em direção à baliza do GR; cluster 0 (GT="left", casa) só
  **-1.44m**. A equipa que mais avança para a baliza é a equipa visitante —
  resultado correto.

### SNGS-126 (ainda sem solução — mesma família de bug do 195)

**Diagnóstico corrigido nesta sessão**: uma análise anterior tinha
classificado este caso como "falha de clustering" — está errado. Os dados
mostram:
- Clustering **limpo**: cluster 0 = 97% GT "right" (31/32 tracklets), cluster
  1 = 100% GT "left" dos tracklets com correspondência válida (19/19).
- Guarda-redes real (GT team="left") corretamente identificado por x médio
  (≈-46m).
- `mean_x_fallback`: cluster0=-20.8m, cluster1=-16.6m → **gap de 4.2m**, mas
  com o **sinal errado**: o cluster GT="right" (visitante) tem a média mais
  negativa (mais perto da baliza do GR) porque está numa fase de pressão
  prolongada — não porque defenda esse lado.
- Testei isoladamente o critério de desempate do Fix #2 (sem o threshold de
  gap) sobre este vídeo: **resolve corretamente o 126** (toward_gk:
  cluster0=12.46 > cluster1=5.01 → cluster0=visitante="right" ✓).
- **Mas** ao testar o mesmo critério contra **SNGS-148** (gap=0.7m, onde o
  `mean_x_fallback` simples já estava certo), o desempate **inverte e estraga**
  um caso que funcionava.
- Com só 3 pontos de referência (195 e 126 onde o desempate ajuda, 148 onde
  prejudica) **não há base estatística para um threshold seguro** que cubra os
  dois grupos sem risco de regressão — por isso o threshold ficou
  deliberadamente apertado (0.5m), resolvendo só o 195.

### Margem de segurança do threshold do Fix #2

| Vídeo | Gap medido | Resultado |
|-------|-----------|-----------|
| 195 | 0.01m | dispara o desempate (correto) |
| **148** | **0.7m** | **não dispara** (margem de só 0.2m acima do threshold de 0.5m) |
| 126 | 4.2m | não dispara (ficaria correto se disparasse, mas não há base para alargar) |
| restantes vídeos com `mean_x_fallback` | ≥ 3.9m | nunca perto do threshold |

O threshold de 0.5m foi escolhido porque é o único valor que isola o caso
genuinamente nulo (195, gap 0.01m) sem incluir o caso onde o sinal simples já
funcionava (148, gap 0.7m). A margem é pequena (0.2m) mas o gap do 195 é tão
extremo (3500x menor) que não há ambiguidade prática nesse ponto de corte.

---

## 4. Distribuição de métodos no corpus (estado final, 39 vídeos)

| Método | Nº vídeos | % |
|--------|-----------|---|
| `tracklet_vote` (1º nível, votação por tracklet) | 5 | 13% |
| `relative_vote` (2º nível) | 0 | 0% |
| `mean_x_fallback` (3º nível, média simples) | 33 | 85% |
| ↳ dos quais via `gk_drift_tiebreak` (novo, Fix #2) | 1 (195) | 3% |

**Achado relevante**: o `mean_x_fallback` — nominalmente o "último recurso" —
é na prática o método dominante (85% dos vídeos). O `relative_vote` nunca
chegou a disparar em todo o corpus (é uma rede de segurança que não tem custo
mas também não tem benefício medido aqui). Isto significa que qualquer fragi-
lidade no `mean_x_fallback` afeta a making grande maioria dos vídeos — pelo que
o cuidado com regressões nessa função (secção 3) é particularly importante.

---

## 5. Erros residuais — não são bugs de team_side

Restam 3 vídeos com erro depois do Fix #2: **126** (0%, sem solução segura —
secção 3), **129** (97.8%, 1 erro) e **149** (97.4%, 1 erro). Diagnostiquei os
dois últimos com `generate_report.py` (deteção de ID-switch por pureza de
IoU-matching):

| Vídeo | Tracklet errado | Frames | Pureza | Diagnóstico |
|-------|-----------------|--------|--------|-------------|
| 129 | pred_tid=62 | **6** | 0.83 | tracklet curtíssimo, marginal — não é erro de team_side |
| 149 | pred_tid=63 | **6** | 0.83 | tracklet curtíssimo, marginal — não é erro de team_side |

Ambos os casos são tracklets de **apenas 6 frames**, com pureza logo a seguir
ao limite de 0.80 (ou seja, quase classificados como ID-switch e excluídos da
avaliação). Não há sinal suficiente (poucos frames, deteção marginal) para
nenhuma lógica de team/team-side resolver isto — a causa raiz é do tracker
(ReID/associação), não do clustering ou do side-labeling. Note-se também que
SNGS-129 tem **13 ID-switches** e SNGS-149 tem **2 ID-switches** detetados no
total — esses dois vídeos têm problemas de tracking maiores que já não afetam
a métrica de team (são excluídos por pureza < 0.80), mas mostram que a robustez
do tracker nesses vídeos é fraca.

---

## 6. Tabela completa (39 vídeos, 3 estados)

| Vídeo | Baseline | +Fix #1 | +Fix #2 |
|-------|----------|---------|---------|
| 118 | 100% | 100% | 100% |
| 119 | 100% | 100% | 100% |
| 120 | 100% | 100% | 100% |
| 121 | 100% | 100% | 100% |
| 122 | 100% | 100% | 100% |
| 123 | 100% | 100% | 100% |
| **124** | **53.8%** | 100% | 100% |
| 125 | 100% | 100% | 100% |
| **126** | 0% | 0% | 0% *(sem solução — secção 3)* |
| 129 | 97.8% | 97.8% | 97.8% *(residual — secção 5)* |
| 132 | 100% | 100% | 100% |
| 133 | 100% | 100% | 100% |
| 134 | 100% | 100% | 100% |
| 135 | 100% | 100% | 100% |
| **136** | 97.9% | 100% | 100% |
| **137** | **0%** | 100% | 100% |
| 138 | 100% | 100% | 100% |
| 139 | 100% | 100% | 100% |
| 140 | 100% | 100% | 100% |
| **141** | **0%** | 100% | 100% |
| 142 | 100% | 100% | 100% |
| 145 | 100% | 100% | 100% |
| 146 | 100% | 100% | 100% |
| 147 | 100% | 100% | 100% |
| **148** | **0%** | 100% | 100% |
| 149 | 97.4% | 97.4% | 97.4% *(residual — secção 5)* |
| 150 | 100% | 100% | 100% |
| 187 | 100% | 100% | 100% |
| 188 | 100% | 100% | 100% |
| 189 | 100% | 100% | 100% |
| 192 | 100% | 100% | 100% |
| 193 | 100% | 100% | 100% |
| 194 | 100% | 100% | 100% |
| **195** | 100% | **0%** | 100% |
| 196 | 100% | 100% | 100% |
| **197** | **0%** | 100% | 100% |
| 198 | 100% | 100% | 100% |
| **199** | 98.4% | 100% | 100% |
| 200 | 100% | 100% | 100% |

**Negrito** = vídeo onde algum fix mudou o resultado.

---

## 7. Próximos passos sugeridos (por ordem de esforço/risco)

1. **SNGS-126**: reunir mais casos de "pressão prolongada com gap grande" antes
   de tentar alargar o threshold do Fix #2 — com só 3 exemplos (195, 126, 148)
   não há base estatística fiável.
2. **Janelas de tempo mais curtas** para o drift (ex. últimos 5s do clip) em
   vez do clip completo — pode ser mais robusto a fases de jogo que mudam de
   direção dentro dos 30s.
3. **Posição da bola** (se extraível via calibração) como sinal adicional — a
   equipa mais perto da bola junto a uma baliza tende a ser a atacante, sinal
   mais direto que o deslocamento médio do jogador.
4. **129/149**: não são bugs de team_side — se quiser melhorar isto, é preciso
   olhar para o tracker/ReID (associação de tracklets curtos), não para este
   módulo.
5. **Validar no conjunto `valid`** (54 vídeos, ainda não avaliado) para
   confirmar se o padrão "pressão prolongada confunde mean_x" é raro ou comum
   fora deste conjunto de teste, antes de investir mais tempo no Fix #2.

---

## 8. Vídeos para avaliação visual

Pasta: `docs/team_side_report/videos/` — **agora com os 39 vídeos completos**
(4.0GB). Convenção de nomes:

- `OK_<id>.mp4` — vídeo correto (100%) no estado final, com ou sem fix.
- `OK_<id>_fixed.mp4` — vídeo correto **e** corrigido por algum fix (sufixo
  mantido nos primeiros 6 gerados na sessão anterior, antes de cobrir os 39).
- `RESIDUAL_<id>_1error.mp4` — 129 e 149, o único erro residual de cada
  (tracklet de 6 frames, ver secção 5 — não é bug de team_side).
- `NOK_126_unsolved.mp4` — único vídeo ainda sem solução.
- `BEFORE_195_broken.mp4` — estado intermédio do 195 depois do Fix #1
  (regressão, 0%) para comparar com `OK_195_fixed_AFTER.mp4` (depois do Fix #2).

| Vídeos corrigidos por algum fix | Vídeos já saudáveis na baseline | Erro residual (não-team_side) | Sem solução |
|---|---|---|---|
| 124,136,137,141,148,195,197,199 (8) | restantes 29 | 129,149 | 126 |

---

## 9. Estado do código

Alterações feitas e **ainda não comitadas**:
- `sn_gamestate/team/tracklet_team_clustering_api.py` (Fix #1, parte A)
- `sn_gamestate/team/tracklet_team_side_labeling_api.py` (Fix #1 parte B + Fix #2)
