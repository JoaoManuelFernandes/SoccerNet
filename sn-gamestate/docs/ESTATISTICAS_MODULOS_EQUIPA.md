# Estatísticas reais dos módulos team / team_side (extraídas dos registos)

**Data:** 2026-09-13
**Fonte treino:** `docs/render_logs/SNGS-*.log` (57 vídeos, render de 2026-09-09 com o código atual)
**Fonte teste:** `docs/team_side_report/REPORT.md` (39 vídeos com estado, secção "método")
**Extração:** expressões regulares sobre o registo com o espaço em branco removido
(o handler rich parte as linhas em colunas estreitas). Tabelas por vídeo em
`train_cluster_branch.tsv` e `train_side_method.tsv` (copiadas abaixo).

## 1. Módulo team: que ramo do agrupamento foi usado

| Conjunto | ReID cru | Cor (27 d) | dos quais só por silhueta |
|---|---|---|---|
| treino (57) | 49 | 8 | 1 (SNGS-069) |
| teste (39)  | 35 | 4 (136, 141, 195, 197) | 0 (a silhueta ainda não existia no REPORT.md; ver nota) |

Treino, vídeos que foram para a cor (legacy = gatilho herdado ≥85% um lado + L2 < 2.04):
SNGS-064 (legacy, sil 0.277, L2 1.467), 067 (legacy, 0.268, 1.536), 068 (legacy, 0.251, 1.511),
**069 (só silhueta, 0.234, 1.566)**, 071 (legacy, 0.269, 1.534), 072 (legacy, 0.281, 1.686),
075 (legacy, 0.291, 1.645), 077 (legacy, 0.265, 1.636).
Guarda de degenerescência disparou 1 vez: SNGS-061 (menor grupo = 14% < 20%, manteve ReID).
Silhueta no treino: mín 0.234, mediana 0.391, máx 0.456. L2 entre centróides: mín 1.467, mediana 2.261, máx 2.681.

Nota: nos 4 vídeos de teste em que o gatilho herdado disparou, o L2 estava a
1.799 (195), 1.896 (197), 1.981 (141) e 2.035 (136); o limiar 2.04 fica 0.005
acima do maior. É um limiar ajustado a esses casos, não uma constante com
significado geral.

## 2. Módulo team_side: que critério decidiu

| Conjunto | 1. votação por tracklet | 2. votação relativa | 3. média de x | 4. desempate GR |
|---|---|---|---|---|
| treino (57) | 17 | 12 | 28 | 0 |
| teste (39)  | 5  | 0  | 33 | 1 (SNGS-195) |

Leitura honesta: na maioria dos vídeos a decisão continua a sair do critério 3,
que é a mesma comparação de médias da solução original (com filtro |x| ≤ 60 m).
Os critérios 1 e 2 só decidem quando os grupos estão claramente em metades
diferentes ou com assimetria ≥ 10 pp.

## 3. SNGS-069 (treino), rasto real

- team: gatilho de silhueta (0.234 < 0.28), legacy False → agrupou pela cor; grupos 11/39
  (menor = 22%, passou a guarda de 20% por pouco).
- team_side: votação por tracklet: C0 = 0 right / 10 left → left; C1 = 16 right / 22 left
  (42%) → indeciso → recebe o oposto (right). Método: tracklet_vote.
- GR: a anotação não tem guarda-redes, mas o pipeline classificou um como GR (47 deteções, left).
- Veredicto visual: "tudo da mesma equipa". Consistente com 39 dos 50 tracklets a right.
- Por apurar: se o ReID cru (que o gatilho de silhueta rejeitou) daria uma divisão melhor.

## 4. SNGS-113 (treino), rasto real

Ver FALHAS_VALIDADAS_VISUALMENTE.md (decidiu no critério 2 com 80% vs 96%, resultado invertido).

## Anexo: tabelas por vídeo

### treino, ramo do agrupamento (vídeo, ramo, legacy, silhueta, L2)
```
SNGS-060	reid	None	0.29	1.585
SNGS-061	reid	None	0.259	1.492
SNGS-062	reid	None	0.285	1.558
SNGS-063	reid	None	0.285	1.589
SNGS-064	cor	True	0.277	1.467
SNGS-065	reid	None	0.292	1.673
SNGS-066	reid	None	0.292	1.554
SNGS-067	cor	True	0.268	1.536
SNGS-068	cor	True	0.251	1.511
SNGS-069	cor	False	0.234	1.566
SNGS-070	reid	None	0.3	1.713
SNGS-071	cor	True	0.269	1.534
SNGS-072	cor	True	0.281	1.686
SNGS-073	reid	None	0.318	1.72
SNGS-074	reid	None	0.317	1.729
SNGS-075	cor	True	0.291	1.645
SNGS-076	reid	None	0.301	1.786
SNGS-077	cor	True	0.265	1.636
SNGS-097	reid	None	0.456	2.681
SNGS-098	reid	None	0.423	2.556
SNGS-099	reid	None	0.422	2.54
SNGS-100	reid	None	0.444	2.616
SNGS-101	reid	None	0.451	2.679
SNGS-102	reid	None	0.424	2.607
SNGS-103	reid	None	0.441	2.545
SNGS-104	reid	None	0.409	2.527
SNGS-105	reid	None	0.409	2.491
SNGS-106	reid	None	0.375	2.332
SNGS-107	reid	None	0.454	2.659
SNGS-108	reid	None	0.417	2.475
SNGS-109	reid	None	0.421	2.484
SNGS-110	reid	None	0.427	2.551
SNGS-111	reid	None	0.391	2.452
SNGS-112	reid	None	0.426	2.604
SNGS-113	reid	None	0.395	2.528
SNGS-114	reid	None	0.431	2.607
SNGS-115	reid	None	0.438	2.472
SNGS-151	reid	None	0.343	2.092
SNGS-152	reid	None	0.438	2.261
SNGS-153	reid	None	0.355	2.159
SNGS-154	reid	None	0.437	2.363
SNGS-155	reid	None	0.426	2.39
SNGS-156	reid	None	0.362	2.206
SNGS-157	reid	None	0.391	2.36
SNGS-158	reid	None	0.394	2.307
SNGS-159	reid	None	0.445	2.485
SNGS-160	reid	None	0.372	2.12
SNGS-161	reid	None	0.425	2.391
SNGS-162	reid	None	0.426	2.483
SNGS-163	reid	None	0.386	2.29
SNGS-164	reid	None	0.373	2.262
SNGS-165	reid	None	0.304	1.967
SNGS-166	reid	None	0.335	2.05
SNGS-167	reid	None	0.428	2.258
SNGS-168	reid	None	0.408	2.206
SNGS-169	reid	None	0.358	2.078
SNGS-170	reid	None	0.424	2.208
```

### treino, critério do lado (vídeo, método, GR detetado)
```
SNGS-060	mean_x_fallback	'right':131,'left':125
SNGS-061	mean_x_fallback	'right':266,'left':126
SNGS-062	tracklet_vote	'right':129,'left':1
SNGS-063	tracklet_vote	'left':102,'right':5
SNGS-064	mean_x_fallback	'left':326
SNGS-065	tracklet_vote	'left':70,'right':47
SNGS-066	mean_x_fallback	'left':513
SNGS-067	mean_x_fallback	'left':208,'right':4
SNGS-068	mean_x_fallback	sem GK detetado
SNGS-069	tracklet_vote	'left':47
SNGS-070	tracklet_vote	'right':169,'left':118
SNGS-071	mean_x_fallback	'right':592
SNGS-072	mean_x_fallback	'right':66
SNGS-073	mean_x_fallback	'right':304,'left':78
SNGS-074	relative_vote	'right':59
SNGS-075	mean_x_fallback	'right':916
SNGS-076	tracklet_vote	sem GK detetado
SNGS-077	mean_x_fallback	'left':233
SNGS-097	mean_x_fallback	'right':452
SNGS-098	relative_vote	'left':167,'right':49
SNGS-099	mean_x_fallback	'left':247,'right':241
SNGS-100	tracklet_vote	'left':300,'right':140
SNGS-101	relative_vote	'right':145,'left':127
SNGS-102	mean_x_fallback	'right':340
SNGS-103	relative_vote	'left':416
SNGS-104	mean_x_fallback	'right':171
SNGS-105	mean_x_fallback	'right':394
SNGS-106	mean_x_fallback	'left':511
SNGS-107	tracklet_vote	'right':307
SNGS-108	relative_vote	'left':282,'right':146
SNGS-109	relative_vote	'left':295
SNGS-110	mean_x_fallback	'left':232
SNGS-111	relative_vote	'left':263,'right':155
SNGS-112	relative_vote	'right':286
SNGS-113	relative_vote	'left':457
SNGS-114	mean_x_fallback	'left':227
SNGS-115	mean_x_fallback	'right':194
SNGS-151	relative_vote	'left':112,'right':94
SNGS-152	mean_x_fallback	'right':688
SNGS-153	relative_vote	'left':256,'right':215
SNGS-154	mean_x_fallback	'left':421
SNGS-155	mean_x_fallback	'left':402
SNGS-156	mean_x_fallback	'left':683
SNGS-157	tracklet_vote	'left':496
SNGS-158	mean_x_fallback	'right':359,'left':113
SNGS-159	mean_x_fallback	'left':361
SNGS-160	tracklet_vote	'right':73
SNGS-161	tracklet_vote	'left':234,'right':44
SNGS-162	mean_x_fallback	'right':413,'left':76
SNGS-163	tracklet_vote	'left':171,'right':92
SNGS-164	mean_x_fallback	'left':158
SNGS-165	tracklet_vote	'left':288,'right':253
SNGS-166	tracklet_vote	'right':97
SNGS-167	tracklet_vote	'right':123,'left':2
SNGS-168	tracklet_vote	'right':90
SNGS-169	tracklet_vote	'right':310,'left':163
SNGS-170	relative_vote	'left':327,'right':76
```

## 5. Rastos usados na figura fig_cascata_exemplos (2026-09-13)

- **SNGS-060** (treino, docs/render_logs/SNGS-060.log): votos A 15R/19L (44%), B 22R/20L (52%) → indecisos;
  relativa 44% vs 52%, diff 8 pp < 10 → falha; média x A −2.0 m, B +2.4 m → B right. GR detetados nos dois lados (131 R, 125 L).
- **SNGS-195** (teste): reexecução dos módulos team + team_side ATUAIS sobre o estado antigo
  `sn-gamestate/outputs/sn-gamestate/2026-06-22/08-37-16/states/sn-gamestate-test-batch7-FIXED.pklz`
  (tem embeddings e embeddings_with_colors). Agrupamento: ramo da cor (legacy=True, sil 0.320, L2 1.799), grupos 20/35.
  Votos A 4R/16L (80% L → left), B 5R/30L (86% L → left) → conflito; relativa 20% vs 14%, diff 6 pp → falha;
  média x A −23.0, B −23.0 (gap 0.01 m) → desempate GR: gk_side=left (315 deteções), toward_gk A 1.44, B 5.75 → B right.
  Coincide com docs/team_side_report/REPORT.md §3 (−23.03/−23.04; −1.44/−5.75 no sinal antigo).
- **SNGS-113**: ver secção 4 e FALHAS_VALIDADAS_VISUALMENTE.md.
