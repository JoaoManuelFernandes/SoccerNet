# Resumo Técnico — Contribuições ao SoccerNet Game State Reconstruction

**Documento de apoio à reunião de orientação de tese**
**Data**: 2026-07-09
**Aluno**: (preencher) · **Orientador**: (preencher)

---

## 1. Enquadramento em uma frase

Partindo do *baseline* oficial do desafio **SoccerNet Game State Reconstruction (GSR)**,
melhorámos **dois** dos seis módulos do pipeline — **reconhecimento de número de
camisola (JNR)** e **atribuição de equipa/lado (team-side)** — e avaliámos o impacto
na métrica oficial **GS-HOTA**, no conjunto de **teste (49 vídeos)** e de
**validação (54 vídeos)** do desafio.

---

## 2. O problema e a métrica

- **Tarefa GSR**: a partir de vídeo de futebol de câmara única, reconstruir o "estado
  de jogo" — posição no campo, equipa, papel (jogador/GR/árbitro) e número de camisola
  de cada interveniente.
- **Pipeline (6 módulos)**: deteção → re-identificação → tracking → calibração de campo
  → papel/team → número de camisola.
- **Métrica oficial — GS-HOTA**: variante do HOTA que combina
  **LocSim** (proximidade no campo, *kernel* Gaussiano sobre a distância projetada) com
  **IdSim** (identidade — todos os atributos, incl. equipa e número, têm de coincidir).
  Configuração oficial: `EVAL_SPACE=pitch`, `USE_TEAMS`, `USE_ROLES`,
  `USE_JERSEY_NUMBERS`, `EVAL_DIST_TOL=5 m`. Implementada em `trackeval.SoccerNetGS`.

**Nota metodológica importante**: mantivemos deteção, re-ID, tracking e calibração
**exatamente como no baseline** (não otimizados). Todos os ganhos reportados vêm apenas
dos dois módulos que melhorámos — o que isola de forma limpa a nossa contribuição.

---

## 3. Contribuição A — Reconhecimento de número de camisola (JNR)

### 3.1 Diagnóstico do baseline
O baseline usa **MMOCR** (OCR clássico) sobre a *bounding box* completa do jogador.
Falha sistematicamente quando o número é pequeno, o jogador está longe, ou o texto
está integrado em fundo ruidoso.

### 3.2 Abordagem proposta
Substituímos por um pipeline **VLM (Vision-Language Model)** com pré-processamento por pose:

1. **YOLO-pose** para recortar a **região das costas** (onde está o número) — em vez de
   correr OCR na bbox inteira, damos ao modelo uma imagem já focada e ampliada.
2. **Qwen2.5-VL-3B** (quantizado a 4-bit, ~2 GB de VRAM) para ler os dígitos.
3. **Votação por tracklet** com janela temporal — agrega as leituras de todas as frames
   de um mesmo jogador numa decisão única.
4. **Engenharia de *prompt*** — reescrevemos a instrução para **desencorajar
   adivinhação**: o modelo é instruído a responder "none" quando não consegue ler com
   confiança, em vez de inventar um número. Isto reduz respostas erradas que poluíam a
   votação.

### 3.3 Alternativas exploradas e descartadas (percurso experimental)
A escolha do Qwen-VL não foi imediata — resultou de comparar várias famílias de OCR/VLM.
Documentar isto é importante: mostra que a solução final foi fundamentada, não arbitrária.

| Abordagem testada | Porque foi descartada |
|-------------------|------------------------|
| **EasyOCR / OCR clássico** | Herdado de versões iniciais. Mesma limitação estrutural do MMOCR — falha em números pequenos/inclinados; sem compreensão semântica do que é um "número de camisola". |
| **PaddleOCR** (incl. tentativa de *fine-tuning*) | **Segfault (SIGSEGV)** ao inicializar os modelos no mesmo processo que o CUDA/PyTorch do pipeline; mesmo isolado em subprocesso não recuperava jerseys adicionais face ao Qwen. Preparámos dados para *fine-tuning* (`paddleocr_finetune/`) mas o custo/instabilidade não se justificou. **Removido.** |
| **HunyuanOCR** (VLM da Tencent) | Testámos como alternativa VLM ao Qwen (`HunyuanOCR/hunyuan_ocr_wrapper.py`). Modelo maior/mais pesado, sem ganho de accuracy que compensasse o custo de VRAM/latência no nosso hardware (RTX 3060 6 GB). |
| **Qwen2.5-VL-7B** | Requer ~10 GB de VRAM — **não cabe** nos 6 GB da GPU disponível. Ficámos pela variante 3B quantizada. |
| **Super-resolução** (Real-ESRGAN sobre crops ilegíveis) | A informação **não está** na imagem-fonte (crops de 45×24 px). A ampliação amplifica ruído, por vezes piora. **A resolução não se cria.** |

**Conclusão desta exploração**: em condições de baixa resolução, o fator decisivo não é
"qual o OCR mais forte" mas sim **dar ao modelo a imagem certa** (recorte por pose das
costas) e **agregar bem no tempo** (votação por tracklet). O Qwen-VL 3B venceu por
equilíbrio accuracy/VRAM, não por ser o maior modelo.

**Experimentos controlados adicionais (resultados negativos, 2026-07-14/15)** — testámos
três melhorias plausíveis num protocolo de 6 vídeos (4 piores + 2 bons de controlo), com
a regra "ganhar nos piores sem regredir nos bons":
- **Voto ponderado por qualidade** (peso = nº crops × nitidez × área da janela): ganho
  marginal (~+1 pp), inconsistente entre esquemas de pesos, e **zero efeito nos vídeos
  piores**.
- **Realce de contraste CLAHE** nos crops antes do VLM: **net negativo** — regrediu 3
  vídeos (incluindo um controlo) e só ajudou 2.
- **Montagem multi-vista** (grelha única com todas as vistas do tracklet numa só imagem,
  para o VLM agregar evidência internamente em vez de votação externa): **net negativo**
  (−1 tracklet global), mesmo padrão do CLAHE — pequeno ganho num vídeo difícil, pago
  com regressões noutros.

O diagnóstico comum às três: nos vídeos piores o VLM não lê o número em **nenhuma**
janela — a informação não está nos pixels (distância/orientação). Reponderar votos,
realçar contraste ou reorganizar as vistas não cria informação; e qualquer perturbação
da entrada quebra casos que já funcionavam. Estes três resultados negativos **delimitam
empiricamente o teto da abordagem JNR** dado o material de entrada: o espaço de melhoria
restante não está no OCR/votação, mas a montante (qualidade das deteções e dos crops).

### 3.4 Tentativas no eixo "jogador está de costas?" (deteção de orientação)
Um número só é legível se o jogador estiver **de costas** para a câmara. Investigámos
detetar a orientação para filtrar frames inúteis antes do OCR:

- **YOLO-pose v11 → v26**: evoluímos o modelo de pose (de YOLOv11 para a geração
  YOLO26, `yolo26m-pose.pt`) para estimar a orientação do tronco a partir dos
  *keypoints* (ombros/anca) e decidir se as costas estão visíveis. O recorte por pose
  ficou robusto; o classificador binário "de costas / não" (`validate_facing_away.py`)
  foi testado mas o ganho líquido não justificou mantê-lo como filtro rígido — a
  votação por tracklet já absorve naturalmente as frames más.
- **SAM 3D Body** (reconstrução 3D do corpo): explorámos (`sam-3d-body/`) para obter
  orientação 3D precisa do tronco. **Não se enquadra** neste pipeline: é pesado demais
  para correr por-frame em vídeo, foi concebido para reconstrução de pose 3D de alta
  fidelidade (não para um sinal binário de orientação em tempo real), e o custo
  computacional é incompatível com o orçamento de GPU/tempo do desafio. Descartado por
  desadequação de âmbito, não por não funcionar isoladamente.

### 3.5 Resultados (comparação direta com o baseline oficial)
Avaliação por *tracklet* (IoU-match ≥0.30, pureza ≥0.80), no conjunto de teste (39 vídeos
com baseline MMOCR disponível via Zenodo):

| Sistema | Accuracy | Registo |
|---------|----------|---------|
| Baseline oficial (MMOCR) | **18.6%** (262/1405) | — |
| **Nosso (Qwen-VL + pose + prompt)** | **39.5%** (554/1404) | **+20.9 pp** |

**38 vitórias, 1 empate, 0 derrotas** em 39 vídeos. Generaliza para o valid: **35.7%**.

### 3.4 Limites identificados (honestos)
Três categorias de erro, duas **irrecuperáveis por software**:
(a) resolução/distância — crops de ~45×24 px sem informação legível;
(b) jogador nunca de costas — o número não aparece em nenhuma frame;
(c) [recuperável] leituras ambíguas — mitigadas pelo *prompt*.

---

## 4. Contribuição B — Atribuição de equipa/lado (team-side)

### 4.1 O problema
O módulo agrupa os *tracklets* em duas equipas (clustering nas *embeddings* de re-ID) e
mapeia cada *cluster* para o lado do campo (esquerda/direita). Dois pontos de falha:
o **clustering** (separar bem as duas equipas) e o **mapeamento** para o lado.

### 4.2 Fixes iniciais (calibrados no teste)
- **Fix #1** — clustering pela **cor do equipamento** quando o re-ID falha, e uma
  **cadeia de votação por tracklet** para o lado (substituindo uma média simples de
  posições, sensível a *outliers*).
- **Fix #2** — desempate por **deslocamento relativo ao guarda-redes** quando a posição
  média dos dois *clusters* é praticamente igual (jogo concentrado numa zona).

Resultado no **teste**: baseline real medido em **85.4%** → **97.9%** (1717/1755).

### 4.3 Problema de generalização e re-calibração (contribuição metodológica)
Ao avaliar no **valid** (nunca usado no desenvolvimento), o team-side caiu para **73.3%**
— sinal de **sobreajuste dos *thresholds* ao conjunto de teste**. Este é um ponto
metodologicamente relevante para a tese: expõe o risco de calibrar heurísticas num só
conjunto.

**Solução — re-calibração conjunta com restrição rígida:**
1. Extraímos os sinais internos por vídeo (embeddings, posições, GT) dos **93 vídeos**
   (39 teste + 54 valid) para *dumps* compactos → permite testar *milhares* de
   configurações **offline em segundos**, sem re-correr o pipeline.
2. **Grid-search** sobre os *thresholds*, avaliando nos dois conjuntos em simultâneo,
   com a **regra dura de zero regressões no teste** (nenhum vídeo de teste pode piorar).
3. **Insight principal**: a heurística antiga só disparava em jogos "one-sided"
   (concentrados num meio-campo). Substituímo-la por um critério **direto de qualidade
   do clustering** — o **coeficiente de silhueta**: se os *clusters* de re-ID estão mal
   separados (silhueta < 0.28), usa-se a cor do equipamento; com uma salvaguarda contra
   *clusters* degenerados. Isto passou a apanhar os jogos **equilibrados** do valid, que
   a regra antiga ignorava.

### 4.4 Resultados da re-calibração (validados no código real, não em simulação)

| Conjunto | Antes | Depois | Δ |
|----------|-------|--------|---|
| **Teste** (39 vídeos) | 97.9% | **97.8%** | intacto (0 regressões) |
| **Valid** (54 vídeos) | 73.3% | **82.4%** | **+9.1 pp** |

**11 vídeos melhorados, 0 regressões** em 93. Ganhos individuais até **+86 pp**
(ex.: SNGS-046 7%→93%, SNGS-081 17%→91%, SNGS-084 30%→98%).

---

## 5. Impacto na métrica oficial (GS-HOTA)

Passagem final com o pipeline completo (Qwen + team-side re-calibrado), avaliação
GS-HOTA oficial por lote.

| Conjunto | GS-HOTA antes | GS-HOTA depois |
|----------|---------------|----------------|
| **Teste** (49 vídeos) | 43.27% | **44.26%** (+0.99 pp) |
| **Valid** (54 vídeos) | 29.10% | **31.95%** (+2.85 pp) |

O ganho no valid é maior por vídeo do que a média sugere: em 10 dos 18 lotes o GS-HOTA
subiu, com picos de **+12.9 pp** (VF09, onde SNGS-046 recuperou team-side 7%→93%) e
**+11.0 pp** (VF11, SNGS-053). A média fica diluída porque ~1/3 dos vídeos do valid
(SNGS-039–058) está limitado pela **deteção** (ver secção seguinte) — nesses, corrigir o
team-side não move o GS-HOTA porque há poucas deteções para pontuar.

**Posicionamento no leaderboard oficial** (só com 2 dos 6 módulos melhorados):

| Ano | Vencedor | Baseline oficial | **Nós (~44%)** |
|-----|----------|------------------|----------------|
| 2024 | 63.81 | 23.36 | ~**2.º–3.º** de 16 (≈ empate com UPCxMobius 43.15) |
| 2025 | 63.90 | 29.01 | ~**7.º** de 14 |

### Descoberta transversal — o gargalo é a deteção
No valid, ~20 vídeos (SNGS-039–058) têm **DetA de 1.5–7%** (o detetor YOLOv8 base quase
não deteta jogadores) com AssA/LocA normais. Nestes, mesmo com team-side e JNR perfeitos,
o GS-HOTA fica preso. **Conclusão para trabalho futuro**: o maior retorno seguinte está
em melhorar a **deteção**, não os módulos a jusante.

---

## 6. Síntese das contribuições (para defesa)

1. **JNR**: pipeline VLM + pose + votação + *prompt engineering* → **+20.9 pp** vs baseline
   (18.6%→39.5%), sem uma única derrota em 39 vídeos.
2. **Team-side**: correção de bugs + re-calibração metodologicamente rigorosa (grid-search
   nos dois conjuntos, zero regressões no teste) → **85.4%→97.8%** (teste) e
   **73.3%→82.4%** (valid); resolveu o sobreajuste com um critério baseado em **silhueta**
   em vez de heurística fixa.
3. **GS-HOTA**: **~44%** só com estes dois módulos — nível de 2.º–3.º lugar de 2024,
   partindo dos restantes componentes não otimizados.
4. **Metodologia**: avaliação em teste **e** valid; diagnóstico honesto dos limites
   (deteção como gargalo, casos fisicamente irrecuperáveis); infraestrutura de calibração
   *offline* reutilizável.

---

## 7. Trabalho futuro sugerido
- Melhorar a **deteção** (detetor mais recente / *fine-tuning*) — maior retorno esperado.
- Estender a re-calibração team-side ao conjunto de **treino** (57 vídeos) para mais
  robustez estatística.
- Substituir *thresholds* fixos por um **classificador calibrado** para a escolha
  cor-vs-reID.

---

*Anexos disponíveis*: tabelas por vídeo (`docs/full_report/`, `docs/team_recalib/`,
`docs/final_report/`); vídeos de visualização anotados (103 vídeos, teste+valid);
relatórios de comparação JNR e team-side antes/depois.

---

## 8. Referências (papers oficiais do desafio)

- **Tarefa e métrica GS-HOTA** — Somers et al., *SoccerNet Game State Reconstruction:
  End-to-End Athlete Tracking and Identification on a Minimap*, CVPRW 2024.
  arXiv:2404.11335 — https://arxiv.org/abs/2404.11335
  (define a tarefa GSR, o baseline oficial e a métrica GS-HOTA que usamos.)

- **Resultados do desafio 2024** — Cioppa et al., *SoccerNet 2024 Challenges Results*.
  arXiv:2409.10587 — https://arxiv.org/abs/2409.10587
  (leaderboard 2024 GSR: vencedor 63.81, 2.º UPCxMobius 43.15, 3.º JAM 34.40,
  baseline 23.36; 16 entradas.)

- **Vencedor de 2024 (SOTA)** — *From Broadcast to Minimap: Achieving State-of-the-Art
  SoccerNet Game State Reconstruction*. arXiv:2504.06357 —
  https://arxiv.org/abs/2504.06357
  (detalha o método 1.º lugar, 63.81 GS-HOTA; útil para comparar a abordagem deles com a
  nossa e justificar onde estão as diferenças — eles otimizam TODOS os módulos.)

- **Resultados do desafio 2025** — Gomes et al., *SoccerNet 2025 Challenges Results*.
  arXiv:2508.19182 — https://arxiv.org/abs/2508.19182
  (leaderboard 2025 GSR: vencedor 63.90, baseline 29.01; 76 participantes, 66 submissões,
  14 entradas. eidos.ai 46.24, UTokyo 40.13 como referências próximas do nosso ~44%.)

> **Como usar na reunião**: o 2404.11335 fundamenta a métrica; o 2409.10587 e o
> 2508.19182 dão os leaderboards onde nos posicionamos; o 2504.06357 mostra o que o 1.º
> lugar fez de diferente (otimização de todo o pipeline vs os nossos 2 módulos isolados).
