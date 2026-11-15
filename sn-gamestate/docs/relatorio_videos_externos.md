# Relatório — Validação em Vídeos Externos (fora do SoccerNet)

**Data:** 2026-09-03 / 2026-09-05
**Objetivo:** avaliar se o sistema desenvolvido generaliza para vídeos reais fora do conjunto
SoccerNet-GSR, e caracterizar em que condições cada módulo funciona ou falha.

---

## 1. Descoberta enquadradora — a origem dos dados do SoccerNet-GSR

Confirmado nos artigos oficiais (arXiv:2404.11335, arXiv:2204.06918):

> Os clips do SoccerNet-GSR provêm de gravações a **1080p de 12 jogos da Super Liga Suíça
> de 2019**, captados pela **câmara principal** de transmissão, a 25 fps.

Isto explica o domínio dos modelos pré-treinados (deteção, ReID, calibração, classificação de
função) e é a chave para interpretar todos os resultados abaixo: quanto mais a filmagem se
afasta dessa produção televisiva específica, mais módulos falham.

Confirmação visual independente: a publicidade nos vídeos do próprio dataset é suíça
(SRG SSR, Migros, Valiant, swisspor).

---

## 2. Metodologia desenvolvida

Para evitar correr o *pipeline* completo (~40 min) em clips que à partida não serviam,
criaram-se duas ferramentas de triagem:

### 2.1 Medição do tamanho dos jogadores em píxeis
Corre-se apenas o detetor (YOLOv8x6) sobre 3 *frames* do clip e mede-se a **altura das caixas
delimitadoras** dos jogadores. Reporta-se a **mediana** e o **percentil 90** (os jogadores mais
próximos da câmara). Custo: ~1 min por clip.

### 2.2 Detetor de cortes
As transmissões completas alternam entre câmara principal, *replays* e grandes planos, o que
quebra o *tracking* e a calibração. O varrimento analisa o vídeo de 10 em 10 *frames* e
classifica cada um como "câmara principal" (≥8 jogadores detetados e mediana de altura entre
60 e 220 px), identificando o **maior segmento contínuo** utilizável.

---

## 3. Resultados por vídeo

| # | Vídeo | Fonte / janela | Mediana | p90 | Calib. | Equipas | Números |
|---|-------|----------------|--------:|----:|:------:|:-------:|:-------:|
| 1 | Amador (telemóvel) | 1040×576 | — | — | ❌ | ❌ | ❌ |
| 2 | Man City × Chelsea | `JA0p0Bg9N1w` 25:00–25:40, 720p | — | — | ✅ | ❌ | ❌ |
| 3 | Tottenham × Watford | `aHByCYiqpIA` 18:00–18:40, 1080p | — | — | ✅ | ❌ | ❌ |
| 4 | Milan × Como | `BtgHTyfW6Wg` 24:45–25:10 | 55 | 72 | ✅ | ✅ | ❌ |
| 5 | Chéquia × Inglaterra | `HOwLN4PhL10` 29:55–30:20 | 63 | 72 | ✅ | ✅ | ❌ |
| 6 | Chéquia × Inglaterra | `HOwLN4PhL10` 11:52–12:17 | 78 | 89 | ✅ | ✅ | ❌ |
| 7 | Arsenal × Chelsea | `iQ7Em5lzEKo` 0:00–0:20 | 74 | 102 | ✅ | ✅ | ✅ (3) |
| 8 | **Arsenal × Chelsea** | `iQ7Em5lzEKo` **57:55–58:20** | 66 | **109** | ✅ | ✅ | **✅ (6)** |
| 9 | YB × Zürich (suíço 2019) | `duyVmRTgSOk` seg. contínuo de 26:00 | ~110 | **158** | ✅ | ❌* | **✅ (10)** |
| — | PSG × Real Madrid | `PL4sebvIBvs` 48:00 (só medido) | 52 | 58 | — | — | — |

\* falha atribuível ao classificador de função — ver secção 5.

**Melhor figura global: #8 (Arsenal × Chelsea @58:00)** — calibração, equipas e números todos
a funcionar. Números lidos: 5, 5, 3, 5, 12, 5 (confianças 0.8–1.0), 6 em 10 *tracklets*.

**Recorde de leitura de números: #9 (YB × Zürich)** — 10 em 15 *tracklets* (67%):
16, 3, 9, 6, 25, 6, 11, 5, 23, 5 (confianças 0.5–1.0).

---

## 4. Resultado principal — limiar de píxeis para a leitura do número

A leitura do número depende do **tamanho do jogador na imagem**, e a transição é **abrupta**,
não gradual:

| p90 da altura do jogador | Números lidos |
|---|---|
| 58 – 89 px | **0** (em 4 clips distintos) |
| 102 px | 3 |
| 109 px | 6 |
| 158 px | 10 |

**Explicação mecânica:** o número ocupa cerca de 15 % da altura do jogador. Assim, 89 px de
jogador ⇒ número com ~13 px (ilegível, o modelo recusa); 102 px ⇒ ~15 px (limiar da leitura);
158 px ⇒ ~24 px (leitura fiável). O critério prático de triagem estabelecido foi
**p90 ≳ 100 px**.

**Observação relevante:** o tamanho varia muito **dentro do mesmo jogo**, consoante o *zoom*.
No mesmo Arsenal × Chelsea mediram-se p90 de 109 px (@58:00), 102 px (@0–20 s) e 78 px
(@35:00), com 6, 3 e 0 números lidos respetivamente. **Não é o jogo que determina o resultado,
é o momento.**

Isto quantifica, com dados independentes do SoccerNet, o "teto" identificado no Capítulo 3.

---

## 5. Ablação — a falha das equipas no vídeo suíço é do classificador de função

No vídeo #9 (YB × Zürich) os números funcionaram excecionalmente bem mas a atribuição de equipa
colapsou. A causa foi isolada experimentalmente.

**Diagnóstico.** O equipamento do Young Boys é **amarelo**; no SoccerNet os **árbitros** vestem
amarelo. O classificador de função (não intervencionado neste trabalho) classifica assim:

| Função atribuída | *Tracklets* |
|---|---|
| `goalkeeper` | **40** (deviam ser 2) |
| `referee` | 13 |
| `player` | 22 |
| `ball` | 2 |

Como o módulo de agrupamento de equipa filtra `role == "player"`, apenas 22 dos 77 *tracklets*
lhe chegam.

**Experiência controlada.** Reprocessou-se o estado guardado, comparando o agrupamento com as
funções originais e com as funções corrigidas (todas forçadas a `player`). A validação da
separação usou a **saturação média da camisola** (amarelo é saturado, branco não):

| Condição | *Tracklets* | Divisão | Saturação por grupo | Separação correta? |
|---|---|---|---|---|
| A) funções originais | 22 | **20 / 2** (degenerada) | S = 0.30 / 0.39 | ❌ (só FCZ) |
| B) funções corrigidas | 75 | **51 / 24** | **S = 0.61 / 0.32** | ✅ (YB / FCZ) |

**Conclusão.** Com as funções corretas, o módulo de atribuição de equipa **separa corretamente
as duas equipas**. A falha é integralmente atribuível ao **classificador de função**, um módulo
não intervencionado neste trabalho.

---

## 6. Síntese — três modos de falha, todos a montante

O sistema, aplicado fora do domínio de treino, falha de três formas distintas, cada uma com
causa identificada e **todas em módulos não intervencionados**:

| Falha | Causa | Evidência |
|---|---|---|
| **Calibração** | ângulo/resolução fora do domínio | vídeo amador: nenhuma linha detetada |
| **Números** | jogador < ~100 px (p90) ⇒ número < ~15 px | 0 leituras em 4 clips; 3/6/10 leituras a 102/109/158 px |
| **Equipas** | colisão da cor do equipamento com a convenção de cor do árbitro | YB amarelo ⇒ 40/77 *tracklets* como guarda-redes |

Os módulos desenvolvidos nesta dissertação (leitura do número e atribuição de equipa)
comportam-se corretamente sempre que recebem entradas válidas — o que reforça a conclusão do
Capítulo 3: a margem de progressão que resta situa-se nas etapas anteriores do *pipeline*.

---

## 7. Correções de código necessárias

Três defeitos impediam o funcionamento com vídeos externos. Todos foram corrigidos e são
seguros para os vídeos SoccerNet 1080p:

1. `sn_gamestate/calibration/nbjw_calib.py:179` — `metadatas["keypoints"][0]` acedia por
   **rótulo**, falhando em todos os *frames* exceto o primeiro. Corrigido para `.iloc[0]`
   (acesso por posição), consistente com o resto do método.
2. `sn_gamestate/visualization/pitch.py`, `draw_radar_view` — dimensões do painel do *minimap*
   fixas em 1920×1080, provocando erro de OpenCV em qualquer outra resolução. Passou a usar
   `patch.shape[:2]` com limitação da região aos limites da imagem. Adicionalmente, as
   coordenadas dos jogadores passaram a ser limitadas às dimensões do campo, para que pontos
   resultantes de calibração imprecisa fiquem contidos no painel.
3. `sn_gamestate/configs/soccernet.yaml` — lista `defaults` corrompida
   (`- _self_  - _self_  -   - _self_` seguida de `_self_` isolado), YAML inválido que impedia
   qualquer execução. Reduzida a um único `- _self_`.

---

## 8. Infraestrutura criada

- **ffmpeg estático** em `/home/joao/soccernet/ffbin/` (binários reais, permanentes). O ffmpeg
  do ambiente conda estava inutilizável (`libavcodec.so.56` incompatível com a libx264 moderna).
- **`run_video.sh`** — corre o *pipeline* num `.mp4` local ou URL do YouTube
  (`VIDEO=<caminho|url> bash run_video.sh <nframes>`). Nota: `nframes` nunca `-1`, pois isso
  reativa o GS-HOTA, que falha sem anotação de referência.
- **Fluxo de aquisição:** `yt-dlp --download-sections` para extrair apenas o trecho pretendido,
  seguido de transcodificação para H.264 (o OpenCV não descodifica AV1, formato que o YouTube
  entrega frequentemente a 720p).

---

## 9. Artefactos (localização dos resultados)

| Vídeo | Execução | Clip de origem |
|---|---|---|
| Arsenal × Chelsea @58:00 | `outputs/sn-gamestate/2026-09-05/10-25-57/` | `/home/joao/soccernet/ars58.mp4` |
| Arsenal × Chelsea 0–20 s | `outputs/sn-gamestate/2026-09-04/09-11-05/` | `/home/joao/soccernet/broadcast_clip2.mp4` |
| YB × Zürich (suíço) | `outputs/sn-gamestate/2026-09-04/17-22-18/` | `/home/joao/soccernet/yb_final.mp4` |
| Milan × Como | `outputs/sn-gamestate/2026-09-04/13-31-28/` | `/home/joao/soccernet/milan_clip.mp4` |
| Chéquia × Inglaterra @30:00 | `outputs/sn-gamestate/2026-09-04/14-43-11/` | `/home/joao/soccernet/euro_clip.mp4` |
| Chéquia × Inglaterra @12:00 | `outputs/sn-gamestate/2026-09-04/16-26-23/` | `/home/joao/soccernet/eng_clip.mp4` |

Em cada execução, o vídeo com o *minimap* está em `visualization/videos/0.mp4` e as leituras de
número em `jnr_v2_debug/*/jnr_results.json`.

---

## 10. Notas para a redação

- **Figura de generalização:** usar o Arsenal × Chelsea @58:00 (tudo a funcionar).
- **Caso de estudo do limite:** usar o YB × Zürich com a ablação da secção 5 — mostra
  capacidade de diagnóstico, não apenas de execução.
- **Contributo quantitativo:** o limiar de píxeis (secção 4) é uma medição original, feita em
  fonte independente, que sustenta numericamente a conclusão do Capítulo 3.
- **Fontes dos vídeos:** canal *Chepsi007* (usar apenas os títulos "VIP Tactical Camera"; as
  variantes "Exclusive VIP Camera" e "Close-Up" são inutilizáveis) e transmissões RE-LIVE da
  Super Liga Suíça (necessitam do detetor de cortes da secção 2.2).
- **Direitos:** os clips foram usados apenas para teste académico privado e não são
  redistribuídos.
