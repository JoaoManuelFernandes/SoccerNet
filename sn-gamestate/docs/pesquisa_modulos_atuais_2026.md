# Módulos mais atuais para cada etapa do pipeline (pesquisa de 2026-09-19)

Objetivo: preparar a evolução do pipeline para o produto de análise de vídeo amador (plano "1")
e, se der tempo, para a tese. Fontes principais: SoccerNet 2025 Challenges Results
(arXiv 2508.19182, leaderboard GSR e relatórios dos participantes), Broadcast2Pitch (WACV 2026,
vencedor GSR 2025), "From Broadcast to Minimap" (Constructor.Tech, arXiv 2504.06357, 2.º lugar),
SoccerNet 2026 Challenges Results (arXiv 2607.07320: em 2026 não houve desafio GSR; o GSR só
aparece dentro do VQA). O que está marcado como "opinião" é avaliação minha, não medição.

## Onde estamos face ao estado da arte

| | Baseline SoccerNet (2024) | Nós (2026-09) | GSR 2025, 1.º (KIST) | GSR 2025, 2.º (Constructor.Tech) | GSR 2025, 4.º (Playbox/MIXI) |
|---|---|---|---|---|---|
| GS-HOTA | 29.0 | ~42.6-44.3 (nossos lotes) | **63.90** | 63.81 | 61.64 |
| deteção | YOLOv8 | YOLOv8 | YOLO-X afinado | YOLOv5m afinado | RF-DETR |
| ReID | PRTReID | PRTReID + cor 27 d | OSNet | ReID + TeamID embeddings próprios | CLIP-ReID |
| tracking | StrongSORT | StrongSORT | Deep-EIoU + split/merge por identidade | Kalman + pós-processamento por número/equipa/ReID | BoT-SORT + GTA-Link |
| calibração | NBJW | NBJW | keypoints multi-frame | SegFormer + refinamento por keypoints | BroadTrack + fluxo ótico |
| número | MMOCR | Qwen2.5-VL-3B (JNR v2) | LLaMA-3.2-Vision (função+número+cor num só prompt) | classificador próprio (dígitos separados) | ViTPose + PARSeq |
| equipa/lado | k-means ReID + média x | k-means auditado + cascata | agrupamento por cor + média x por grupo | 5 clusters (2 equipas, árbitros, 2 GR) por posição + votação por frame | (não descrito) |

Leitura: o salto de 29 → 64 nos vencedores não vem de um módulo, vem de (1) detetores afinados no
SoccerNet, (2) tracking com re-associação de tracklets usando número e equipa, (3) VLM ou OCR
moderno para o número, (4) pós-processamento de consistência (voto maioritário, interpolação,
suavização). Os nossos módulos de equipa e número estão na linha dos vencedores; o que nos falta
é sobretudo o tracking/re-associação e o detetor afinado.

## Por módulo: o que existe de mais atual e o que eu faria

### 1. Deteção de jogadores
- Vencedores 2025: YOLO-X, YOLOv8, YOLOv11 e RF-DETR, **sempre afinados no SoccerNet-GSR**.
- 2026: **RF-DETR** (Roboflow, ICLR 2026, Apache 2.0; primeiro em tempo real acima de 60 mAP COCO;
  lidera RF100-VL, o benchmark de transferência entre domínios) e **YOLO26** (jan. 2026, sem NMS,
  STAL para objetos pequenos, foco em CPU/edge). Já temos `yolo26m-pose.pt` no repositório.
- Para vídeo de bancada (jogadores pequenos, câmara fixa): afinar um detetor com 200-400 imagens
  anotadas do nosso próprio setup vale mais do que qualquer modelo novo sem afinação. Opinião:
  RF-DETR-S/M afinado para exatidão; YOLO26 se for preciso correr em CPU.

### 2. Reidentificação
- Vencedores: OSNet (1.º), CLIP-ReID (4.º), embeddings próprios de ReID + TeamID (2.º).
- Alternativas atuais: **CLIP-ReID** (texto+imagem, generaliza bem), **KPR / keypoint-promptable
  ReID** (Somers 2024, sucessor do PRTReID, lida com oclusão), SOLIDER.
- O que já fizemos (descritor de cor do torso) é o equivalente barato do "TeamID embedding" do 2.º
  classificado. Opinião: manter PRTReID/KPR para tracking; para a equipa, o descritor de cor chega
  em vídeo de bancada, onde a cor é estável.

### 3. Tracking
- É onde os vencedores mais ganharam: **Deep-EIoU** (1.º), **BoT-SORT + GTA-Link** (4.º), fluxo
  ótico (Farneback) a alimentar o Kalman (3.º), e em todos um passo de **split/merge de tracklets**
  guiado por número de camisola, equipa e ReID, mais interpolação e suavização (Savitzky-Golay).
- Opinião: a maior melhoria possível para nós com menos risco é o pós-processamento de tracklets
  (juntar fragmentos com o mesmo número+equipa e ReID próximo; separar tracklets com número
  inconsistente). Na tese, isto explicaria diretamente a fragmentação que vimos no SNGS-113
  (65 tracklets para ~20 jogadores). Em vídeo de bancada (câmara fixa), o tracking é bem mais
  fácil e o BoT-SORT chega.

### 4. Campo e calibração
- Vencedores: NBJW, **PnLCalib** (já está no repositório: `calibration/pnlcalib.py`), BroadTrack,
  com a novidade de 2025 de **propagar a calibração no tempo com fluxo ótico** a partir de frames
  estáveis (4.º e 13.º).
- Para câmara fixa de bancada: calibra-se **uma vez** (4 cliques nas linhas do campo ou keypoints
  numa imagem) e fica para o jogo todo; nada disto é necessário. É a maior simplificação do produto.

### 5. Número da camisola
- Vencedores: VLMs (LLaMA-3.2-Vision no 1.º, Qwen2-VL no 7.º, com "zoom" por chamada de função) e
  OCR de cena (PARSeq sobre torso localizado por ViTPose, no 4.º). Trabalho académico 2025:
  "Single-Stage Uncertainty-Aware JNR" (CVPRW 2025), cabeça compositiva por dígito + incerteza de
  Dirichlet + agregação por tracklet, ViT-S com 82.7% no teste SoccerNet-JN só com dados públicos.
- 2026: **Qwen3-VL** (família 2B a 235B; a 4B/8B em 4 bits cabe em 6 GB) substitui o Qwen2.5-VL
  que usamos; em OCR de documentos é claramente melhor, em dígitos de camisola não há medição
  publicada (a fazer por nós: é trocar o `qwen_model_path` e correr os 49 de teste, ~1 dia de GPU).
- Opinião: com um PC melhor (RTX 4090 / 24 GB), Qwen3-VL-8B em 16 bits ou PARSeq afinado em
  números do SoccerNet-JN. Para o produto amador, o número pode ser substituído pela confirmação
  manual do treinador no início do jogo.

### 6. Equipa e lado do campo
- Vencedores: cor da camisola como sinal principal (1.º: clustering de cores; 2.º: cinco clusters
  em vez de dois, separando árbitros e guarda-redes por posição no campo) e lado por **votação
  frame a frame da média de x** (2.º), a mesma suposição de base que nós usamos.
- O que nos distingue: a auditoria (silhueta) e a cascata com guarda-redes. O que podemos importar
  do 2.º classificado: os cinco clusters (evita que árbitros e guarda-redes contaminem os dois
  grupos, que é exatamente o modo de falha "função" dos vídeos 039-058) e a votação por frame em
  vez de por tracklet (reduz o peso da fragmentação, o problema do 113).
- Em vídeo de bancada: as duas equipas estão sempre inteiras no enquadramento; o lado resolve-se
  pela posição dos guarda-redes no início de cada parte, sem cascata.

### 7. Pós-processamento (o que nenhum dos nossos módulos faz e todos os vencedores fazem)
- Voto maioritário por tracklet para função, equipa e número; interpolação de falhas curtas;
  suavização das trajetórias no campo; regra "no máximo 22 jogadores + árbitros" (GSR-9).

## Hardware
- Os nossos tempos (RTX 3060 portátil, 6 GB, 7.6 GB RAM): ~21 min por clip de 30 s. Os gargalos são
  ReID (7 min), número (7 min) e campo (3.5 min).
- Com 24 GB de VRAM: Qwen3-VL-8B sem quantização, RF-DETR-L, lotes maiores no ReID; estimativa
  (opinião) de 3-5 min por clip. Para o produto, um jogo de 90 min de bancada a 25 fps são 135 000
  imagens; é obrigatório processar a 5 fps e em resolução reduzida, ou o custo por jogo não fecha.

## Fontes
- SoccerNet 2025 Challenges Results: https://arxiv.org/abs/2508.19182
- Broadcast2Pitch (WACV 2026): https://openaccess.thecvf.com/content/WACV2026/papers/Oo_Broadcast2Pitch_Game_State_Reconstruction_from_Unconstrained_Soccer_Videos_WACV_2026_paper.pdf
- From Broadcast to Minimap (Constructor.Tech): https://arxiv.org/abs/2504.06357
- SoccerNet 2026 Challenges Results: https://arxiv.org/abs/2607.07320
- Single-Stage Uncertainty-Aware JNR (CVPRW 2025): https://openaccess.thecvf.com/content/CVPR2025W/CVSPORTS/papers/Grad_Single-Stage_Uncertainty-Aware_Jersey_Number_Recognition_in_Soccer_CVPRW_2025_paper.pdf
- RF-DETR: https://github.com/roboflow/rf-detr ; YOLO26: https://blog.roboflow.com/yolo26/
- Qwen3-VL: https://github.com/qwenlm/qwen3-vl ; relatório técnico https://arxiv.org/abs/2511.21631

## Anexo (2026-09-19): reestruturação para o produto de vídeo de bancada, abordagens

- SAM 3 / 3.1: máscaras por prompt de texto e seguimento em vídeo, mas sem semântica das linhas do
  campo; ~840 M parâmetros, 3.4 GB de pesos, ~0.3-0.5 s/imagem num 3060 (estimativa). Usar só offline
  e uma vez: máscara do relvado; máscaras de torso em tracklets de cor ambígua. Calibração em câmara
  fixa: uma vez por jogo (cliques ou PnLCalib numa imagem).
- Duas passagens: (1) barata e contínua: deteção + seguimento por movimento/IoU + cor; (2) dirigida:
  ReID e número só onde a passagem 1 deixou dúvidas (split/merge de tracklets como os vencedores 2025).
- Número dirigido: só em tracklets sem número, 3-5 leituras em imagens de costas e grandes (pose),
  parar ao consenso; releitura periódica só se a aparência do tracklet se afastar da média.
- Conjunto fechado: ficha de jogo do treinador (11-18 números por equipa) → leitura restrita ao
  conjunto + atribuição global tracklet↔jogador (húngara, "cada número num tracklet de cada vez").
- ReID a pedido: 5-10 embeddings por tracklet, só para re-associar tracklets que acabam/começam
  perto no tempo e no campo.
- Equipa: 5 grupos de cor (2 equipas, árbitros, 2 GR) estimados nos primeiros 2 min com ajuda da
  posição; depois classificação por centróide; lado por parte pelo guarda-redes.
- 5 fps + 720p na deteção, recortes a resolução total só para o número; interpolar o resto.
- Pós-processamento: voto por tracklet, interpolação, suavização, máximo 11 por equipa em campo.
- Orçamento estimado (opinião) para 90 min no RTX 3060: ~40 min por jogo; com 24 GB de VRAM, ~15 min.
- Ordem de implementação sugerida: ficha de jogo + número dirigido → equipa 5 grupos → duas passagens.
