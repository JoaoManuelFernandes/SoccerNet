# Jersey Number Recognition — TrackletsJNRv2

## Visão Geral

O módulo `TrackletsJNRv2` reconhece números de camisola em vídeos de futebol.  
Motor principal: **YOLO26m-pose** (extracção de ROI das costas) + **Qwen2.5-VL-3B** (OCR visual).

**Ficheiros principais:**

| Ficheiro | Papel |
|---|---|
| `jersey/tracklets_jnr_v2_module.py` | `DetectionLevelModule` — integração TrackLab |
| `jersey/tracklets_jnr_v2.py` | Lógica standalone — `qwen_ocr_multi_tracklet`, scoring |
| `configs/modules/jersey_number_detect/tracklets_jnr_v2.yaml` | Configuração activa |

---

## Pipeline por Frame

```
frame RGB
  └─► preprocess()       bbox do jogador (tracker)
        └─► YOLO26m-pose  detectar keypoints no crop
              └─► _yolo_back_crop_from_pred()
                    ├─ rejeitar se face visível (de frente)
                    ├─ rejeitar se ombros/ancas < 0.10 confiança
                    ├─ rejeitar se crop < min_crop_h / min_crop_w
                    └─ rejeitar se sharpness (Laplaciano) < min_sharpness=40
                          └─► pending_crops[tid].append(crop)
                              best_crops[tid]  ← top-5 mais nítidos (heapq)
```

Quando `len(pending_crops[tid]) >= min_crops_before_qwen` (=6):

```
PRIMEIRA janela do tracklet → _run_qwen_solo(tid)   ← 1 tracklet, max 3 crops
Janelas seguintes           → _run_qwen_batch(tids)  ← até 4 tracklets, max 20 crops total
```

> **Porquê solo na primeira janela?**  
> No início do vídeo muitos tracklets disparam simultaneamente. Misturar um
> tracklet com jersey nítido com 3 tracklets de má qualidade faz o Qwen devolver
> `None` para **todos** (batch contamination). A primeira janela solo elimina este
> problema com overhead mínimo (~3 chamadas extra, ~3s por vídeo).

---

## Chamada Qwen — `qwen_ocr_multi_tracklet`

```
{tid_0: [crop0, crop1, ...], tid_1: [...], ...}
  └─► seleccionar crops_per_group = max(1, max_total // n_groups) mais nítidos
        └─► prompt multi-grupo:
              "group0: <img><img><img>
               group1: <img><img>
               For each group, reply ONLY: groupN: X or groupN: none"
                └─► parse "groupN: 28" → {'tid_0': ('28', 0.9), ...}
```

**Limite seguro:** 4 tracklets × 5 crops = 20 imagens/chamada.  
Acima de ~25 imagens o Qwen 3B falha silenciosamente (atenção saturada).

---

## Votação por Tracklet — `finalize_ocr`

Para cada tracklet, agrega todas as janelas em `qwen_results[tid]`:

```
historial = [None, '28', '28', None, '28', '28', '28', None, ...]
            ↓
track_min_agreement = 0.45  → o número vencedor deve aparecer em ≥45% de TODAS as janelas
                               (None conta contra — evita aceitar 1 detecção + 10 None)
min_valid_windows = 1       → mínimo de janelas não-None antes de aceitar
                              ↓
                         jersey = '28', conf = 0.647
```

Se `track_min_agreement` falhar → **segmentação temporal**: divide o historial em
segmentos contíguos e aceita o segmento com maior razão de concordância (útil quando
o jogador muda de plano ao longo do vídeo).

---

## Rescue Pass

Após o voto global, para tracklets ainda `None`:

```python
best_crops[tid]   # top-5 mais nítidos acumulados durante todo o vídeo
  └─► qwen_ocr_multi_tracklet({tid: top3_crops}, max_total_crops=3)
        # chamada solo com os 3 melhores crops do vídeo inteiro
```

Resolve casos em que todas as janelas foram contaminadas ou tinham poucos crops,
mas o tracklet tem pelo menos 1 crop de qualidade no `best_crops`.

---

## Configuração

```yaml
# YOLO pose
pose_weights:       /home/joao/soccernet/weights/yolo26m-pose.pt
pose_conf:          0.25

# ROI das costas
pad_x:              0.55   # expansão lateral (captura largura da camisola)
pad_y_top:          0.10   # margem superior (captura número no topo)
roi_height_frac:    0.90   # fracção do tronco (ombros → ancas)

# Qwen
qwen_model_path:    Qwen/Qwen2.5-VL-3B-Instruct  # 4-bit, ~2GB VRAM

# Janelas
min_crops_before_qwen:      6   # crops por tracklet antes de disparar Qwen
max_tracklets_per_qwen_batch: 4
max_total_crops_per_batch:  20  # limite seguro (>25 → Qwen falha silenciosamente)

# Filtro qualidade
min_crop_h:      30    # px
min_crop_w:      25    # px
min_sharpness:   40.0  # variância Laplaciano (relva≈5-20, jersey≈50-4000)

# Votação
window_min_agreement:  0.75   # acordo mínimo dentro de uma janela
track_min_agreement:   0.45   # fracção do historial total (None conta contra)
min_valid_windows:     1
finalize_min_crops:    3      # mínimo de crops para chamar Qwen no finalize

# Debug
save_debug_crops:    true   # grava frame_{fid}_qwen{nr}.jpg por tracklet
save_distill_crops:  false
```

---

## Resultados (3 vídeos — validação SoccerNetGS)

> Métrica: **value-level** (conjunto de jerseys detectados vs GT).  
> Upper bound — não penaliza atribuição de número ao tracklet errado.

| Vídeo | Frames | Tracklets | Prec | Rec | F1 | Qwen calls | Tempo |
|---|---|---|---|---|---|---|---|
| SNGS-021 | ~2900 | 31 | 88% | 78% | 82% | 264 | 399s |
| SNGS-022 | ~2340 | 31 | 86% | 80% | 83% | 275 | 388s |
| SNGS-023 | ~2100 | 26 | 93% | 81% | 87% | 212 | 380s |
| **GLOBAL** | | | **89%** | **80%** | **84%** | | **~6.5 min/vídeo** |

**FN sistemáticos (jerseys nunca detectados):**  
Jersey 10 está em FN nos 3 vídeos — o jogador aparece consistentemente de frente,
todos os crops são rejeitados pelo YOLO antes de chegarem ao Qwen.

**FP típicos:**  
Jerseys que existem noutras equipas/árbitros mas não no GT (e.g. '5', '1', '6', '16').
Indicam tracklets de árbitros ou membros de equipa técnica cujas camisolas têm números.

---

## Decisões de Design

### Porquê Qwen e não OCR clássico (PaddleOCR)?

PaddleOCR tem incompatibilidade ABI com o ambiente (`PaddlePaddle` + CUDA PyTorch → SIGSEGV).
Mesmo em subprocess isolado, não recuperou nenhum jersey adicional nos testes:
os crops que o Qwen falha são genuinamente ilegíveis (jogadores de frente, tronco parcial),
não problemas de OCR textual.

### Porquê 3B e não 7B?

A GPU disponível é uma RTX 3060 Laptop (6GB VRAM). Qwen 7B requer ~10GB.
O 3B em 4-bit ocupa ~2GB e deixa margem para YOLO (~1.5GB) em runtime simultâneo.

### Porquê não recarregar o Qwen entre vídeos?

`bitsandbytes` 4-bit não suporta reload no mesmo processo. O Qwen é carregado uma
vez e mantido em memória durante os 3 vídeos.

### `torch.cuda.synchronize()` antes de cada batch Qwen

YOLO executa operações assíncronas na GPU. Sem sincronização, o Qwen pode começar
a inferência com o contexto CUDA em estado intermédio → todos os resultados None.
O `print(..., flush=True)` no fim de `qwen_ocr_multi_tracklet` também é necessário
(force yield do GIL Python que permite ops pendentes da GPU completar).

---

## Limitações Conhecidas

| Problema | Causa | Impacto |
|---|---|---|
| Jersey 10 nunca detectado | Jogador sistematicamente de frente | FN constante nos 3 vídeos |
| Instabilidade janelas tardias | Qwen 3B com poucos crops (final de tracklet) | Historial ruidoso (`'25'`, `'5'` aleatórios) |
| Métricas value-level infladas | Não penaliza tracklet errado | F1 real < 84% |
| Sem avaliação track-level | Sem matching IoU GT↔tracklet | Impossível saber precision real |

---

## Próximos Passos Possíveis

1. **Avaliação track-level** — matching IoU bbox GT↔tracklet para precision/recall real
2. **Melhoria ROI** — aumentar `roi_height_frac` ou usar segmentação de corpo para capturar número mesmo quando keypoints são parciais
3. **Qwen 7B quantizado** — se houver GPU ≥10GB disponível, esperado +10-15pp F1
4. **Dataset de distilação** — usar os crops aceites para treinar um modelo leve dedicado (evitar Qwen em produção)
