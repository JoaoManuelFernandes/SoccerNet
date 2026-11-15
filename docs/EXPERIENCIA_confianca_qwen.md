# Reconhecimento do número: onde está o erro e como o reduzir

**Data:** 2026-09-23 · **Estado:** hipótese validada numa amostra aleatória de 10 vídeos; falta confirmar
no conjunto de treino e implementar no módulo.

Todas as percentagens são **ponderadas por deteções**, que é o que o GS-HOTA conta. As categorias são:
*certo*; *errado* (a anotação tem número e damos outro); *vazio* (a anotação tem número e não damos
nenhum); *abstenção certa* (nem a anotação nem nós damos número); *inventado* (a anotação não tem
número e damos um). *Passa* = certo + abstenção certa.

---

## 1. Onde está o custo do número (159 vídeos, sistema da dissertação)

1 303 412 deteções de jogador emparelhadas com a anotação.

| Categoria | % das deteções |
|---|---|
| Certo | 51.4% |
| Anotação com número, nós sem número (vazio) | 22.9% |
| Número errado | 12.3% |
| Abstenção certa | 10.9% |
| Número inventado | 2.5% |

Por comprimento do *tracklet* (% do total de deteções):

| | errado | vazio | certo |
|---|---|---|---|
| Curtos (< 50 frames) | 0.2% | 2.3% | 0.2% |
| Médios (50–200) | 2.8% | 9.2% | 7.0% |
| **Longos (≥ 200)** | **9.2%** | **11.4%** | 44.2% |

**20.6% das deteções estão em *tracklets* longos que acabam errados ou vazios.** É aí que uma
melhoria aparece no GS-HOTA. Uma correção que atue sobre *tracklets* curtos, por muito que melhore a
exatidão por *tracklet*, é quase invisível na métrica oficial (ver `EXPERIENCIA_fusao_fragmentos.md`).

## 2. Porque é que cada deteção falha (deteções com número anotado)

Reconstruído a partir dos votos por janela guardados nos registos de execução.

| Causa | % |
|---|---|
| Certo | 59.4% |
| Sem número: o *tracklet* nunca chegou ao Qwen (filtro de tamanho/nitidez) | 23.8% |
| Errado: o número certo nunca apareceu nos votos | 9.7% |
| Errado: o número certo estava nos votos e perdeu | 4.5% |
| Sem número: o Qwen só respondeu *none* | 2.0% |
| Sem número: a votação rejeitou | 0.7% |

## 3. O que se testou e não resultou

| Ideia | Resultado |
|---|---|
| Fusão de fragmentos do seguimento | +5.5 pp por *tracklet*, **+0.04 GS-HOTA** — rejeitada |
| Limiares de votação mais exigentes | cortam mais certos do que errados |
| Unicidade do número dentro da equipa | +0.04 a +0.12 pp — irrelevante |
| Filtro de qualidade relaxado, sem mais | ganha nos vídeos de câmara afastada, mas **inventa números** (SNGS-095: abstenções 23% → 11%) |
| Filtro relaxado em vídeos típicos | não acrescenta nada: os *tracklets* novos são curtos |

## 4. A causa: o módulo não tem confiança

A função que chama o Qwen devolve sempre `(número, 0.90)`, qualquer que seja a certeza do modelo. Um
14 nítido e um número adivinhado num borrão pesam o mesmo na votação. A probabilidade que o modelo
atribui a cada dígito existe, mas é descartada.

## 5. A hipótese: usar a confiança real do Qwen

Cada janela (6 crops consecutivos do mesmo *tracklet*) é lida sozinha, com os 5 crops mais nítidos,
pelo mesmo modelo, com o mesmo prompt e a mesma descodificação. A confiança é o produto das
probabilidades dos tokens da resposta. **Uma janela só vota se a confiança for ≥ 0.5; o *tracklet*
fica com o número mais votado entre as janelas confiantes.**

### Amostra aleatória: 10 vídeos de teste/validação (semente 20260923)

SNGS-022, 029, 039, 058 (validação) e 119, 120, 142, 147, 191, 199 (teste).

| Regra | certo | errado | vazio | abst. certa | inventado | passa |
|---|---|---|---|---|---|---|
| Dissertação | 45.5 | 17.5 | 20.0 | 13.6 | 3.3 | 59.1 |
| **Confiança ≥ 0.5** | **49.6** | **2.6** | 30.9 | **16.8** | **0.1** | **66.4** |
| Confiança ≥ 0.7 | 44.4 | 0.8 | 37.9 | 16.9 | 0.0 | 61.3 |

**Mais certos (+4.1), errados −85%, números inventados praticamente eliminados, mais abstenções
certas.**

### GS-HOTA oficial nos mesmos 10 vídeos

| Regra | Teste | Δ | Validação | Δ |
|---|---|---|---|---|
| Dissertação | 45.99 | — | 31.66 | — |
| **Confiança ≥ 0.5** | **48.92** | **+2.94** | **33.96** | **+2.30** |
| Confiança ≥ 0.7 | 47.21 | +1.23 | 33.29 | +1.63 |

Por vídeo, com confiança ≥ 0.5:

| Vídeo | Dissertação | Confiança ≥ 0.5 | Δ |
|---|---|---|---|
| SNGS-119 | 77.26 | 86.17 | +8.91 |
| SNGS-029 | 35.96 | 44.27 | +8.31 |
| SNGS-120 | 42.37 | 47.83 | +5.46 |
| SNGS-022 | 61.54 | 64.74 | +3.20 |
| SNGS-199 | 31.56 | 33.46 | +1.90 |
| SNGS-191 | 46.35 | 47.63 | +1.29 |
| SNGS-147 | 47.87 | 48.95 | +1.08 |
| SNGS-039 | 4.18 | 4.24 | +0.06 |
| SNGS-058 | 20.77 | 20.77 | +0.00 |
| SNGS-142 | 37.10 | 36.98 | −0.12 |

Nove em dez vídeos melhoram ou ficam iguais; o único abaixo fica a −0.12. Com o limiar 0.7 há três
vídeos a piorar (SNGS-191 −3.36), pelo que **0.5 é o ponto escolhido**.

### Os 5 vídeos difíceis (câmara afastada), filtro totalmente relaxado (20/15/15/1)

Os mesmos crops do primeiro ensaio, relidos com confiança. Sem confiança, o filtro relaxado inventava
números (SNGS-095 perdia 3.4 pontos de GS-HOTA).

| Regra | certo | errado | vazio | abst. certa | inventado | passa |
|---|---|---|---|---|---|---|
| Dissertação | 30.9 | 13.5 | 44.1 | 11.5 | 0.0 | 42.4 |
| Relaxado, sem confiança | 50.3 | 27.9 | 10.2 | 8.4 | 3.1 | 58.8 |
| **Relaxado + confiança ≥ 0.5** | **53.4** | **8.2** | 26.8 | 10.8 | 0.7 | **64.3** |

O único número "inventado" que resta (0.7%) é o 14 do SNGS-187, que a secção 7 mostra estar visível
e em falta na anotação.

| Vídeo | Dissertação | Relaxado + confiança ≥ 0.5 | Δ |
|---|---|---|---|
| SNGS-196 | 17.07 | 38.66 | +21.58 |
| SNGS-187 | 33.12 | 46.28 | +13.16 |
| SNGS-188 | 42.65 | 53.63 | +10.98 |
| SNGS-095 | 53.34 | 59.93 | +6.59 |
| SNGS-027 | 37.01 | 37.76 | +0.76 |
| **Teste (3) / Validação (2)** | 34.07 / 44.66 | **47.69 / 47.94** | **+13.62 / +3.28** |

**Nos 15 vídeos ensaiados, 14 melhoram ou ficam iguais; o único abaixo fica a −0.12.**

### Dia 24/09: 35 vídeos novos (15 piores + 20 sorteados), treino incluído

Relatórios completos: `jnr_exp4/RESULTADO.md` (15 piores, 5 por conjunto) e `jnr_exp5/RESULTADO.md`
(20 sorteados, semente 20260924). O treino entra pela primeira vez e serve de retenção: o limiar 0.5 foi
escolhido sem ele.

| GS-HOTA oficial | Treino | Teste | Validação |
|---|---|---|---|
| 15 piores | +5.54 | +5.63 | +2.18 |
| 20 sorteados | +3.10 | −1.14 | +1.95 |

### Agregado dos 50 vídeos testados (em cada 100 aparições de um jogador)

| O que acontece ao jogador | Dissertação | Confiança ≥ 0.5 | Diferença |
|---|---|---|---|
| Recebe o número certo | 46.6 | 55.1 | +8.5 |
| Recebe um número errado | 14.3 | 4.1 | −10.2 |
| Recebe um número inventado | 1.6 | 0.3 | −1.3 |
| Fica sem número, e bem | 8.1 | 9.3 | +1.2 |
| Fica em branco, mas tinha número | 29.4 | 31.1 | +1.7 |

| Asneiras / certos por conjunto | Treino | Teste | Validação |
|---|---|---|---|
| Número errado | 16.8 → 2.6 | 13.8 → 4.7 | 13.0 → 4.5 |
| Número certo | 43.0 → 51.9 | 47.6 → 56.0 | 48.2 → 56.6 |

GS-HOTA sobe em 11 das 12 combinações ensaio × conjunto. 42 de 50 vídeos melhoram ou ficam iguais;
4 perdem mais de 1 ponto (SNGS-121, 133, 139, 145, todos de teste com a dissertação já a acertar muito).
Causa diagnosticada: confusões de dígitos parecidos com confiança alta nos dois sentidos (8↔6, 14↔16).
Testadas e rejeitadas: soma das confianças em vez de contagem (resultado idêntico) e híbrido que
recupera a decisão da dissertação quando não há janelas confiantes (não salva estes vídeos e traz as
asneiras de volta).

## 6. Alternativa sem GPU, também validada

Manter a decisão da dissertação só quando tem pelo menos 2 janelas a votar nela. Usa apenas os votos
já guardados nos registos, e foi medida nos 159 vídeos:

| | certo | errado | vazio | abst. certa | inventado |
|---|---|---|---|---|---|
| Dissertação | 51.4 | 12.3 | 22.9 | 10.9 | 2.5 |
| ≥ 2 votos | 48.7 | 5.9 | 32.0 | 12.8 | 0.6 |

GS-HOTA: teste 43.76 → 44.01, validação 35.13 → 34.85, treino 42.79 → 42.25. Neutra na métrica
oficial; corta os errados para metade. A regra da confiança domina-a em tudo.

## 7. Evidência: a anotação também falha

No SNGS-187, o *tracklet* 4 tem o **14** visível nas costas ao longo de todo o vídeo; a anotação
regista o jogador sem número. Os crops têm 23×22 píxeis — abaixo do mínimo do filtro da dissertação
(25 × 30), que por isso nunca o leu. Imagem em `jnr_exp2/evidencia/`. A métrica conta esta leitura
correta como número inventado.

## 8. Ressalvas e passos seguintes

- O limiar 0.5 foi escolhido em teste e validação, como os restantes limiares da dissertação.
  **Falta confirmar no conjunto de treino**, que funciona como conjunto de retenção.
- A amostra é de 10 vídeos. O ganho no conjunto completo pode ser diferente.
- A leitura por janela passa a ser individual em vez de agrupada; na releitura, 29% das janelas deram
  resposta diferente da corrida original — sintoma da instabilidade das leituras pouco confiantes.
- Para aplicar aos 159 vídeos: implementar no módulo (leitura por janela + probabilidade dos tokens +
  votação com limiar) e voltar a correr o módulo do número e a agregação (~26 h de GPU). Não toca na
  deteção, no seguimento, na calibração nem na classificação de função.

## Reprodução

| Ficheiro | Função |
|---|---|
| `jnr_exp3/corre_amostra.sh` | corre o módulo do número sobre os estados da dissertação (filtro 24/18/25/2), com crops guardados |
| `jnr_exp3/releitura_conf.py` | relê as janelas com a confiança real do Qwen |
| `jnr_exp3/avalia_conf.py` | categorias ponderadas por deteções para cada regra |
| `jnr_exp3/gshota_conf.py` | GS-HOTA oficial por regra, alterando o número nos JSON de predição |
| `jnr_exp3/gshota_tese2_total.py` | regra dos 2 votos nos 159 vídeos, sem GPU |
