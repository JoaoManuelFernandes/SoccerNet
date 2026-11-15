# Falhas de atribuição confirmadas por inspeção visual

Lista do que o João viu nos vídeos renderizados. Substitui a triagem automática
de `VALIDAR_team_assignment.md`, que listava candidatos; esta lista é observada.

## TESTE

| Vídeo | O que se vê | Modo de falha |
|---|---|---|
| SNGS-126 | Nós estamos corretos; a anotação é que está errada | **Não é falha nossa** |
| SNGS-131 | Todos os jogadores dados do mesmo lado | Inversão/colapso do lado |
| SNGS-192 | Vídeo sem conteúdo útil | Não avaliável |

## VALIDAÇÃO

| Vídeo | O que se vê | Modo de falha |
|---|---|---|
| SNGS-025 | Tudo do mesmo lado, como o 131 | Inversão/colapso do lado |
| SNGS-027 | Tudo do mesmo lado | Inversão/colapso do lado |
| SNGS-028 | Igual ao 027 | Inversão/colapso do lado |
| SNGS-029 | Igual ao 027 | Inversão/colapso do lado |
| SNGS-036 | Falha sem ser lance de bola parada | Inversão/colapso do lado |
| SNGS-038 | Igual ao 036 | Inversão/colapso do lado |
| SNGS-041 a SNGS-058 | Jogadores de equipamento amarelo classificados como árbitro; o lado fica logo mal escolhido | **Erro de FUNÇÃO que arrasta o lado** |
| SNGS-091 | Tudo do mesmo lado | Inversão/colapso do lado |
| SNGS-190 | Correto. Só faltam as linhas de calibração no início | **Não é falha** |

## TREINO

| Vídeo | O que se vê | Modo de falha | GK na anotação |
|---|---|---|---|
| SNGS-069 | Tudo dado como a mesma equipa | **Colapso do agrupamento** (não é falha de lado) | **NÃO TEM** |
| SNGS-113 | Equipas trocadas: o que dizemos "right" é "left" | Inversão do lado | Sim (473+28 anot.) |

## Observação do João, transversal

Vários destes vídeos começam num recomeço de jogo (bola parada), com as equipas
agrupadas numa zona do campo em vez de distribuídas. Isto é consistente com o
medido no SNGS-113, em que as duas equipas têm as medianas a +7.86 e +17.58,
ambas na mesma metade e separadas por apenas 9.7 m.

## Modos de falha distintos (não confundir na tese)

1. **Colapso do agrupamento** — um só grupo. Não há decisão de lado a tomar. (069)
2. **Inversão do lado** — dois grupos bem separados, mapeados ao contrário. (113, 131, 025-029, 036, 038, 091)
3. **Erro de função** — jogadores classificados como árbitro, o que corrompe o
   agrupamento e o lado a jusante. (041-058) Este é o que a Secção 4.3.1 discute.
4. **Anotação errada** — o sistema está certo. (126)

## Disponibilidade de guarda-redes

Só 4 dos 160 vídeos não têm guarda-redes anotado: **SNGS-069, SNGS-055,
SNGS-127, SNGS-138**. Dos 9 vídeos de treino com falha, 8 têm guarda-redes.
O sinal que resolveria a maioria destes casos existe, mas a cascata coloca-o
em último lugar (critério 4) e só o consulta quando o critério 3 é atingido e as
médias dos grupos ficam a menos de 0.3 m.

## SNGS-113: rasto real da cascata (fonte: docs/render_logs/SNGS-113.log, linhas 3206-3234)

Não foi preciso reexecutar: o registo do render de 2026-09-09 já tem a decisão.

| Passo | Valor registado | Resultado |
|---|---|---|
| 1. Votação por tracklet | C0 = 33 right / 8 left (80%); C1 = 23 right / 1 left (96%) | ambos ≥60% right → conflito, avança |
| 2. Votação relativa | C0_right=80%, C1_right=96%, diff=−15 pp (limiar 10 pp) | **decide**: C0→left, C1→right |
| 3. Média de x | nunca calculada | — |
| 4. Desempate pelo GR | nunca consultado (GR detetado: 457 deteções, lado left) | — |

Veredicto visual do João: mapeamento invertido. Ou seja, a decisão saiu do
critério 2, não do 3, e a frase anterior ("estavam a 9.7 m") refere-se às
medianas da anotação, não a algo que a cascata tenha calculado. O que o
critério 2 assume é que o grupo mais à direita defende a baliza direita; com
as duas equipas na mesma metade (medianas da anotação +7.9 e +17.6) essa
assunção fica sem margem. Nota: a votação conta fragmentos de tracklet
(65 tracklets para ~20 jogadores), não jogadores, pelo que um jogador
fragmentado pesa várias vezes.

## Por correr

Reexecutar com registo ativo no módulo do lado do campo, para apurar que
critério disparou em cada caso e por que margem:
SNGS-131, SNGS-025, SNGS-027, SNGS-028, SNGS-029, SNGS-036,
SNGS-038, SNGS-091. (~20 min de GPU por vídeo; o SNGS-113 já está apurado
acima a partir do registo existente. Verificar primeiro se docs/render_logs/
tem o registo destes vídeos antes de reexecutar.)

## Baseline ORIGINAL no teste (2026-09-14, 48 vídeos vistos, ver baseline_out/VEREDICTOS_VISUAIS_BASELINE.md)

- NOK-grupo: 124. NOK-lado: 137, 140, 143, 148, 190.
- **Anotação invertida confirmada visualmente: 126, 131, 197** (a baseline está certa e a métrica dá 0%).
  Isto fecha a dúvida de verificacao_anotacao_lado.md: o teste do guarda-redes não apanha a inversão,
  logo a anotação inverteu equipas E guarda-redes de forma consistente nesses vídeos.

- **Anotação invertida também no treino: SNGS-111** (baseline original visualmente certa, métrica 0%). Total de anotações invertidas confirmadas: 126, 131, 197 (teste) e 111 (treino).
