# Tese com o sistema final (leitor com confiança) — para rever a 27/09

Nada da versão anterior foi alterado: `tese/*_final.tex`, `tese/overleaf/para_carregar/` e os dados em
`nosso_out/` (sem sufixo) continuam iguais. Esta pasta é a versão NOVA, completa.

## Para carregar no Overleaf (depois de aprovares)
`para_carregar/` — mesma árvore e mesmo procedimento de sempre (Upload → arrastar; substitui).
Verificado: todas as `\ref` resolvem, todas as `\cite` estão no .bib, ambientes e chavetas equilibrados.
Não há LaTeX neste PC, por isso não foi compilado — compilar no Overleaf duas vezes.

## O que muda
| Onde | O quê |
|---|---|
| Cap. 3 (Chapter_02.tex) | filtro de qualidade permissivo (15×20 px, nitidez 15) e porquê; subsecção **Confiança da Leitura e Decisão por Janela** substitui a "Agregação Temporal por Votação"; tabela de hiperparâmetros atualizada |
| Cap. 4 (Chapter3.tex) | tab:jnr e distribuição com o sistema final (número 15.3% → **42.6%**, nenhuma regressão); fecho novo do "Teto"; subsecções novas **Controlo pela Confiança da Leitura** (5 categorias, GS-HOTA antes/depois, 056/188/196) e **Alternativas Testadas e Rejeitadas** (fusão, regras alternativas, unicidade do número incl. filtro espacial, lição sobre a métrica); custo da releitura; GS-HOTA e decomposição com o sistema final; síntese |
| Conclusão | a confiança como 3.º passo decisivo; tentativas rejeitadas; 5.ª linha de trabalho futuro (cortar tracklets nas trocas de identidade) |
| Introdução, Resumo, Abstract | "votação" → decisão com as leituras seguras; números finais |
| Anexo | tabela vídeo a vídeo com o sistema final |
| Figuras | fig_delta_por_video e fig_decomposicao_gshota com o sistema final |

## Números finais (proposto; referência inalterada)
GS-HOTA teste **47.87** (ref. 31.30), validação **36.79** (23.68), treino **45.37** (31.52).
Número por tracklet 15.3% → **42.6%**; equipa 83.5% → 92.8% (inalterada face à versão anterior).

## Como refazer
`SUF=_conf TESE_OUT=tese/final_conf TPL_DIR=tese/gerador_conf` + gera_tese.py / gera_anexo.py / fig_cap4.py
(e nosso_out/tools/tabela_video_a_video.py); dados: nosso_out/COMPARACAO_FINAL_conf.md,
TABELA_VIDEO_A_VIDEO_conf.csv, gshota/RESUMO_GSHOTA_conf.md, DECOMPOSICAO_ATRIBUTOS_conf.md, conf_*.json.
Pacote: `bash tese/final_conf/monta_para_carregar_conf.sh`.
