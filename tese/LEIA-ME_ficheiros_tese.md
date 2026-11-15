# ATUALIZAÇÃO 2026-09-27 — sistema final com o leitor com confiança
`overleaf/para_carregar/` passou a ter a versão NOVA (a anterior está em `overleaf/para_carregar_antes_confianca/`).
Upload no Overleaf, cada ficheiro para a pasta com o mesmo nome (substitui):

| Pasta no Overleaf | Ficheiros |
|---|---|
| `Chapters/` | `Introdução.tex`, `Chapter_01.tex` (igual), `Chapter_02.tex` (Cap. 3), `Chapter3.tex` (Cap. 4), `Conclusions.tex`, `Attachments.tex` |
| `FrontCover/` | `AbstractInPortuguese.tex`, `AbstractInEnglish.tex` |
| `Images/` | `fig_delta_por_video.pdf`, `fig_decomposicao_gshota.pdf`, `fig_frames_148.pdf` |
| `Bibliography/`, `Configurations/` | iguais à versão anterior (podem ir na mesma) |

Depois: Recompile duas vezes. Detalhe do que mudou: `final_conf/LEIA-ME.md`.

---

# Ficheiros finais da tese (gerados 2026-09-21 23:32, referência recalibrada e corrigida)

## Para carregar no Overleaf: pasta `overleaf/para_carregar/`
Cada ficheiro vai para a pasta com o MESMO nome no projeto Overleaf (Upload → arrastar; substitui o existente).

| Pasta no Overleaf | Ficheiro | O que muda |
|---|---|---|
| `Chapters/` | `Introdução.tex` | Cap. 1, números finais; ref. `cap:estado-arte` corrigida |
| `Chapters/` | `Chapter_01.tex` | Estado da arte (Cap. 2) REVISTO a 2026-09-22 (ponto 5): panorama das edições 2024/2025/2026 em §2.1, decomposição de Oo et al. corrigida em §2.2.3, decomposição por atributo como instrumento em §2.4, precisão sobre a reidentificação em §2.1 e §2.5, label `sec:sistemas_gsr` |
| `Bibliography/` | `references.bib` | + entrada `cioppa2026soccernet` (SoccerNet 2026 Challenges Results, arXiv:2607.07320) |
| `Chapters/` | `Chapter_02.tex` | Metodologia (Cap. 3), validada; só acrescenta os labels `sec:ambiente` e `sec:protocolo` |
| `Chapters/` | `Chapter3.tex` | Resultados (Cap. 4), números finais |
| `Chapters/` | `Conclusions.tex` | Conclusão (Cap. 5), números finais |
| `Chapters/` | `Attachments.tex` | Anexo: tabela vídeo a vídeo (159 vídeos, `longtable`) |
| `FrontCover/` | `AbstractInPortuguese.tex` | Resumo, números finais |
| `FrontCover/` | `AbstractInEnglish.tex` | Abstract, números finais |
| `Configurations/` | `Configurations.tex` | + `\usepackage{longtable}` (o anexo precisa) |
| `Images/` | `fig_delta_por_video.pdf`, `fig_decomposicao_gshota.pdf`, `fig_frames_148.pdf` | figuras do Cap. 4 (as do Cap. 3 já lá estão e não mudaram) |

Depois: Recompile duas vezes (referências cruzadas). Verificação feita aqui: todas as `\ref` resolvem,
todas as imagens existem, todas as `\cite` estão no `references.bib`.

## Números finais (referência corrigida vs proposto)
GS-HOTA teste 31.30 → 44.00 (+12.7), validação 23.68 → 33.17 (+9.5), treino 31.52 → 42.79 (+11.3).
"Só posição" e "+função" idênticos nos dois sistemas (teste de aceitação da recalibração).
Número 15.3% → 37.8% (159 vídeos); equipa/lado 83.5% → 91.7%.

## Como refazer
Os `*_final.tex` dos caps. 1, 4, 5, Resumo e Abstract saem de `gerador/gera_tese.py` (texto em `gerador/*_template.tex`,
dados em `nosso_out/`); o anexo de `gerador/gera_anexo.py`; as figuras de `sn-gamestate/docs/figuras_tese/fig_cap4.py`;
a pasta de upload de `gerador/monta_para_carregar.sh`. Cadeia completa: `nosso_out/gshota/pos_recalibracao.sh`.
Citações usadas: apenas `\citet{somers2024soccernet}` / `\citep{somers2024soccernet}`.
