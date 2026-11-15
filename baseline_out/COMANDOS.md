# Comandos essenciais (baseline original, cadeia `baseline-tudo`)

Tudo corre como unidade `systemd --user` chamada `baseline-tudo`. Faz, por vídeo e por esta ordem
(test → valid → train, 160 vídeos): pipeline original → estado + MP4; OCR original a partir do estado;
encolhe os estados; atualiza a tabela. Retoma sempre onde ficou (só refaz o que não tem MP4/OCR).

## Ver como está
    systemctl --user is-active baseline-tudo          # active = a correr
    tail -5 /home/joao/soccernet/baseline_out/progresso.log
    ls /home/joao/soccernet/baseline_out/videos | wc -l        # vídeos prontos
    ls /home/joao/soccernet/baseline_out/states_ocr/*.pklz | wc -l   # vídeos com OCR original
    df -h /home/joao                                    # disco livre (a cadeia pára aos 3 GB)
    free -h                                             # RAM

## (Re)lançar a cadeia  (depois de parar, de reiniciar o PC, ou de ela parar por disco/CUDA)
    /home/joao/soccernet/baseline_out/tools/lanca_tudo.sh

## Parar a cadeia (antes de suspender o PC, ou para libertar a GPU)
    systemctl --user stop baseline-tudo
    (perde-se só o vídeo que estiver a meio; refaz-se sozinho no relançamento)

## Depois de suspender/acordar o PC (a CUDA morre)
    sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm
    /home/joao/soccernet/venv_tl/bin/python -c "import torch; print(torch.cuda.is_available())"   # tem de dar True
    /home/joao/soccernet/baseline_out/tools/lanca_tudo.sh

## Controlar o disco
    Apagar SÓ os MP4 já avaliados (nunca apagar states/ nem states_ocr/):
    rm /home/joao/soccernet/baseline_out/videos/SNGS-118.mp4 /home/joao/soccernet/baseline_out/videos/SNGS-119.mp4
    Se a cadeia parou por disco ("PARADO: disco cheio" no progresso.log): apagar MP4 e relançar (lanca_tudo.sh).

## Tabela de comparação (a cadeia atualiza-a sozinha a cada vídeo)
    /home/joao/soccernet/baseline_out/COMPARACAO_POR_VIDEO.md
    Para forçar: /home/joao/soccernet/venv_metrics/bin/python /home/joao/soccernet/baseline_out/tools/compara_por_video.py

## Onde está o quê
    baseline_out/videos/       MP4 renderizados (equipa/lado originais; números do JNR v2 no ecrã)
    baseline_out/states/       estados (encolhidos, ~8 MB): equipa original + números JNR v2
    baseline_out/states_ocr/   estados 100% originais (números do MMOCR)
    baseline_out/logs/ logs_ocr/   registos por vídeo (têm as linhas [TEMPO] por módulo)
    baseline_out/progresso.log     diário da cadeia
    baseline_out/tools/            scripts (corre_tudo.sh, lanca_tudo.sh, passagem_ocr.sh, slim_estado.py, compara_por_video.py)

## Regras que evitam matar a sessão (7.6 GB de RAM)
    Uma coisa pesada de cada vez. Não correr outra pipeline/treino/pip install enquanto a cadeia corre.
    Não abrir os MP4 no VS Code; usar o reprodutor do sistema.

## Fase 3: a NOSSA pipeline nos 160 (unidade `nosso-tudo`)
    Arranca sozinha quando a baseline-tudo acabar. Só estados (sem MP4), encolhidos, em /home/joao/soccernet/nosso_out/states.
    systemctl --user is-active nosso-tudo
    tail -3 /home/joao/soccernet/nosso_out/progresso.log
    ls /home/joao/soccernet/nosso_out/states | wc -l                    # de 160
    /home/joao/soccernet/nosso_out/tools/lanca_nosso.sh                  # (re)lançar; retoma onde ficou
    systemctl --user stop nosso-tudo                                     # parar (antes de suspender)
    Tabela final (original vs nosso, por vídeo e por conjunto): /home/joao/soccernet/nosso_out/COMPARACAO_FINAL.md
    Disco: a nossa cadeia não gera MP4; cada vídeo custa ~8 MB. Pára sozinha aos 3 GB.

## 2026-09-20 — fecho dos textos da tese (caso eu fique sem tokens)
- Tudo automático esta noite: `recalibra-c1` (baseline recalibrada frame a frame) → `pos-recalibracao` (GS-HOTA, decomposição, tabelas, capítulos, figuras).
- Ver se acabou: `tail -3 nosso_out/gshota/pos_progresso.log` → tem de dizer `PÓS-RECALIBRAÇÃO CONCLUÍDA`.
- Se tiver acabado, os ficheiros finais são `tese/*_final.tex` + `tese/fig_*.pdf` (mapa em `tese/LEIA-ME_ficheiros_tese.md`). Copiar e colar.
- Se precisares de refazer à mão: `venv_metrics/bin/python tese/gerador/gera_tese.py` (capítulos) e `venv_tl/bin/python sn-gamestate/docs/figuras_tese/fig_cap4.py` (figuras).
- Estado das unidades: `systemctl --user is-active recalibra-c1 pos-recalibracao`.
