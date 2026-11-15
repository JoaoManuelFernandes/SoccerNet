#!/bin/bash
# Espera a cadeia pós-recalibração, gera o anexo e o zip de plano B.
while systemctl --user is-active pos-recalibracao >/dev/null 2>&1; do sleep 60; done
cd /home/joao/soccernet
venv_metrics/bin/python tese/gerador/gera_anexo.py > /dev/null 2>&1
rm -f tese/pacote_chatgpt.zip
zip -q -j tese/pacote_chatgpt.zip tese/PROMPT_CHATGPT.md tese/LEIA-ME_ficheiros_tese.md tese/*_final.tex tese/fig_*.pdf tese/gerador/*_template.tex tese/gerador/gera_tese.py \
  nosso_out/gshota/DECOMPOSICAO_ATRIBUTOS.md nosso_out/gshota/RESUMO_GSHOTA.md nosso_out/TABELA_VIDEO_A_VIDEO.md nosso_out/COMPARACAO_FINAL.md \
  nosso_out/VEREDICTOS_VISUAIS_NOSSO.md baseline_out/VEREDICTOS_VISUAIS_BASELINE.md nosso_out/TEMPOS_COMPARACAO.md
echo "[$(date '+%F %T')] pacote_chatgpt.zip feito ($(du -h tese/pacote_chatgpt.zip | cut -f1))" >> nosso_out/gshota/pos_progresso.log
