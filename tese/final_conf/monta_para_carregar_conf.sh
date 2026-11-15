#!/bin/bash
# Monta tese/overleaf/para_carregar/ com a árvore EXATA do projeto Overleaf (IPCAThesisJF): cada ficheiro
# vai para a pasta com o mesmo nome no Overleaf (Upload → arrastar; substitui os existentes).
set -e
T0=/home/joao/soccernet/tese; T=$T0/final_conf; S=$T0/overleaf/src; D=$T/para_carregar
rm -rf "$D"; mkdir -p "$D/Chapters" "$D/FrontCover" "$D/Configurations" "$D/Images"
# capítulos gerados / validados
cp "$T/Chapter1_Introducao_final.tex"  "$D/Chapters/Introdução.tex"
cp "$T/Chapter3_final.tex"             "$D/Chapters/Chapter_02.tex"      # Metodologia (cap. 3)
cp "$T/Chapter4_final.tex"             "$D/Chapters/Chapter3.tex"        # Resultados (cap. 4)
cp "$T/Chapter5_Conclusao_final.tex"   "$D/Chapters/Conclusions.tex"
cp "$T/Anexo_final.tex"                "$D/Chapters/Attachments.tex"
cp "$T/Resumo_final.tex"               "$D/FrontCover/AbstractInPortuguese.tex"
cp "$T/Abstract_final.tex"             "$D/FrontCover/AbstractInEnglish.tex"
# Estado da arte (Cap. 2) revisto a 2026-09-22 (ponto 5 do orientador) + bibliografia com a entrada cioppa2026soccernet
mkdir -p "$D/Bibliography"
cp "$T/Chapter2_EstadoArte_final.tex" "$D/Chapters/Chapter_01.tex"
cp "$T0/references_final.bib"          "$D/Bibliography/references.bib"
cp "$T/jnr_flow.png" "$D/Images/jnr_flow.png"
# Configurations: + longtable (anexo)
sed 's/^\\usepackage{array}$/\\usepackage{array}\n\\usepackage{longtable}\t\t\t\t% Tabela do anexo (vídeo a vídeo)/' "$S/Configurations/Configurations.tex" > "$D/Configurations/Configurations.tex"
grep -q 'usepackage{longtable}' "$D/Configurations/Configurations.tex" || { echo "ERRO: longtable não inserido"; exit 1; }
printf '\n%% Espaço entre a legenda e a tabela (legendas acima das tabelas)\n\\AtBeginEnvironment{table}{\\setlength{\\belowcaptionskip}{6pt}}\n' >> "$D/Configurations/Configurations.tex"
# Figuras: só as que faltam ou mudaram no Overleaf
for f in "$T"/fig_*.pdf; do b=$(basename "$f"); if [ ! -f "$S/Images/$b" ] || ! cmp -s "$f" "$S/Images/$b"; then cp "$f" "$D/Images/"; fi; done
# verificação de referências: labels vs refs no conjunto final
ALL="$D/Chapters/*.tex $D/FrontCover/*.tex"
grep -ohE "\\\\cite[pt]?\{[^}]+\}" $ALL | sed -E 's/\\cite[pt]?\{//;s/}//' | tr ',' '\n' | sed 's/ //g' | sort -u > /tmp/_cites.txt
grep -oE "^@[a-zA-Z]+\{[^,]+," "$D/Bibliography/references.bib" | sed -E 's/^@[a-zA-Z]+\{//;s/,$//' | sort -u > /tmp/_bib.txt
FALTA=$(comm -23 /tmp/_cites.txt /tmp/_bib.txt); [ -z "$FALTA" ] && echo "cites: todas no .bib" || { echo "CITES SEM ENTRADA: $FALTA"; exit 1; }
grep -ohE "\\\\label\{[^}]+\}" $ALL | sed 's/\\label{//;s/}//' | sort -u > /tmp/_labels.txt
grep -ohE "\\\\(ref|autoref|pageref)\{[^}]+\}" $ALL | sed -E 's/\\(ref|autoref|pageref)\{//;s/}//' | sort -u > /tmp/_refs.txt
FALTA=$(comm -23 /tmp/_refs.txt /tmp/_labels.txt); [ -z "$FALTA" ] && echo "refs: todas resolvem" || { echo "REFS SEM LABEL: $FALTA"; exit 1; }
grep -ohE "\\\\includegraphics(\[[^]]*\])?\{[^}]+\}" $ALL | sed -E 's/.*\{([^}]+)\}/\1/' | sort -u | while read -r im; do [ -f "$S/Images/$im" ] || [ -f "$D/Images/$im" ] || echo "IMAGEM EM FALTA: $im"; done
echo "=== para_carregar:"; (cd "$D" && find . -type f | sort)
cp -r /home/joao/soccernet/tese/final_conf/extra_overleaf/. "$D/"   # abreviaturas sem o acento solto
