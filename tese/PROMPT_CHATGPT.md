# Instruções para fechar os textos da tese sem o Claude (plano B)

## Situação
Todos os capítulos estão escritos em LaTeX na pasta `tese/` (ficheiros `*_final.tex`). Os números das tabelas
e do texto são preenchidos automaticamente por `gerador/gera_tese.py` a partir de ficheiros de dados; o texto
está em `gerador/*_template.tex` com marcadores `@@NOME@@`. Uma cadeia automática (`pos-recalibracao`) refaz os
dados da baseline corrigida e regenera os ficheiros. Só é preciso intervenção humana/IA se essa cadeia falhar
ou para rever dois parágrafos cujo sentido depende dos números finais.

## Como verificar se a cadeia acabou
`tail -3 nosso_out/gshota/pos_progresso.log` → deve terminar em `PÓS-RECALIBRAÇÃO CONCLUÍDA`.
Se sim: os `tese/*_final.tex` estão prontos a copiar (ver `LEIA-ME_ficheiros_tese.md`). Saltar para "Revisão".
Se não: correr à mão, por esta ordem (cada comando é independente):
```
venv_metrics/bin/python nosso_out/tools/compara_final.py
venv_metrics/bin/python nosso_out/tools/tabela_video_a_video.py
venv_metrics/bin/python tese/gerador/gera_tese.py
venv_metrics/bin/python tese/gerador/gera_anexo.py
venv_tl/bin/python sn-gamestate/docs/figuras_tese/fig_cap4.py
```
(o GS-HOTA e a decomposição são `bash nosso_out/gshota/corre_gshota.sh` e `bash nosso_out/gshota/corre_decomposicao.sh`,
com `export ORIG_SRC=/home/joao/soccernet/baseline_out/states_ocr_c1`, a partir de `sn-gamestate/`; demoram ~40 min.)

## Prompt para o ChatGPT (colar, anexando os ficheiros listados abaixo)
> Sou aluno de mestrado e estou a fechar a minha dissertação (português de Portugal, LaTeX). Anexo:
> (1) `Chapter4_final.tex` e `Chapter5_Conclusao_final.tex`, já gerados com os números finais;
> (2) `DECOMPOSICAO_ATRIBUTOS.md` (GS-HOTA com atributos ativados progressivamente, para a solução de
> referência corrigida e para o sistema proposto, nos três conjuntos) e `RESUMO_GSHOTA.md`.
> Tarefa: rever APENAS a subsecção "Decomposição por Atributo" do capítulo 4 (parágrafo que começa por
> "A leitura é direta") e o parágrafo da conclusão que começa por "Os ganhos, medidos sobre a totalidade",
> confirmando frase a frase que o que o texto afirma é o que os números do `DECOMPOSICAO_ATRIBUTOS.md`
> mostram. Regras: (a) o custo de cada atributo é a diferença entre colunas consecutivas; (b) o ganho do
> sistema proposto num atributo é o custo na referência menos o custo no proposto; (c) se "só posição" e
> "+função" não coincidirem entre os dois sistemas por mais de 0.5 pontos, a frase "coincidem, como têm de
> coincidir" tem de ser reescrita para dizer que diferem por X pontos devido a diferenças residuais da
> calibração (RANSAC); (d) se o ganho da equipa em algum conjunto for negativo ou nulo, dizê-lo
> explicitamente em vez de "vem inteiramente dos dois atributos alterados". Não alterar mais nada, não
> inventar números, devolver os dois parágrafos corrigidos em LaTeX.

## Ficheiros a anexar ao ChatGPT
- `tese/Chapter4_final.tex`, `tese/Chapter5_Conclusao_final.tex`
- `nosso_out/gshota/DECOMPOSICAO_ATRIBUTOS.md`, `nosso_out/gshota/RESUMO_GSHOTA.md`
- (se for preciso preencher marcadores à mão) `tese/gerador/*_template.tex`, `nosso_out/TABELA_VIDEO_A_VIDEO.md`,
  `nosso_out/COMPARACAO_FINAL.md`, `nosso_out/VEREDICTOS_VISUAIS_NOSSO.md`, `baseline_out/VEREDICTOS_VISUAIS_BASELINE.md`,
  `nosso_out/TEMPOS_COMPARACAO.md`. Os marcadores `@@...@@` estão definidos em `tese/gerador/gera_tese.py` (secção por secção).

Um zip com tudo isto é feito automaticamente no fim da cadeia: `tese/pacote_chatgpt.zip`.
