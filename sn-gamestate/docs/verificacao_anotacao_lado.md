# Verificação da anotação de equipa no conjunto de teste

**Data:** 2026-09-09
**Motivo:** Oo et al., *Broadcast2Pitch* (WACV 2026), vencedores do desafio de 2025,
afirmam na nota da sua Tabela 7 que os vídeos **SNGS-126, SNGS-131 e SNGS-197**
estão *"incorrectly annotated for the team attribute"* no conjunto de teste do
SoccerNet-GSR. Como o SNGS-126 é o único caso sem solução segura do nosso módulo
de atribuição de lado, a afirmação tinha de ser verificada.

**Reprodução:** `python3 verifica_anotacao_lado.py [caminho_do_split]`

---

## 1. Método

Dois testes independentes sobre a anotação de referência, por vídeo.

**Teste 1, âncora do guarda-redes.** O guarda-redes da equipa rotulada `left` tem de
estar junto à baliza esquerda (x negativo) e o da equipa `right` junto à direita.
É o sinal mais fiável de qual lado uma equipa ocupa, porque, ao contrário dos
jogadores de campo, o guarda-redes não acompanha as fases de ataque. Exigiram-se
pelo menos 30 deteções para o considerar.

**Teste 2, heurística ingénua da média.** Compara a posição horizontal mediana dos
jogadores de cada equipa e assinala os vídeos onde a ordem se inverte. É a
suposição usada por várias soluções do desafio, incluindo a vencedora de 2025, que
atribui o lado agrupando por cor e comparando as posições médias de cada grupo.

## 2. Resultados (49 vídeos de teste)

### Teste 1

| | |
|---|---|
| Guarda-redes coerente com o rótulo | **47** |
| Sem guarda-redes utilizável | 2 (SNGS-127, SNGS-138) |
| **Incoerentes** | **0** |

Nos três vídeos assinalados, a anotação é coerente:

| Vídeo | Equipa do GR | x mediano | Deteções | Veredicto |
|---|---|---|---|---|
| SNGS-126 | `left` | −48.1 m | 605 | coerente |
| SNGS-131 | `left` | −48.6 m | 675 | coerente |
| SNGS-197 | `right` | +48.0 m | 372 | coerente |

### Teste 2

A heurística da média inverte-se em exatamente **cinco** dos 49 vídeos:

| Vídeo | Jogadores `left` | Jogadores `right` |
|---|---|---|
| SNGS-126 | −18.0 m | −22.7 m |
| SNGS-130 | +33.2 m | +33.1 m |
| SNGS-131 | −27.4 m | −31.7 m |
| SNGS-195 | −23.7 m | −24.4 m |
| SNGS-197 | +18.4 m | +13.3 m |

## 3. Conclusão

**O teste geométrico exclui uma inversão global da anotação, mas não a esgota.** Nos
155 vídeos em que existe guarda-redes observável, a posição dele nunca contradiz o
rótulo da sua própria equipa. Se os rótulos `left` e `right` estivessem trocados na
totalidade de um vídeo, o guarda-redes apareceria do lado errado e o teste
detetá-lo-ia. Não é isso que acontece em nenhum dos três vídeos assinalados.

**A inspeção visual identificou, ainda assim, um erro que este teste não alcança.** A
observação direta do SNGS-126 indica que a atribuição correta é a produzida pelo
sistema e não a da anotação. As duas conclusões são compatíveis, e a hipótese que as
concilia é a de uma **inconsistência interna**, ou seja, o rótulo do guarda-redes estar
correto e os rótulos dos jogadores de campo estarem trocados em relação a ele. O teste
aqui descrito compara o guarda-redes apenas consigo próprio, e por isso é cego a esse
caso.

Existem assim **duas indicações independentes** de que a anotação do SNGS-126 tem
problema, a saber, a afirmação de Oo et al. e a inspeção visual feita neste trabalho.
O que este relatório acrescenta é a delimitação do erro, que não é uma troca global
dos lados.

**Consequência para os valores reportados.** Se a anotação estiver errada neste vídeo,
a exatidão da atribuição de equipa no conjunto de teste está **subestimada**, e não
sobrestimada. Nenhum valor publicado nesta dissertação fica inflacionado por esta
questão. Enquanto a verificação não estiver fechada, mantêm-se os valores medidos
contra a anotação oficial, que é o procedimento correto.

**Verificação em falta.** Confirmar, sobre as predições guardadas, se o sistema
concorda com a anotação quanto à equipa do guarda-redes e discorda quanto aos
jogadores de campo. Esse padrão confirmaria a inconsistência interna de forma objetiva.

## 4. O resultado que se mantém, e é o mais útil

Independentemente do que se conclua sobre o SNGS-126, o Teste 2 produz uma medição que
não depende dessa questão. **A heurística da posição média, que é o critério em que
assentam várias das soluções apresentadas ao desafio, inverte-se em 9 dos 160 vídeos,
ou seja em 5.6% dos casos.** Nenhum trabalho publicado quantifica esta taxa de falha.

Os três vídeos que Oo et al. assinalam são um subconjunto exato desses nove, e o
critério de atribuição de lado que eles descrevem é precisamente a comparação das
posições médias dos grupos. A leitura mais plausível é que tenham identificado o ponto
de falha do próprio método.

## 5. Limitações

O Teste 1 é ao nível do vídeo e compara o guarda-redes apenas com o seu próprio rótulo,
pelo que não deteta inconsistências entre o guarda-redes e os jogadores de campo, nem
erros em *tracklets* individuais. O Teste 2 assume que a anotação de referência está
correta, o que é a hipótese em avaliação, e por isso serve para caracterizar o
comportamento do critério e não para validar os dados.
