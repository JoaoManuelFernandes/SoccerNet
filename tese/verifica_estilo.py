"""Verificador de estilo da tese.

Lê um ou mais parágrafos em LaTeX (ficheiro ou stdin) e assinala, frase a frase,
o que o João já rejeitou por soar a IA ou a português do Brasil, mais as frases
longas e as palavras repetidas na mesma frase.

Uso: python3 -I verifica_estilo.py paragrafo.tex
     pbpaste | python3 -I verifica_estilo.py
"""
import re
import sys

# (padrão, mensagem). Os padrões vêm das correções do João (feedback_escrita_tese).
PROIBIDOS = [
    (r"[—–]", "travessão"),
    (r";", "ponto e vírgula, partir em duas frases"),
    (r"\ba saber\b", "«a saber»"),
    (r"\b[Ii]mporta (sublinhar|referir|notar|ler)", "«importa ...»"),
    (r"\bNão obstante\b", "«não obstante»"),
    (r"\bConvém\b", "«convém»"),
    (r"\bEste achado\b", "«este achado»"),
    (r"\bprofundamente\b", "«profundamente»"),
    (r"\bde forma concertada\b", "«de forma concertada»"),
    (r"\bRecorde-se\b", "«recorde-se»"),
    (r"\bsuscit", "«suscitar»"),
    (r"\barticulando\b", "«articulando»"),
    (r"\bincorpor", "«incorporar», usar «incluir»"),
    (r"\bpreviamente\b", "«previamente»"),
    (r"\bintervencionad", "«intervencionado», usar «alterado»"),
    (r"\bpor seu turno\b", "«por seu turno»"),
    (r"\bmerece ser\b", "«merece ser»"),
    (r"\bcom honestidade\b", "«com honestidade»"),
    (r"\bé apenas parte\b", "«é apenas parte»"),
    (r"\b(é|são) medid[oa]s? pel[oa]s?\b", "«é medido pela», dizer o que se conta"),
    # Floreados típicos de texto gerado (vistos numa reescrita de outra IA, 26/10)
    (r"\btorna-se (ainda )?mais (evidente|claro)\b", "«torna-se mais evidente»"),
    (r"\bEm contrapartida\b", "«em contrapartida»"),
    (r"\bpatamar\b|\ba marca d[oa]s?\b", "«patamar»/«a marca de», dizer o valor"),
    (r"\bsem registar qualquer\b", "«sem registar qualquer»"),
    (r"\b(alcançando|evidenciando|demonstrando|revelando)\b", "gerúndio de floreado, partir a frase"),
    (r"\bAdicionalmente\b|\bpropriamente dit[oa]\b|\bmotivo fundamental\b", "floreado"),
    (r"\bsignificativamente\b", "«significativamente» sem teste estatístico, dar o número"),
    (r"pixels|pixéis", "usar «píxeis», como no resto da tese"),
    (r"\b[Ff]orma mais (simples|direta)\b|\buma forma de\b", "«forma» para um método, usar método/abordagem"),
    # Termos decididos
    (r"\bseguimento\b", "«seguimento», usar \\emph{tracking}"),
    (r"\bsolução de referência\b", "usar «a \\emph{baseline}»"),
    (r"\bexecutad[ao] por nós\b|\b(substituímos|alterámos|usámos)\b", "voz pessoal, usar a impessoal"),
    (r"\bleitor de números\b", "usar «reconhecimento do número da camisola»"),
    (r"\bnúmero de camisola\b", "usar «número DA camisola»"),
    (r"\bprecisão\b", "usar «exatidão»"),
    (r"\bnunca (foi )?(observad|interveio)|\bpor observar\b", "treino: dizer «nenhuma regra nem limiar foi escolhido com o treino»"),
    (r"\b(tasks?|features?)\b", "inglês com tradução corrente (tarefa, características)"),
    # Português do Brasil
    (r"\b(registro|arquivo|tela|você|treinamento|usuário|equipe|controle|time)\b", "brasileirismo"),
    (r"\b(está|estão|estava|estavam|vai|vão|fica|ficam|vem|vêm) \w+ndo\b", "gerúndio brasileiro, usar «a + infinitivo»"),
]

# Palavras que podem repetir-se na mesma frase sem problema.
VAZIAS = set("""
o a os as um uma uns umas de do da dos das em no na nos nas por pelo pela pelos
pelas para com sem que se e ou mas como mais menos não é são foi foram ser ter
tem têm cada este esta estes estas esse essa esses essas isso isto entre sobre
também já só ainda mesmo mesma mesmos mesmas quando onde porque qual quais seu
sua seus suas lhe lhes ao aos à às N baseline tracklets tracklet frames frame
""".split())

MAX_PALAVRAS = 30


def limpa_latex(texto):
    texto = re.sub(r"(?<!\\)%.*", "", texto)
    texto = re.sub(r"\$[^$]*\$", "N", texto)
    texto = re.sub(r"\\(cite[pt]?|ref|label|eqref)(\[[^\]]*\])*\{[^}]*\}", "X", texto)
    texto = re.sub(r"\\(emph|textbf|textit|texttt)\{([^}]*)\}", r"\2", texto)
    texto = re.sub(r"\\(chapter|section|subsection)\*?\{[^}]*\}", "", texto)
    texto = re.sub(r"\\[a-zA-Z]+\*?", "", texto)
    texto = texto.replace("~", " ").replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", texto).strip()


def frases(texto):
    texto = texto.replace("et al.", "et al")
    return [f.strip() for f in re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÀ])", texto) if f.strip()]


def verifica_frase(frase):
    avisos = []
    for padrao, msg in PROIBIDOS:
        if re.search(padrao, frase):
            avisos.append(msg)
    if re.search(r":\s+[a-záéíóúâêôàç]", frase):
        avisos.append("«:» a meio da frase")
    palavras = re.findall(r"[\wÀ-ÿ-]+", frase)
    if len(palavras) > MAX_PALAVRAS:
        avisos.append(f"frase longa ({len(palavras)} palavras)")
    contagem = {}
    for p in palavras:
        p = p.lower()
        if len(p) >= 5 and p not in VAZIAS:
            contagem[p] = contagem.get(p, 0) + 1
    repetidas = [p for p, n in contagem.items() if n > 1]
    if repetidas:
        avisos.append("repete " + ", ".join(f"«{p}»" for p in repetidas))
    return len(palavras), avisos


def main():
    fonte = open(sys.argv[1], encoding="utf-8") if len(sys.argv) > 1 else sys.stdin
    paragrafos = [p for p in re.split(r"\n\s*\n", fonte.read()) if p.strip()]
    total = 0
    for i, par in enumerate(paragrafos, 1):
        lista = frases(limpa_latex(par))
        print(f"== Parágrafo {i} ({len(lista)} frases)")
        for j, frase in enumerate(lista, 1):
            n, avisos = verifica_frase(frase)
            if avisos:
                total += len(avisos)
                print(f"  {j:2d}. [{n} pal.] {frase[:70]}{'…' if len(frase) > 70 else ''}")
                for a in avisos:
                    print(f"      - {a}")
    print(f"== {total} avisos")


if __name__ == "__main__":
    main()
