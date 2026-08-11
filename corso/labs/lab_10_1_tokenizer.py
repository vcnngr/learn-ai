#!/usr/bin/env python3
"""
Lab 10.1 — BPE addestrato da zero, e il danno silenzioso di un template.  CPU.

    python3 corso/labs/lab_10_1_tokenizer.py

Nessuna libreria di tokenizzazione: il BPE e' scritto qui, ~40 righe.
"""

from collections import Counter


def sez(t):
    print(f"\n{'─' * 72}\n{t}\n{'─' * 72}")


# ----------------------------------------------------------------------
sez("1. BPE: l'algoritmo, per intero")

print("""  1. parti dai singoli byte: vocabolario di 256 simboli, copre TUTTO
  2. conta le coppie adiacenti piu' frequenti nel corpus
  3. fondi la coppia piu' frequente in un simbolo nuovo
  4. ripeti finche' il vocabolario ha la dimensione voluta

  Nient'altro. Non c'e' linguistica: solo conteggi di frequenza.""")

CORPUS = """il gatto dorme sul tappeto
il cane dorme sul divano
il gatto mangia il pesce
il cane mangia la carne
la gatta dorme sul tappeto
il topo mangia il formaggio
il gatto insegue il topo
il cane insegue il gatto"""


# Simboli = SEQUENZE DI BYTE. Il vocabolario di partenza sono i 256 byte,
# quindi qualunque testo e' rappresentabile per costruzione: non e' una
# proprieta' simulata, e' una conseguenza di lavorare sui byte.
VOCAB_BASE = {bytes([b]) for b in range(256)}
FINE = b"</w>"


def mostra(s):
    """Rende leggibile una sequenza di byte, con i non-ASCII in esadecimale."""
    if s == FINE:
        return "_"
    try:
        return s.decode("utf-8")
    except UnicodeDecodeError:
        return "<" + s.hex() + ">"


def conta_coppie(parole):
    c = Counter()
    for simboli, freq in parole.items():
        for i in range(len(simboli) - 1):
            c[(simboli[i], simboli[i + 1])] += freq
    return c


def fondi(parole, coppia):
    nuove, (a, b) = {}, coppia
    for simboli, freq in parole.items():
        out, i = [], 0
        while i < len(simboli):
            if i < len(simboli) - 1 and simboli[i] == a and simboli[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(simboli[i])
                i += 1
        nuove[tuple(out)] = freq
    return nuove


def addestra_bpe(testo, n_fusioni):
    freq = Counter(testo.split())
    # ogni parola -> tupla di SINGOLI BYTE, piu' il marcatore di fine parola
    parole = {tuple(bytes([b]) for b in p.encode("utf-8")) + (FINE,): f
              for p, f in freq.items()}
    fusioni, vocab = [], set(VOCAB_BASE) | {FINE}
    for _ in range(n_fusioni):
        c = conta_coppie(parole)
        if not c:
            break
        migliore, n = c.most_common(1)[0]
        if n < 2:
            break
        fusioni.append((migliore, n))
        vocab.add(migliore[0] + migliore[1])
        parole = fondi(parole, migliore)
    return fusioni, vocab


fusioni, VOCAB = addestra_bpe(CORPUS, 14)
print(f"\n  Corpus: {len(CORPUS.split())} parole, {len(set(CORPUS.split()))} uniche")
print(f"  Vocabolario: 256 byte + '{mostra(FINE)}' + {len(fusioni)} fusioni"
      f" = {len(VOCAB)} simboli")
print(f"\n  {'#':>3}  {'coppia fusa':<20} {'occorrenze':>10}  simbolo nato")
for i, ((a, b), n) in enumerate(fusioni, 1):
    print(f"  {i:>3}  {mostra(a) + ' + ' + mostra(b):<20} {n:>10}  {mostra(a + b)!r}")

print("\n  Le prime fusioni sono le sequenze piu' frequenti del corpus.")
print("  Un tokenizer addestrato su testo italiano fondera' cose diverse da")
print("  uno addestrato su codice: e' per questo che i tokenizer NON sono")
print("  intercambiabili fra modelli.")


# ----------------------------------------------------------------------
sez("2. Cosa succede a una parola mai vista")

def segmenta(parola, fusioni):
    simboli = tuple(bytes([b]) for b in parola.encode("utf-8")) + (FINE,)
    for (a, b), _ in fusioni:
        out, i = [], 0
        while i < len(simboli):
            if i < len(simboli) - 1 and simboli[i] == a and simboli[i + 1] == b:
                out.append(a + b); i += 2
            else:
                out.append(simboli[i]); i += 1
        simboli = tuple(out)
    return simboli


print(f"  {'testo':<16} {'segmentazione':<46} {'token':>6}")
for p in ["gatto", "cane", "gattopardo", "perché", "日本語", "🐈"]:
    s = segmenta(p, fusioni)
    print(f"  {p:<16} {' | '.join(mostra(x) for x in s):<46} {len(s):>6}")

fuori = [x for p in ["perché", "日本語", "🐈"] for x in segmenta(p, fusioni)
         if x not in VOCAB]
print(f"\n  Simboli prodotti che NON stanno nel vocabolario: {len(fuori)}")
print("  Zero, e non per fortuna: ogni parola si scompone al peggio nei suoi")
print("  BYTE, e tutti i 256 byte sono nel vocabolario per costruzione.")
print("  Nessun testo puo' essere 'fuori vocabolario'. E' la proprieta'")
print("  centrale del BPE byte-level - copertura totale, a costo di piu' token.")

print("\n  Guarda il costo del non-ASCII: un carattere accentato occupa 2 byte")
print("  in UTF-8, un ideogramma 3, un'emoji 4. Se nessuna fusione li copre,")
print("  paghi un token PER BYTE. Un dominio lontano dal corpus del tokenizer")
print("  consuma molto piu' contesto a parita' di testo leggibile.")


# ----------------------------------------------------------------------
sez("3. Il confine di parola decide il costo")

print("  Stessi caratteri iniziali, confini diversi:\n")
print(f"  {'testo':<16} {'token':>6}  segmentazione")
for w in ["gatto", "gattone", "gattopardo"]:
    seg = segmenta(w, fusioni)
    print(f"  {w:<16} {len(seg):>6}  {' | '.join(mostra(x) for x in seg)}")

print("""
  'gatto' costa 1 token perche' e' stato fuso durante l'addestramento del
  tokenizer. 'gattopardo' ne costa 8: la fusione 'gatto</w>' richiede il
  marcatore di FINE PAROLA, che qui non c'e', quindi non si applica e la
  parola si sbriciola.

  Il punto generalizzabile: le fusioni imparate valgono in un CONTESTO
  preciso. Cambia il contesto - un carattere in piu' attaccato, uno spazio
  in meno, una virgola - e la segmentazione puo' cambiare completamente,
  anche se il testo a occhio e' quasi identico.""")

print("""
  Nei tokenizer reali lo spazio iniziale fa parte del token: 'gatto' e
  ' gatto' sono DUE token diversi. Qui non lo vedi perche' segmento parola
  per parola, gia' separate - ma e' la stessa classe di fenomeno, e puoi
  verificarlo in dieci secondi sul tokenizer del tuo modello:

      t = tokenizer
      print(t.encode("gatto"), t.encode(" gatto"))
      print(t.encode("Ciao"), t.encode("Ciao "))

  Se gli id differiscono, e differiranno, allora uno spazio in piu' o in
  meno nel template produce una sequenza diversa da quella vista in
  addestramento. Il testo sembra identico. Gli id no. -> sezione 5""")


# ----------------------------------------------------------------------
sez("4. Il chat template: da messaggi a una stringa")

MESSAGGI = [
    {"role": "system", "content": "Sei conciso."},
    {"role": "user", "content": "Ciao"},
    {"role": "assistant", "content": "Salve."},
]

TEMPLATE_A = ("<|im_start|>{role}\n{content}<|im_end|>\n", "<|im_start|>assistant\n")
TEMPLATE_B = ("<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>", "")
TEMPLATE_C = ("### {role}:\n{content}\n", "### assistant:\n")


def rendi(messaggi, template, prompt_generazione=False):
    corpo, coda = template
    s = "".join(corpo.format(role=m["role"], content=m["content"]) for m in messaggi)
    return s + (coda if prompt_generazione else "")


for nome, t in [("A (stile im_start)", TEMPLATE_A),
                ("B (stile header_id)", TEMPLATE_B),
                ("C (markdown)", TEMPLATE_C)]:
    r = rendi(MESSAGGI, t)
    print(f"  {nome}: {len(r)} caratteri")
    print("    " + repr(r))

print("\n  Stessi messaggi, tre stringhe completamente diverse.")
print("  Il template NON e' formattazione: e' la sintassi che il modello ha")
print("  imparato a riconoscere. Usarne uno diverso da quello dell'addestramento")
print("  significa parlargli in una lingua che non ha mai visto.")


# ----------------------------------------------------------------------
sez("5. Il danno, misurato")

class Tok:
    """Tokenizer giocattolo: token speciali interi, resto per byte."""
    def __init__(self, speciali):
        self.speciali = speciali
        self.vocab = {s: 1000 + i for i, s in enumerate(speciali)}

    def encode(self, testo):
        ids, i = [], 0
        while i < len(testo):
            for s in self.speciali:
                if testo.startswith(s, i):
                    ids.append(self.vocab[s]); i += len(s); break
            else:
                ids.append(ord(testo[i])); i += 1
        return ids


tok = Tok(["<|im_start|>", "<|im_end|>", "<|start_header_id|>",
           "<|end_header_id|>", "<|eot_id|>"])

ids_a = tok.encode(rendi(MESSAGGI, TEMPLATE_A))
ids_b = tok.encode(rendi(MESSAGGI, TEMPLATE_B))
ids_c = tok.encode(rendi(MESSAGGI, TEMPLATE_C))

print(f"  template A -> {len(ids_a):>3} token   primi 8: {ids_a[:8]}")
print(f"  template B -> {len(ids_b):>3} token   primi 8: {ids_b[:8]}")
print(f"  template C -> {len(ids_c):>3} token   primi 8: {ids_c[:8]}")

comune = sum(1 for x, y in zip(ids_a, ids_c) if x == y)
print(f"\n  A e C condividono {comune}/{min(len(ids_a), len(ids_c))} posizioni iniziali uguali")
print(f"  A usa {sum(1 for i in ids_a if i >= 1000)} token speciali, C ne usa"
      f" {sum(1 for i in ids_c if i >= 1000)}")

print("""
  Il template C non usa NESSUN token speciale: i marcatori '###' sono
  testo normale, scomposto in caratteri. Per un modello addestrato con
  il template A questo e' testo senza struttura: i confini fra i turni,
  che nel template A sono un singolo token dedicato imparato durante
  l'addestramento, qui semplicemente non ci sono.

  E il tutto senza nessun errore: la stringa e' valida, la tokenizzazione
  riesce, l'addestramento parte e la loss scende. Il modello impara -
  imparare qualcosa di diverso da quello che volevi.""")


# ----------------------------------------------------------------------
sez("6. Loss masking: quali posizioni contano")

IGNORA = -100
prompt = rendi(MESSAGGI[:2], TEMPLATE_A, prompt_generazione=True)
risposta = "Salve.<|im_end|>\n"
ids_prompt = tok.encode(prompt)
ids_risposta = tok.encode(risposta)
ids = ids_prompt + ids_risposta

etichette_corrette = [IGNORA] * len(ids_prompt) + ids_risposta[:]
etichette_tutto = ids[:]                     # errore: addestra anche sul prompt

print(f"  sequenza totale: {len(ids)} token")
print(f"    prompt   {len(ids_prompt)} token")
print(f"    risposta {len(ids_risposta)} token\n")
for nome, et in [("solo risposta (corretto)", etichette_corrette),
                 ("tutto (sbagliato)", etichette_tutto)]:
    attive = sum(1 for e in et if e != IGNORA)
    print(f"  {nome:<26} posizioni che contribuiscono: {attive:>3}/{len(ids)}"
          f"  ({100 * attive / len(ids):.0f}%)")

print(f"\n  Nel caso corretto solo il {100 * len(ids_risposta) / len(ids):.0f}% delle posizioni"
      f" contribuisce alla loss.")
print("  Addestrando su tutto, il modello passa la maggior parte dello sforzo")
print("  a imparare a riprodurre i PROMPT, che non e' quello che gli chiedi.")

print("""
  ATTENZIONE - due maschere possono avere lo STESSO CONTEGGIO e coprire
  posizioni DIVERSE. Contare non basta: si confrontano elemento per
  elemento. Il caso peggiore e' lo sfasamento di una posizione, che
  lascia il conteggio identico e allinea ogni token al bersaglio
  sbagliato. -> M02, M21""")

print("\n  NON tutti i token di struttura vanno mascherati. Guarda cosa c'e'")
print("  dentro la risposta supervisionata qui sopra:\n")
for tid in ids_risposta:
    speciale = tid >= 1000
    nome = next((k for k, v in tok.vocab.items() if v == tid), None)
    print(f"    id {tid:>5}  {'SPECIALE ' + nome if speciale else repr(chr(tid))}")

print("""
  Il terminatore di turno e' SUPERVISIONATO, ed e' corretto che lo sia:
  e' il token con cui il modello impara a FERMARSI. Se lo maschera,
  il modello non impara mai a smettere di generare - e in inferenza
  continua a produrre testo oltre la risposta.

  La distinzione da tenere:
    - PREFISSI di struttura (l'intestazione che apre il turno assistant):
      mascherati. Sono il prompt, non la risposta.
    - TERMINATORE di fine turno / EOS: supervisionato. E' parte di cio'
      che il modello deve produrre.

  Quindi "trovo token speciali fra quelli attivi" NON e' di per se' un
  errore: dipende da QUALI. Trovare il terminatore e' giusto, trovare
  l'intestazione del turno no.""")

print("\n  Come si ispeziona davvero, su un tokenizer vero:")
print("""
    for tok_id, etichetta in zip(input_ids[0], labels[0]):
        segno = "  " if etichetta == -100 else "->"
        print(segno, repr(tokenizer.decode([tok_id])))

  Le righe con '->' sono quelle su cui il modello viene addestrato.
  Guardale. Se ci trovi il prompt, o token di struttura, hai trovato
  il problema prima di sprecare ore di GPU.""")

print()
