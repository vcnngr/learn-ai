#!/usr/bin/env python3
"""
Lab 11.1 — Dataset: maschere, packing, duplicati, contaminazione.   CPU.

    python3 corso/labs/lab_11_1_dataset.py

Ogni difetto viene COSTRUITO di proposito e poi MISURATO, cosi' sai che
segnale cercare quando lo incontri per caso.
"""

import hashlib
import re
from collections import Counter


def sez(t):
    print(f"\n{'─' * 72}\n{t}\n{'─' * 72}")


IGNORA = -100

# ----------------------------------------------------------------------
sez("1. Due maschere, stesso conteggio, posizioni diverse")

# sequenza: 6 token di prompt, 5 di risposta
ids = [10, 11, 12, 13, 14, 15,  20, 21, 22, 23, 24]
NP = 6

giusta = [IGNORA] * NP + ids[NP:]
sfasata = [IGNORA] * (NP - 1) + ids[NP - 1:-1] + [IGNORA]   # spostata di 1

print(f"  {'token':<18}{' '.join(f'{t:>5}' for t in ids)}")
for nome, m in [("maschera GIUSTA", giusta), ("maschera SFASATA", sfasata)]:
    print(f"  {nome:<18}{' '.join(('    .' if e == IGNORA else f'{e:>5}') for e in m)}")

ng = sum(1 for e in giusta if e != IGNORA)
ns = sum(1 for e in sfasata if e != IGNORA)
print(f"\n  posizioni attive: giusta {ng}, sfasata {ns}   -> uguali? {ng == ns}")
diverse = sum(1 for a, b in zip(giusta, sfasata) if a != b)
print(f"  posizioni in cui DIFFERISCONO, confronto elemento per elemento: {diverse}")
print("\n  Un controllo che conta le posizioni attive NON vede questo bug.")
print("  Un confronto elemento per elemento lo vede subito.")
print("  E' il caso peggiore: ogni token viene allineato al bersaglio sbagliato,")
print("  la loss scende comunque, e il modello impara uno spostamento.")


# ----------------------------------------------------------------------
sez("2. Packing: cosa risparmia e cosa rompe")

esempi = [[1, 2, 3], [4, 5], [6, 7, 8, 9], [10], [11, 12]]
MAXLEN = 6

# senza packing: una sequenza per esempio, riempita di padding
padded = [e + [0] * (MAXLEN - len(e)) for e in esempi]
tot_pad = sum(MAXLEN - len(e) for e in esempi)

# con packing: si concatena finche' si riempie
pacchi, corrente = [], []
for e in esempi:
    if len(corrente) + len(e) > MAXLEN:
        pacchi.append(corrente); corrente = []
    corrente += e
if corrente:
    pacchi.append(corrente)

print(f"  {len(esempi)} esempi, lunghezza massima {MAXLEN}\n")
print(f"  senza packing: {len(padded)} sequenze, {tot_pad} token di padding sprecati"
      f"  ({100 * tot_pad / (len(padded) * MAXLEN):.0f}%)")
print(f"  con packing:   {len(pacchi)} sequenze, "
      f"{sum(MAXLEN - len(p) for p in pacchi)} token sprecati")
for i, p in enumerate(pacchi):
    print(f"    pacco {i}: {p}")

print("""
  Il risparmio e' reale. Il rischio anche: dentro un pacco ci sono
  documenti DIVERSI, e senza precauzioni l'attention del secondo documento
  puo' guardare i token del primo. Il modello impara correlazioni fra
  testi che non c'entrano niente fra loro.

  La difesa si chiama attention mask per documento (o block-diagonal):
  ogni documento vede solo se stesso. Se la tua libreria non la applica,
  il packing introduce rumore invece di risparmiare.

  COME VERIFICARLO sul tuo stack: costruisci un pacco con due documenti,
  passa position_ids/attention mask come li passa il tuo codice, e
  controlla che i pesi di attention fra i due documenti siano ZERO.
  Se non lo sono, stai addestrando su correlazioni inventate.""")


# ----------------------------------------------------------------------
sez("3. Duplicati esatti e quasi-duplicati")

DATASET = [
    "Come si configura il timeout del server?",
    "Come si configura il timeout del server?",          # esatto
    "come si configura il  timeout del server?",         # differisce per caso e spazi
    "Come si configura il timeout del client?",          # quasi-duplicato semantico
    "Qual e' la porta di default di Postgres?",
    "Spiega la differenza fra TCP e UDP.",
]

def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())

esatti = Counter(DATASET)
norm_c = Counter(norm(s) for s in DATASET)
print(f"  esempi totali:                    {len(DATASET)}")
print(f"  unici per confronto ESATTO:       {len(esatti)}")
print(f"  unici dopo NORMALIZZAZIONE:       {len(norm_c)}")
print("\n  Il terzo esempio differisce solo per maiuscole e spazi doppi.")
print("  Un dedup esatto NON lo trova. Uno normalizzato si'.")

def ngrammi(s, n=3):
    p = norm(s).split()
    return {tuple(p[i:i + n]) for i in range(max(1, len(p) - n + 1))}

def jaccard(a, b):
    A, B = ngrammi(a), ngrammi(b)
    return len(A & B) / len(A | B) if A | B else 0.0

print(f"\n  Somiglianza per 3-grammi (Jaccard):\n")
print(f"  {'coppia':<10} {'sim':>6}  testi")
for i in range(len(DATASET)):
    for j in range(i + 1, len(DATASET)):
        s = jaccard(DATASET[i], DATASET[j])
        if s > 0.3:
            print(f"  {i}-{j:<8} {s:>6.2f}  {DATASET[i][:34]!r} / {DATASET[j][:34]!r}")

print("""
  La coppia 0-3 ('server' vs 'client') ha somiglianza alta ma NON e' un
  duplicato: e' un esempio diverso e utile. La soglia non e' un parametro
  universale - dipende da cosa consideri duplicato NEL TUO dominio, e va
  scelta guardando i casi al confine, non copiata da un tutorial.""")


# ----------------------------------------------------------------------
sez("4. Contaminazione: misurata, non assunta")

TRAIN = [
    "Come si configura il timeout del server?",
    "Spiega la differenza fra TCP e UDP.",
    "Qual e' la porta di default di Postgres?",
    "Come si monta un volume NFS?",
]
EVAL = [
    "Spiega la differenza fra TCP e UDP.",             # ESATTO in train
    "come si configura il  timeout del server?",       # stesso testo, spazi/maiuscole diversi
    "Come si abilita il logging su syslog?",           # pulito
]

def h(s):
    return hashlib.sha256(norm(s).encode()).hexdigest()[:12]

train_h = {h(t) for t in TRAIN}
print(f"  {'esempio di eval':<48} {'esatto':>7} {'max sim':>8}  verdetto")
for e in EVAL:
    esatto = h(e) in train_h
    ms = max(jaccard(e, t) for t in TRAIN)
    if esatto:
        v = "CONTAMINATO (match normalizzato)"
    elif ms > 0.6:
        v = f"SOSPETTO (sim {ms:.2f})"
    else:
        v = "pulito"
    print(f"  {e[:48]:<48} {str(esatto):>7} {ms:>8.2f}  {v}")

print("""
  Nota cosa serve per dare questi verdetti: il corpus di TRAIN, una
  definizione di 'match' (qui: normalizzato), e una soglia. Senza quei
  tre ingredienti la contaminazione non e' rilevabile - non e' una
  proprieta' che un file possiede, e' una RELAZIONE fra due insiemi.

  Un validator puo' dichiarare cosa cerca. Non puo' garantire che non
  esista contaminazione di tipo diverso da quello che sa cercare
  (parafrasi, traduzione, stesso problema con numeri diversi).""")


# ----------------------------------------------------------------------
sez("5. Un validator spec-driven")

SPEC = {
    "template_atteso": "<|im_start|>",
    "lunghezza_max": 12,
    "campi_richiesti": ("prompt", "risposta"),
}

RECORD = [
    {"prompt": "<|im_start|>user\nA<|im_end|>", "risposta": "B<|im_end|>"},
    {"prompt": "### user:\nA", "risposta": "B"},                      # template errato
    {"prompt": "<|im_start|>user\nA<|im_end|>", "risposta": ""},       # risposta vuota
    {"prompt": "<|im_start|>user\n" + "x" * 50, "risposta": "B<|im_end|>"},  # troppo lungo
    {"prompt": "<|im_start|>user\nA<|im_end|>"},                       # campo mancante
]

def valida(r, spec):
    problemi = []
    for c in spec["campi_richiesti"]:
        if c not in r:
            problemi.append(f"campo mancante: {c}")
    if "prompt" in r and spec["template_atteso"] not in r["prompt"]:
        problemi.append("template diverso da quello atteso")
    if "risposta" in r and not r["risposta"].strip():
        problemi.append("risposta vuota")
    n = sum(len(str(v)) for v in r.values())
    if n > spec["lunghezza_max"] * 6:
        problemi.append(f"probabile troncamento ({n} caratteri)")
    return problemi

print(f"  {'#':>3}  {'esito':<10} problemi rilevati")
for i, r in enumerate(RECORD):
    p = valida(r, SPEC)
    print(f"  {i:>3}  {'OK' if not p else 'SCARTATO':<10} {'; '.join(p) if p else '-'}")

print("""
  CLASSI CHE QUESTO VALIDATOR RILEVA (dichiarate, non implicite):
    - campi mancanti
    - template diverso da quello in spec
    - risposte vuote
    - lunghezze anomale

  CLASSI CHE NON RILEVA, e va detto:
    - risposte plausibili ma sbagliate nel merito
    - contaminazione (serve il corpus di riferimento -> sezione 4)
    - maschere errate (serve il tokenizer -> sezione 1)
    - sbilanciamento fra categorie
    - parafrasi e traduzioni di esempi gia' presenti

  Un validator vale quanto la sua spec. 'Il validator e' passato' significa
  'nessuno dei controlli che ho scritto ha trovato qualcosa', non
  'il dataset e' buono'.""")


# ----------------------------------------------------------------------
sez("6. Quantita' contro qualita': cosa si puo' dire davvero")

print("""  Non ti do una regola del tipo 'servono N esempi': dipende dal compito,
  dal modello base, da quanto il compito e' lontano da cio' che gia' sa.
  Chiunque ti dia un numero senza quel contesto sta indovinando.

  Quello che PUOI fare, ed e' l'unico modo onesto: misurare la curva.

    1. Prendi il dataset completo. Tienine da parte una fetta per l'eval.
    2. Addestra con il 25%, 50%, 100% degli esempi rimasti.
    3. Valuta ogni run sullo STESSO insieme tenuto da parte.
    4. Guarda l'andamento.

  Se la metrica e' ancora in salita al 100%, altri dati aiuteranno.
  Se e' piatta fra 50% e 100%, il problema non e' la quantita': e' la
  qualita', il metodo, o il compito stesso.

  Costa tre run invece di uno, e ti dice se il quarto ha senso.
  Gli strumenti statistici per dire se una differenza e' reale sono
  in M12: senza quelli, tre punti rumorosi non sono una curva.""")

print()
