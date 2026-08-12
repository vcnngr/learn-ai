#!/usr/bin/env python3
"""
Gli invarianti di impaginazione reggono a TUTTI i breakpoint?

    python3 deploy/prova-layout.py [corso/assets/style.css]

Nasce da una regressione che nessun altro controllo poteva vedere: il
comando del tema era ancorato al piede di una sidebar alta 100vh che
scorre internamente — corretto sopra i 980px, dove pero' quella
struttura non esiste piu'. Sotto, la sidebar torna un blocco a flusso e
il comando finiva dopo l'intero indice: il difetto che la
ristrutturazione aveva appena chiuso.

Un invariante che vale a un solo breakpoint non e' un invariante.

E' in Python e non in sh per una ragione trovata dalla meta-prova: la
versione precedente cercava le regole con `grep` dentro il blocco, e si
faceva ingannare DAL PROPRIO COMMENTO — la riga che spiega
«il piede passa in TESTA (order: -1)» contiene la stessa stringa della
regola, quindi togliendo la regola il gate la trovava lo stesso.
Ancorare a inizio riga rompeva le regole scritte a meta' riga
(`display: flex; flex-direction: column;`). La risposta giusta e'
togliere i commenti prima di guardare, e qui si fa in due righe.
"""

import pathlib
import re
import sys

CSS = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "corso/assets/style.css")
sorgente = CSS.read_text()

# via i commenti: sono documentazione, non regole
senza_commenti = re.sub(r"/\*.*?\*/", "", sorgente, flags=re.S)


def blocco(apertura):
    """Il testo dalla riga che apre il blocco fino alla sua chiusura."""
    i = senza_commenti.find(apertura)
    if i < 0:
        return ""
    return senza_commenti[i:senza_commenti.find("\n}", i) + 2]


CONTROLLI = [
    ("sopra i 980px — colonna con indice che scorre e piede ancorato", [
        ("la sidebar e' una colonna flex",       ".sidebar {",   "flex-direction: column"),
        ("scorre l'indice, non la colonna",      ".sidebar-inner {", "overflow-y: auto"),
        ("il piede non si comprime",             ".sidebar-piede {", "flex: none"),
    ]),
    ("sotto i 980px — blocco a flusso, il piede passa in testa", [
        ("il piede va in testa",                 "@media (max-width: 980px)", "order: -1"),
        ("l'indice smette di scorrere da solo",  "@media (max-width: 980px)", "overflow: visible"),
    ]),
]

print("Invarianti di impaginazione della sidebar.\n")
falliti = 0
for titolo, righe in CONTROLLI:
    print(f"  {titolo}")
    for nome, apertura, regola in righe:
        b = blocco(apertura)
        if regola in b:
            print(f"    {nome:<48} ok")
        else:
            print(f"    {nome:<48} MANCA: {regola}")
            falliti += 1
    print()

if falliti:
    print(f"  {falliti} invarianti violati.")
    print("  Un comando ancorato a una struttura che esiste a un solo")
    print("  breakpoint torna irraggiungibile a tutti gli altri.")
    sys.exit(1)

print("  Il comando resta raggiungibile senza scorrere, a entrambi i breakpoint.")
