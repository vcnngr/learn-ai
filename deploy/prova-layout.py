#!/usr/bin/env python3
"""
Gli invarianti di impaginazione reggono a TUTTI i breakpoint?

    python3 deploy/prova-layout.py [corso/assets/style.css]

Nasce da una regressione che nessun altro controllo poteva vedere: il
comando del tema era ancorato al piede di una sidebar alta 100vh che
scorre internamente — corretto sopra i 980px, dove pero' quella
struttura non esiste piu'. Sotto, la sidebar torna un blocco a flusso e
il comando finiva dopo l'intero indice.

Un invariante che vale a un solo breakpoint non e' un invariante.

TRE VERSIONI PER ARRIVARCI, e le prime due erano gate finti:

  1. `grep` dentro la regione: si faceva ingannare DAL PROPRIO COMMENTO,
     perche' la riga che spiega «il piede passa in TESTA (order: -1)»
     contiene la stessa stringa della regola.
  2. commenti tolti, ma la ricerca restava «la stringa compare da
     qualche parte in questa regione»: una regola messa sul SELETTORE
     SBAGLIATO dentro lo stesso media query passava lo stesso.
  3. questa: si estraggono le dichiarazioni DI QUEL SELETTORE in QUEL
     contesto, e si controlla li'.

Le prime due le ha trovate la meta-prova e una review, non io.
"""

import pathlib
import re
import sys

CSS = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "corso/assets/style.css")
sorgente = re.sub(r"/\*.*?\*/", "", CSS.read_text(), flags=re.S)   # via i commenti


def corpo(testo, apertura):
    """Il corpo di un blocco, bilanciando le graffe.

    Serve il bilanciamento e non un `find('}')`: un @media contiene
    regole annidate, e fermarsi alla prima graffa chiusa darebbe un
    frammento che sembra il blocco e non lo e'.
    """
    i = testo.find(apertura)
    if i < 0:
        return None
    i = testo.find("{", i)
    if i < 0:
        return None
    livello, j = 0, i
    while j < len(testo):
        if testo[j] == "{":
            livello += 1
        elif testo[j] == "}":
            livello -= 1
            if livello == 0:
                return testo[i + 1:j]
        j += 1
    return None


def dichiarazioni(testo, selettore):
    """Le dichiarazioni della regola con QUEL selettore, in quel testo.

    Il confronto e' sul selettore esatto: `.sidebar` non deve combaciare
    con `.sidebar-inner`, altrimenti il controllo verifica una regola
    diversa da quella che crede.
    """
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", testo):
        sels = [s.strip() for s in m.group(1).split(",")]
        if selettore in sels:
            return m.group(2)
    return None


DESKTOP = re.sub(r"@media[^{]*\{(?:[^{}]|\{[^{}]*\})*\}", "", sorgente, flags=re.S)
MOBILE = corpo(sorgente, "@media (max-width: 980px)")

CONTROLLI = [
    ("sopra i 980px — colonna con indice che scorre e piede ancorato", DESKTOP, [
        (".sidebar",        "flex-direction: column", "la sidebar e' una colonna flex"),
        (".sidebar-inner",  "overflow-y: auto",       "scorre l'indice, non la colonna"),
        (".sidebar-piede",  "flex: none",             "il piede non si comprime"),
    ]),
    ("sotto i 980px — blocco a flusso, il piede passa in testa", MOBILE, [
        (".sidebar-piede",  "order: -1",              "il piede va in testa"),
        (".sidebar-inner",  "overflow: visible",      "l'indice smette di scorrere da solo"),
        (".sidebar",        "position: static",       "la colonna torna a flusso"),
    ]),
]

print("Invarianti di impaginazione della sidebar.\n")
falliti = 0
for titolo, contesto, righe in CONTROLLI:
    print(f"  {titolo}")
    if contesto is None:
        print("    contesto assente nel CSS")
        falliti += len(righe)
        print()
        continue
    for selettore, regola, nome in righe:
        d = dichiarazioni(contesto, selettore)
        if d is None:
            print(f"    {nome:<48} SELETTORE ASSENTE: {selettore}")
            falliti += 1
        elif regola in d:
            print(f"    {nome:<48} ok")
        else:
            print(f"    {nome:<48} MANCA su {selettore}: {regola}")
            falliti += 1
    print()

if falliti:
    print(f"  {falliti} invarianti violati.")
    print("  Un comando ancorato a una struttura che esiste a un solo")
    print("  breakpoint torna irraggiungibile a tutti gli altri.")
    sys.exit(1)

print("  Il comando resta raggiungibile senza scorrere, a entrambi i breakpoint.")
