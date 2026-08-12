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
    """Proprieta' -> valore per QUEL selettore, con la cascata rispettata
    per quanto un controllo semplice possa farlo — e con un rifiuto
    esplicito quando non puo'.

    COSA GESTISCE
      - ULTIMA, non prima: il selettore puo' comparire piu' volte e in
        CSS vince l'ultima. Fermarsi alla prima fa passare un valore che
        una regola successiva sovrascrive.
      - !important: batte le dichiarazioni normali qualunque sia
        l'ordine, quindi va tracciato o il modello dice il contrario del
        browser.

    COSA NON GESTISCE, E LO DICE
      La SPECIFICITA'. `.sidebar .sidebar-piede` vince su
      `.sidebar-piede` a parita' d'ordine, e valutarlo davvero vuol dire
      scrivere un motore di cascata — che sarebbe una nuova fonte di
      difetti, non un controllo.
      Quindi: se esiste QUALUNQUE altra regola il cui selettore nomina
      questo elemento senza essere il selettore esatto, il gate non
      dichiara "ok": dichiara NON DETERMINABILE e fallisce. Un
      "non lo so" rumoroso vale piu' di un verde che non significa
      niente.
    """
    props, sospetti = {}, []
    token = re.escape(selettore)
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", testo):
        sels = [s.strip() for s in m.group(1).split(",")]
        esatto = selettore in sels
        if not esatto:
            # Conta solo se il token e' il SOGGETTO del selettore, cioe'
            # l'ultimo compound: `.wrap .sidebar` colpisce lo stesso
            # elemento con piu' specificita' ed e' un override vero,
            # mentre `.sidebar h4` colpisce un DISCENDENTE e non
            # sovrascrive niente di `.sidebar`. Senza questa distinzione
            # il gate dichiarava non determinabile mezzo foglio di stile.
            for s in sels:
                soggetto = re.split(r"[\s>+~]+", s.strip())[-1]
                if re.search(token + r"(?![\w-])", soggetto):
                    sospetti.append(s.strip())
            continue
        for d in m.group(2).split(";"):
            if ":" not in d:
                continue
            k, _, v = d.partition(":")
            # I NOMI DELLE PROPRIETA' SONO CASE-INSENSITIVE in CSS:
            # `FLEX: 1` e' identico a `flex: 1`. Tenendo il caso
            # originale finivano in due chiavi distinte, e una
            # `FLEX: 1` scritta dopo `flex: none` non sovrascriveva
            # niente nel modello mentre nel browser vinceva: falso
            # verde. Si normalizza qui, una volta.
            k = k.strip().lower()
            v = " ".join(v.split())
            # `!important` non si cerca come stringa letterale: la
            # sintassi ammette spazio fra ! e important ed e'
            # case-insensitive, quindi `! important` e `!IMPORTANT`
            # sono validi e sfuggivano al confronto letterale — tre
            # falsi verdi su tre varianti provate.
            imp = re.search(r"!\s*important\b", v, re.I) is not None
            v = re.sub(r"!\s*important\b", "", v, flags=re.I).strip()
            # una dichiarazione !important non si lascia sovrascrivere
            # da una normale, anche se questa viene dopo
            if props.get(k, (None, False))[1] and not imp:
                continue
            props[k] = (v, imp)
    if not props and not sospetti:
        return None, []
    return {k: v for k, (v, _) in props.items()}, sospetti


def attesa(regola):
    k, _, v = regola.partition(":")
    return k.strip().lower(), " ".join(v.split())


def uguali(atteso, trovato):
    """Stesso valore, ignorando il caso.

    Anche le parole chiave sono case-insensitive: `COLUMN` e' un
    flex-direction valido quanto `column`, e il confronto esatto lo
    dava per sbagliato — falso rosso, meno pericoloso del falso verde
    ma sempre una risposta errata.

    LIMITE DICHIARATO: non tutti i valori CSS sono case-insensitive
    (nomi di font, url, identificatori personalizzati non lo sono).
    Qui vale perche' ogni invariante di questo file ha per valore una
    parola chiave o un numero. Se un invariante futuro guardera' un
    nome di font, questo confronto andra' ristretto.
    """
    return atteso.lower() == trovato.lower()


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
        d, sospetti = dichiarazioni(contesto, selettore)
        prop, valore = attesa(regola)
        if sospetti:
            print(f"    {nome:<48} NON DETERMINABILE: altre regole "
                  f"toccano {selettore} -> {sospetti[0]}")
            falliti += 1
            continue
        if d is None:
            print(f"    {nome:<48} SELETTORE ASSENTE: {selettore}")
            falliti += 1
        elif prop in d and uguali(valore, d[prop]):
            print(f"    {nome:<48} ok")
        elif prop in d:
            print(f"    {nome:<48} MANCA su {selettore}: {prop} vale "
                  f"{d[prop]!r}, atteso {valore!r}")
            falliti += 1
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
