#!/usr/bin/env python3
"""
verifica.py — controlla che i numeri scritti nelle pagine vengano davvero
dai lab.

    python3 corso/verifica.py             # rapporto completo
    python3 corso/verifica.py --brevi     # solo il conteggio per pagina
    python3 corso/verifica.py --no-cache  # riesegue tutti i lab
    python3 corso/verifica.py --copertura # solo il rapporto di marcatura

Il corso dichiara: "nessuna affermazione empirica senza la misura accanto".
Questo script verifica che la dichiarazione sia vera, invece di fidarsi.

Esce con codice 1 se trova qualcosa. E' pensato per la CHECKLIST DI RILASCIO,
non per il corso: va lanciato prima di pubblicare una modifica ai lab, ai
dati o all'ambiente.

──────────────────────────────────────────────────────────────────────────
DUE CONTROLLI

  1. BLOCCHI  ogni numero dentro un <pre><code> deve comparire nell'output
              del lab che quel blocco dichiara, via data-lab="lab_09_1".
              I blocchi sono output COPIATO: il confronto e' stretto.

  2. PROSA    ogni numero "distintivo" citato nel testo deve comparire in
              almeno un output di lab. Piu' permissivo: la prosa arrotonda,
              riformula e cita valori di altri moduli.

──────────────────────────────────────────────────────────────────────────
LA MARCATURA data-lab, E PERCHE' SERVE

    <pre data-lab="lab_13_1"><code>...</code></pre>
    <pre data-lab="nessuno"><code>...</code></pre>   <- esempio didattico

Senza marcatura un numero si cerca nell'UNIONE dei 24 output, e basta che
combaci con qualunque cosa in qualunque lab. E' una rete cosi' larga da
essere quasi inutile: la prima versione di questo script dichiarava zero
problemi mentre M13 pubblicava una perplessita' di 676,06 contro le 694,15
prodotte dal lab — il valore sbagliato combaciava con un numero di un altro
modulo. Con data-lab quel blocco si confronta solo con lab_13_1, e il
disallineamento salta fuori.

I blocchi non marcati NON vengono ignorati: finiscono nel rapporto di
copertura, che dice quanti sono e dove. La marcatura si puo' completare a
poco a poco, ma finche' non e' completa il controllo resta piu' debole di
quanto sembri, e il rapporto lo dice invece di lasciarlo credere.

──────────────────────────────────────────────────────────────────────────
LE TOLLERANZE, DICHIARATE

Un confronto per uguaglianza di stringa non regge sui float: 2.384e-07 su
un backend puo' essere 1.192e-07 su un altro, e nessuno dei due e' "il"
valore giusto. Ma una tolleranza implicita e' peggio di nessuna tolleranza,
perche' nasconde quanto sta lasciando passare. Qui sono tre, esplicite:

  ESATTO       interi e valori deterministici: devono coincidere.
  ARROTONDA    la pagina puo' arrotondare: 0,55 in prosa vale se il lab
               stampa 0.547157. Confronto alla precisione SCRITTA IN PAGINA
               (meno cifre scrivi, piu' larga e' la maglia: e' voluto).
  REL_FP       scarti numerici dipendenti dal backend (max|diff|, epsilon,
               zero macchina): confronto sull'ORDINE DI GRANDEZZA, con la
               tolleranza relativa qui sotto. Vale solo per i valori sotto
               SOGLIA_FP, cioe' quelli che sono rumore per costruzione.

Un valore che passa solo grazie a REL_FP non e' "verificato": e' "compatibile
con l'ambiente dichiarato". La differenza va scritta accanto al numero in
pagina, non lasciata al lettore.
"""

import html
import pathlib
import re
import subprocess
import sys

BREVI = "--brevi" in sys.argv
SOLO_COPERTURA = "--copertura" in sys.argv
USA_CACHE = "--no-cache" not in sys.argv

QUI = pathlib.Path(__file__).resolve().parent
CACHE_DIR = QUI / ".verifica-cache"

# --- tolleranze dichiarate -------------------------------------------
REL_FP = 0.51        # ±51%: un fattore 2 in piu' o in meno NON passa,
                     # ma 1.192e-07 vs 2.384e-07 (esattamente 2x) e' il
                     # caso classico di scarto backend-dipendente e va
                     # gestito scrivendo l'ordine di grandezza in pagina,
                     # non allargando questa soglia.
SOGLIA_FP = 1e-4     # sotto questo valore assoluto un numero e' trattato
                     # come rumore numerico, non come misura.


# ----------------------------------------------------------------------
# 1. esegue tutti i lab e raccoglie l'output

def esegui_lab():
    """Esegue i lab, con cache su disco. La cache viene invalidata quando
    il file .py cambia: e' esattamente il caso in cui le pagine possono
    essere rimaste indietro."""
    CACHE_DIR.mkdir(exist_ok=True)
    out = {}
    for f in sorted((QUI / "labs").glob("*.py")):
        c = CACHE_DIR / (f.stem + ".txt")
        if USA_CACHE and c.exists() and c.stat().st_mtime > f.stat().st_mtime:
            out[f.stem] = c.read_text()
            print(f"  {f.name:<32} da cache")
            continue
        print(f"  {f.name:<32} eseguo ...", end="", flush=True)
        try:
            r = subprocess.run([sys.executable, str(f)], capture_output=True,
                               text=True, timeout=900, cwd=QUI)
            out[f.stem] = r.stdout + r.stderr
            c.write_text(out[f.stem])
            print(f" ok ({len(out[f.stem]):,} char)")
        except subprocess.TimeoutExpired:
            out[f.stem] = ""
            print(" TIMEOUT")
    return out


# ----------------------------------------------------------------------
# 2. normalizzazione e ricerca

def norm(s):
    """Collassa spazi, toglie entita' HTML, uniforma i segni meno."""
    s = html.unescape(s)
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", s).strip()


def norm_num(s, italiano=True):
    """Forma confrontabile di un numero.

    ATTENZIONE alla lingua: le pagine sono in italiano (virgola = decimale,
    punto = migliaia), l'output dei lab e' in inglese (il contrario).
    Leggere '1,074' di una pagina come 1074 invece che 1,074 e' esattamente
    il tipo di errore che questo script esiste per trovare.
    """
    s = s.strip().replace("−", "-")
    segno = ""
    if s[:1] in "+-":
        segno, s = ("-" if s[0] == "-" else ""), s[1:]
    if italiano:
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", s):        # 1.234.567
            return segno + s.replace(".", "")
        return segno + s.replace(",", ".")
    if re.fullmatch(r"\d{1,3}(,\d{3})+", s):             # 1,234,567 e 4,096
        return segno + s.replace(",", "")
    return segno + s


def val(s, italiano=True):
    """Valore numerico, o None se non convertibile."""
    try:
        return float(norm_num(s, italiano))
    except ValueError:
        return None


def cifre_decimali(s, italiano=True):
    m = re.search(r"\.(\d+)", norm_num(s, italiano))
    return len(m.group(1)) if m else 0


def confronta(testo_num, valori, italiano=True):
    """Cerca il numero fra i valori. Restituisce (regola, valore_trovato)
    oppure (None, piu_vicino) se non passa nessuna delle tre regole."""
    v = val(testo_num, italiano)
    if v is None:
        return None, None
    d = cifre_decimali(testo_num, italiano)

    for lv in valori:                                    # ESATTO
        if lv == v:
            return "ESATTO", lv
    # ARROTONDA. Vale anche con d = 0: una pagina che scrive '694' sta
    # arrotondando 694.15, ed e' corretto. La prima versione saltava questo
    # caso, e lasciava passare solo i numeri con la virgola - cioe' proprio
    # non quelli interi, dove stavano gli errori di M13.
    for lv in valori:
        if round(lv, d) == round(v, d):
            return "ARROTONDA", lv
    if abs(v) < SOGLIA_FP:                               # REL_FP
        for lv in valori:
            if lv != 0 and abs(lv) < SOGLIA_FP and abs(lv - v) <= REL_FP * abs(v):
                return "REL_FP", lv
    # nessuna regola: restituisco il piu' vicino, per il rapporto
    vicino = min(valori, key=lambda x: abs(x - v)) if valori else None
    return None, vicino


# ----------------------------------------------------------------------
# 3. estrazione dei numeri

# CATTURA IL SEGNO. Senza, '-0,073' viene cercato come '+0,073': un valore
# che nell'output del lab non esiste, e che passa solo perche' il suo
# opposto compare da qualche altra parte. Era il secondo buco della prima
# versione, ed e' il piu' insidioso perche' non produce nessun rumore.
SEGNO = r"[-+−]?"

# Numeri nei BLOCCHI: tutti, anche interi e a una cifra decimale. I blocchi
# sono output copiato, quindi ogni numero deve tornare.
RE_BLOCCO = re.compile(rf"(?<![\w.]){SEGNO}\d+(?:[.,]\d+)*(?:e[+-]?\d+)?(?![\w])", re.I)

# Stringhe di versione: tre o piu' gruppi separati da punto. Non sono
# quantita' e non vanno confrontate con niente.
RE_VERSIONE = re.compile(r"\d+(?:\.\d+){2,}")

# Numeri nella PROSA: piu' selettivi, ma la soglia della prima versione
# (>=3 cifre significative) era troppo alta. '676', '4,6', '321' e '96'
# le stavano tutti sotto - ed erano i quattro numeri sbagliati di M13.
RE_PROSA = re.compile(rf"""
    (?<![\w.])
    ( {SEGNO} (?:
        \d+[.,]\d+(?:e[+-]?\d+)?        |   # 0.0032  61,983  4,6
        \d+(?:[.,]\d+)?e[+-]?\d+        |   # 2.384e-07
        \d{{1,3}}(?:[.,]\d{{3}})+       |   # 87.506  1,234,567
        \d{{3,}}                            # 676  321  1500
    ))
    (?![\w])
""", re.X | re.I)

# Numeri che NON vengono da un lab, e non devono venirci. Ogni voce qui
# e' una rinuncia: va motivata, non aggiunta per far tacere il rapporto.
IGNORA = {
    # versioni di libreria e di strumenti
    "2.2.2", "1.26.4", "3.12", "2.9.1", "0.145.0", "1.0.3",
    # ore del corso e delle proposte: contabilita', non misure
    "144.5", "178.5", "153.5", "150.5", "145.5", "154.0", "156.0",
    "160.0", "161.0", "149.0", "200.0", "75.0", "16.5", "34.0",
    # byte per parametro: calcolati in prosa, mai stampati da un lab
    "0.531", "0.53125", "0.406", "1.031",
    # costanti e valori didattici calcolati a mano nel testo
    "1.074",      # M00: GiB = 1,074 x 10^9 byte
    "3.001",      # M02: l'esempio della derivata, 3,001 x 3,001
    "6.001",      # M02: il risultato di quel calcolo
    "0.64",       # M06: risposta di una verifica, calcolata in prosa
    "27.00004",   # M04: dimensione ipotetica in una domanda
    "0.045",      # M06 cita la configurazione q,v di M14 (lab_14_1)
    "16.23",      # M06/M14: QLoRA ricalcolato in prosa con 0,53125
    "0.491",      # frazione LoRA del caso: la stampa lab_14_1, non lab_06_1
    "132644864",  # idem: parametri addestrabili del caso di studio
    "0.00491",    # idem, in forma frazionaria
    "132.6",      # idem, in milioni (M08)
    "1.06",       # idem x 8 byte: aritmetica di M08, arrotondata a 1,1 in M06

    # --- aritmetica svolta IN PROSA, a partire da numeri dei lab ---
    # Non sono misure: sono passaggi di calcolo mostrati al lettore.
    # Toglierli dal rapporto e' legittimo solo perche' il conto e' scritto
    # accanto e verificabile a mano.
    "67108864", "1572864", "6291456",   # M01: elementi x byte, conti espliciti
    "6.3",                              # M01: MB del conto qui sopra
    "304", "180",                       # M02: 1,1^60 e i MB di un gradiente
    "378", "16.2", "138", "17.1",       # M06: differenze e somme della tabella
    "6620",                             # M08: picco arrotondato (lab: 6620.82)
    "10.7",                             # M14: degrado in %, arrotondato
    "220",                              # M17: 219,7 GB arrotondato
    "3000",                             # M20: 3241x arrotondato, e dichiarato
    "0.028",                            # M16: |−0,028| usato come "spazio
                                        # disponibile verso l'alto"
    "118",                              # M16: −118,501 citato senza decimali

    # --- numeri di esercizi INVENTATI, non prodotti da nessun lab ---
    # Sono domande di verifica: il valore serve alla domanda, non misura nulla.
    "2047",       # M09: posizione oltre la finestra addestrata
    "340",        # M10: due maschere con 340 posizioni attive
    "50000",      # M11: "il validator passa su 50 000 esempi"
    "2.5", "97.5",            # M12: il 2,5o e il 97,5o percentile del bootstrap
    "71.2", "73.5",           # M12: un confronto ipotetico in una domanda
    "5000",       # M13: dimensione di corpus ipotetica
}

# Configurazioni: numeri che descrivono un setup, non un risultato.
# Compaiono in pagina perche' sono stati SCELTI, non misurati.
IGNORA_CONFIG = {
    "128", "256", "512", "1024", "2048", "4096", "5120", "8192",
    "11008", "13824", "32000", "128000", "141", "200", "500", "1000",
}


def blocchi_pre(testo):
    """Ogni <pre> con il suo eventuale data-lab. Restituisce
    (data_lab_o_None, contenuto)."""
    out = []
    for m in re.finditer(r"<pre([^>]*)><code[^>]*>(.*?)</code></pre>", testo, re.S):
        attr = m.group(1)
        dl = re.search(r'data-lab\s*=\s*"([^"]+)"', attr)
        out.append((dl.group(1) if dl else None, m.group(2)))
    return out


def prosa(testo):
    """Testo della pagina SENZA i blocchi di codice, normalizzato."""
    t = re.sub(r"<pre[^>]*>.*?</pre>", " ", testo, flags=re.S)
    t = re.sub(r"<code[^>]*>.*?</code>", " ", t, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    t = html.unescape(t)
    t = t.replace(" ", " ").replace(" ", " ").replace(" ", " ")
    # In prosa le migliaia sono separate da uno SPAZIO: '16 777 216',
    # '65 504'. Senza ricucirle la regex vede '777', '216', '504' - numeri
    # che nessun lab stampa, e il rapporto si riempie di rumore.
    for _ in range(3):
        t = re.sub(r"(?<=\d) (?=\d{3}(?!\d))", "", t)
    # Riferimenti ai lab: 'Lab 06.1', 'lab 14.1'. Sono nomi, non misure.
    t = re.sub(r"\bLab\s+\d{2}\.\d\b", " ", t, flags=re.I)
    return t


def ha_numeri_di_misura(blocco):
    """Un blocco e' candidato 'output di lab'.

    La soglia della prima versione era due decimali o una notazione
    scientifica, e lasciava fuori proprio i blocchi piu' esposti: la
    tabella dei token di M13 (441 / 214 / 64, nessun decimale) e la riga
    'ppl 694' (un decimale solo) erano invisibili, ed erano sbagliate
    entrambe. Meglio troppi candidati e triage a mano che una soglia che
    decide da sola cosa non guardare.
    """
    t = html.unescape(blocco)
    righe = [r.strip() for r in t.splitlines() if r.strip()]
    if not righe:
        return False
    # Blocco di comandi: decide la PRIMA riga non vuota. Guardarle tutte
    # sembrava piu' prudente e non lo era: un comando spezzato su piu'
    # righe con '\' ha continuazioni che cominciano con '--', e il blocco
    # finiva fra i candidati a output di lab.
    if righe[0].startswith(("python3", "torchrun", "pip", "#", "$")):
        return False
    # Blocco di codice sorgente. Il segnale devono darlo ';' '{' '}' e le
    # parole chiave: parentesi tonde e '=' da soli NON bastano, perche' le
    # tabelle di output ne sono piene - 'sigma(-z) (il PESO)', 'z = beta *
    # margine'. Con la regola larga due blocchi di M16 sparivano dal
    # controllo senza che niente lo segnalasse.
    codice = sum(1 for r in righe
                 if re.search(r"[;{}]|^\s*(def|for|if|import|class|return)\b", r))
    if codice >= max(2, len(righe) * 0.6):
        return False
    if re.search(r"\d+[.,]?\d*e[+-]?\d+", t, re.I):
        return True
    if re.search(r"\d+[.,]\d+", t):                  # almeno un decimale
        return True
    return len(re.findall(r"(?<![\w.])\d{2,}(?![\w])", t)) >= 3


def valori_di(testo):
    s = set()
    for m in re.finditer(r"[-+]?\d+(?:[.,]\d+)*(?:e[+-]?\d+)?", testo, re.I):
        v = val(m.group(0), italiano=False)
        if v is not None:
            s.add(v)
    return s


# ----------------------------------------------------------------------
def main():
    print("Eseguo i lab...\n")
    lab_out = esegui_lab()

    valori_lab = {n: valori_di(t) for n, t in lab_out.items()}
    unione = set().union(*valori_lab.values()) if valori_lab else set()

    pagine = sorted(QUI.glob("*.html"))
    tot_b = tot_p = 0
    marcati = non_marcati = 0
    rapporto, senza_marca = [], []
    usi_tolleranza = {"ESATTO": 0, "ARROTONDA": 0, "REL_FP": 0}

    for p in pagine:
        testo = p.read_text()
        prob_blocchi, prob_prosa = [], []

        # --- controllo 1: BLOCCHI -------------------------------------
        for i, (data_lab, blocco) in enumerate(blocchi_pre(testo)):
            if not ha_numeri_di_misura(blocco):
                continue
            if data_lab == "nessuno":
                # Blocco dichiarato NON output di lab: esempio didattico,
                # schema inventato, formato illustrativo. Va marcato lo
                # stesso, perche' "non viene da un lab" e' una decisione da
                # prendere una volta e scrivere, non da ripetere a ogni giro.
                marcati += 1
                continue
            if data_lab:
                marcati += 1
                chiavi = [k for k in valori_lab if k.startswith(data_lab)]
                if not chiavi:
                    prob_blocchi.append(
                        (i, data_lab, "—", f"data-lab sconosciuto: {data_lab}"))
                    continue
                ambito = set().union(*(valori_lab[k] for k in chiavi))
                etichetta = data_lab
            else:
                non_marcati += 1
                senza_marca.append((p.name, i))
                ambito = unione
                etichetta = "(non marcato)"

            for riga in html.unescape(blocco).splitlines():
                n = norm(riga)
                # '# ' con lo spazio e' un commento; '#1' e' un'etichetta di
                # riga. Scartare anche la seconda buttava via l'intera
                # tabella delle esecuzioni di M15.
                if not n or n.startswith(("python3", "torchrun", "pip", "$")) \
                        or n.startswith("# "):
                    continue
                n_pulita = re.sub(r"\[[\d,\s]+\]", " ", n)   # forme di tensore
                for m in RE_BLOCCO.finditer(n_pulita):
                    t = m.group(0)
                    # '3.12.2', '24.6.0': stringhe di versione, non numeri.
                    # Non sono convertibili, e trattarle come numeri manda
                    # nel rapporto ogni riga che dichiara un ambiente.
                    if RE_VERSIONE.fullmatch(t):
                        continue
                    chiave = norm_num(t, italiano=False)
                    if chiave in IGNORA or chiave in IGNORA_CONFIG:
                        continue
                    regola, vicino = confronta(t, ambito, italiano=False)
                    if regola:
                        usi_tolleranza[regola] += 1
                        continue
                    prob_blocchi.append((i, etichetta, t, vicino))

        # --- controllo 2: PROSA ---------------------------------------
        for m in RE_PROSA.finditer(prosa(testo)):
            t = m.group(1)
            chiave = norm_num(t)
            if chiave in IGNORA or chiave in IGNORA_CONFIG:
                continue
            regola, vicino = confronta(t, unione)
            if regola:
                usi_tolleranza[regola] += 1
                continue
            prob_prosa.append((t, vicino))

        tot_b += len(prob_blocchi)
        tot_p += len(prob_prosa)
        if prob_blocchi or prob_prosa:
            rapporto.append((p.name, prob_blocchi, prob_prosa))

    # ------------------------------------------------------------------
    print(f"\n{'=' * 74}")
    print(f"PAGINE: {len(pagine)}   LAB: {len(lab_out)}")
    print(f"{'=' * 74}")
    print("COPERTURA DELLA MARCATURA data-lab")
    tot_blocchi = marcati + non_marcati
    quota = 100 * marcati / tot_blocchi if tot_blocchi else 0
    print(f"  blocchi di output marcati     : {marcati:>4} / {tot_blocchi}"
          f"  ({quota:.0f}%)")
    print(f"  blocchi NON marcati           : {non_marcati:>4}"
          f"   <- confrontati sull'unione dei lab, controllo debole")
    if non_marcati and not BREVI:
        for nome, i in senza_marca[:20]:
            print(f"      {nome} blocco #{i}")
        if len(senza_marca) > 20:
            print(f"      ... e altri {len(senza_marca) - 20}")

    if SOLO_COPERTURA:
        return 0

    print(f"\n{'=' * 74}")
    print(f"numeri nei blocchi non trovati : {tot_b}")
    print(f"numeri in prosa non trovati    : {tot_p}")
    print(f"\nregole di tolleranza usate: "
          f"ESATTO {usi_tolleranza['ESATTO']}, "
          f"ARROTONDA {usi_tolleranza['ARROTONDA']}, "
          f"REL_FP {usi_tolleranza['REL_FP']}")
    if usi_tolleranza["REL_FP"]:
        print("  I REL_FP sono scarti backend-dipendenti: la pagina dovrebbe")
        print("  dichiarare ordine di grandezza e ambiente, non un valore secco.")
    print(f"{'=' * 74}")

    for nome, blocchi, numeri in rapporto:
        print(f"\n── {nome} ── {len(blocchi)} nei blocchi, {len(numeri)} in prosa")
        if BREVI:
            continue
        for i, lab, atteso_no, vicino in blocchi[:12]:
            print(f"   BLOCCO #{i} [{lab}]  pagina={atteso_no:>14}"
                  f"   piu' vicino nel lab={vicino}")
        if len(blocchi) > 12:
            print(f"   ... e altri {len(blocchi) - 12}")
        for t, vicino in numeri[:12]:
            print(f"   PROSA   pagina={t:>14}   piu' vicino nei lab={vicino}")
        if len(numeri) > 12:
            print(f"   ... e altri {len(numeri) - 12}")

    print(f"\n{'=' * 74}")
    print("""COME SI LEGGE

  Ogni riga dice: che cosa c'e' in PAGINA, e qual e' il valore piu' vicino
  prodotto dal lab. Se i due numeri si somigliano, la pagina e' rimasta
  indietro rispetto al lab: si ricopia l'output. Se sono lontani, il numero
  in pagina non viene da nessuna misura - o viene da un calcolo fatto in
  prosa, e allora va in IGNORA con la sua motivazione.

  Falsi positivi attesi, da mandare in IGNORA/IGNORA_CONFIG una volta letti:
    - valori di configurazione scelti, non misurati (hidden=4096, r=16)
    - numeri di esempi didattici inventati apposta
    - risultati citati da un ALTRO modulo (li' la marcatura data-lab non
      aiuta: il numero e' corretto ma nasce in un altro lab)

  Cio' che va guardato uno per uno: perplessita', loss, max|diff|, GB,
  percentuali, conteggi di token. Sono le misure su cui il corso si regge.""")
    return 1 if (tot_b or tot_p) else 0


if __name__ == "__main__":
    sys.exit(main())
