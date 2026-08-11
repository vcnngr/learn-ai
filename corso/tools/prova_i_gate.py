#!/usr/bin/env python3
"""
prova_i_gate.py — i controlli sanno fallire?

    python3 corso/tools/prova_i_gate.py

Un gate che non fallisce mai non e' un gate: e' una riga di log che
rassicura. Questo script rompe di proposito una cosa alla volta e
verifica che il controllo corrispondente se ne accorga, poi ripristina.

NON e' zelo. `verifica.py --copertura` faceva `return 0` incondizionato
mentre la CI lo invocava dichiarandolo «controllo DURO»: per due commit
il corso ha avuto un controllo che non poteva fallire, ed e' il difetto
tipo 2 del brief - una procedura che non puo' produrre la risposta che
le si chiede - commesso proprio nello strumento che dovrebbe trovarlo
negli altri.

Se aggiungi un controllo, aggiungi qui la prova che sa fallire.
"""

import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

QUI = pathlib.Path(__file__).resolve().parent.parent
PY = sys.executable

PROVE = []


def prova(nome, file_da_rompere, rompi, comando, firma):
    """`firma` e' cio' che il rapporto DEVE nominare quando il difetto c'e'.

    Il codice di uscita da solo non basta, ed e' il secondo difetto che
    questo file ha avuto. Su una macchina dove il gate fallisce gia' a
    riposo — la CI, dove la CPU non e' quella di riferimento — iniettare
    una mutazione lascia l'uscita a 1 comunque, e la prova passa senza
    aver dimostrato niente.

    Quindi si verificano due cose insieme:
      - col difetto:   il rapporto NOMINA la mutazione
      - senza difetto: il rapporto NON la nomina
    E' la differenza fra 'il controllo era rosso' e 'il controllo ha
    visto QUESTO'.
    """
    PROVE.append((nome, file_da_rompere, rompi, comando, firma))


# ----------------------------------------------------------------------
prova("gate numerico: un numero che nessun lab produce",
      "m20-parity.html",
      lambda t: t.replace("7.727e-03", "9.999e-03", 1),
      [PY, "verifica.py"],
      firma="9.999e-03")

prova("copertura: un blocco senza data-lab",
      "m20-parity.html",
      lambda t: t.replace('<pre data-lab="lab_20_1">', "<pre>", 1),
      [PY, "verifica.py", "--copertura"],
      firma="m20-parity.html blocco #")

prova("parity: conti.js che diverge dal lab",
      "assets/conti.js",
      lambda t: t.replace("adamw8bit: 2", "adamw8bit: 3", 1),
      [PY, "tools/parity_conti.py"],
      firma="DIVERGONO")


# ----------------------------------------------------------------------
def esegui(comando):
    r = subprocess.run(comando, cwd=QUI, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


print("Provo che i gate vedano il difetto, non solo che diventino rossi.\n")
print(f"  {'prova':<46} {'uscita':>7} {'nomina?':>9} {'tace?':>7}")

falliti = 0
for nome, rel, rompi, comando, firma in PROVE:
    f = QUI / rel
    originale = f.read_text()

    # 1. col difetto
    try:
        rotto = rompi(originale)
        if rotto == originale:
            print(f"  {nome:<46} {'—':>7} {'NON ROTTO':>9}")
            print(f"      la modifica non ha cambiato {rel}: la prova non "
                  f"dimostra niente")
            falliti += 1
            continue
        f.write_text(rotto)
        uscita, out_rotto = esegui(comando)
    finally:
        f.write_text(originale)

    # 2. senza difetto: la firma NON deve comparire, altrimenti la prova
    #    passerebbe anche senza aver visto la mutazione
    _, out_integro = esegui(comando)

    nomina = firma in out_rotto
    tace = firma not in out_integro
    ok = uscita == 1 and nomina and tace
    falliti += not ok
    print(f"  {nome:<46} {uscita:>7} {'si' if nomina else 'NO':>9}"
          f" {'si' if tace else 'NO':>7}{'' if ok else '   <- NON DIMOSTRA'}")

# A corso integro devono tacere. Ma non tutti allo stesso titolo:
#
#   copertura e parity NON dipendono dalla CPU - la prima guarda la
#   marcatura, il secondo e' aritmetica esatta. Devono passare ovunque.
#
#   il gate NUMERICO dipende dalla CPU, e lo dice AMBIENTI.md: i numeri
#   pubblicati sono l'output del container sulla macchina che li ha
#   generati, e su un processore diverso divergono. Preteserlo verde
#   ovunque sarebbe la stessa presunzione che questo file esiste per
#   smontare - e infatti la prima versione lo pretendeva, e la CI l'ha
#   colta in fallo.
RIFERIMENTO = "--riferimento" in sys.argv

print()
for nome, comando, dipende_da_cpu in [
        ("copertura", [PY, "verifica.py", "--copertura", "--brevi"], False),
        ("parity", [PY, "tools/parity_conti.py"], False),
        ("gate numerico", [PY, "verifica.py", "--brevi"], True)]:
    u, _ = esegui(comando)
    if dipende_da_cpu and not RIFERIMENTO:
        stato = "ok" if u == 0 else "diverge (atteso fuori dal riferimento)"
        print(f"  {'a corso integro: ' + nome:<48} {'—':>7} {u:>9}   {stato}")
        continue
    ok = u == 0
    falliti += not ok
    print(f"  {'a corso integro: ' + nome:<48} {0:>7} {u:>9}"
          f"{'' if ok else '  <- FALSO ALLARME'}")

if not RIFERIMENTO:
    print("""
  Il gate numerico e' stato eseguito ma NON conteggiato: qui non si sa se
  la CPU sia quella che ha generato i numeri. Sulla macchina di
  riferimento va preteso verde, e li' si lancia con --riferimento.""")

print()
if falliti:
    print(f"""  {falliti} prove non superate.

  Un controllo che non si accorge di un difetto introdotto apposta non
  sta controllando: il verde che produce non significa niente.""")
    sys.exit(1)

print("""  Tutti i gate si accorgono del difetto e tacciono quando non c'e'.

  Vale la pena notare cosa NON dimostra: che i gate trovino OGNI difetto.
  Dimostra che trovano questi, che sono quelli che li hanno gia' colti in
  fallo una volta. E' un test di regressione, non una garanzia.""")
