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


def prova(nome, file_da_rompere, rompi, comando, atteso=1):
    """Rompe, lancia, verifica il codice di uscita, ripristina SEMPRE."""
    PROVE.append((nome, file_da_rompere, rompi, comando, atteso))


# ----------------------------------------------------------------------
prova("gate numerico: un numero che nessun lab produce",
      "m20-parity.html",
      lambda t: t.replace("7.727e-03", "9.999e-03", 1),
      [PY, "verifica.py", "--brevi"])

prova("copertura: un blocco senza data-lab",
      "m20-parity.html",
      lambda t: t.replace('<pre data-lab="lab_20_1">', "<pre>", 1),
      [PY, "verifica.py", "--copertura", "--brevi"])

prova("parity: conti.js che diverge dal lab",
      "assets/conti.js",
      lambda t: t.replace("adamw8bit: 2", "adamw8bit: 3", 1),
      [PY, "tools/parity_conti.py"])


# ----------------------------------------------------------------------
def esegui(comando):
    r = subprocess.run(comando, cwd=QUI, capture_output=True, text=True)
    return r.returncode


print("Provo che i gate sappiano fallire.\n")
print(f"  {'prova':<48} {'atteso':>7} {'ottenuto':>9}")

falliti = 0
for nome, rel, rompi, comando, atteso in PROVE:
    f = QUI / rel
    originale = f.read_text()
    try:
        rotto = rompi(originale)
        if rotto == originale:
            print(f"  {nome:<48} {'—':>7} {'NON ROTTO':>9}")
            print(f"      la modifica non ha cambiato {rel}: la prova non "
                  f"dimostra niente")
            falliti += 1
            continue
        f.write_text(rotto)
        uscita = esegui(comando)
    finally:
        f.write_text(originale)

    ok = uscita == atteso
    falliti += not ok
    print(f"  {nome:<48} {atteso:>7} {uscita:>9} {'' if ok else '  <- NON SE NE ACCORGE'}")

# e a corso integro devono passare tutti
print()
for nome, comando in [("gate numerico", [PY, "verifica.py", "--brevi"]),
                      ("copertura", [PY, "verifica.py", "--copertura", "--brevi"]),
                      ("parity", [PY, "tools/parity_conti.py"])]:
    u = esegui(comando)
    ok = u == 0
    falliti += not ok
    print(f"  {'a corso integro: ' + nome:<48} {0:>7} {u:>9}"
          f"{'' if ok else '  <- FALSO ALLARME'}")

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
