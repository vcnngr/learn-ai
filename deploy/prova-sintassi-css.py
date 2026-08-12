#!/usr/bin/env python3
"""
Il gate di layout legge il CSS come lo legge un browser?

    python3 deploy/prova-sintassi-css.py

Il gate confronta stringhe; il browser applica una sintassi. Dove le
due cose divergono il gate mente, e mente in silenzio: dice ok mentre
la pagina e' rotta.

Tre famiglie di divergenza, tutte trovate da una review e tutte
confermate iniettandole:

  1. `!important` ammette spazio fra `!` e `important` ed e'
     case-insensitive. Il confronto letterale su "!important" mancava
     quattro varianti su cinque.
  2. I NOMI delle proprieta' sono case-insensitive: `FLEX: 1` dopo
     `flex: none` vince nel browser. Tenendo il caso originale
     finivano in due chiavi distinte e il gate non vedeva niente.
  3. Anche i VALORI a parola chiave lo sono: `COLUMN` e' un
     flex-direction valido. Il confronto esatto lo dava per rotto.

Le prime due DEVONO essere viste: sono violazioni vere. La terza deve
essere TOLLERATA: e' CSS legittimo, e accusarlo e' un falso rosso.
Un gate che sbaglia in questa direzione si impara a ignorare, che e'
il modo in cui muore.
"""

import pathlib
import subprocess
import sys

QUI = pathlib.Path(__file__).resolve().parent.parent
CSS = QUI / "corso/assets/style.css"
GATE = QUI / "deploy/prova-layout.py"

# (ancora, sostituzione, riga da cercare, deve_essere_visto, etichetta)
#
# Le violazioni si iniettano PRIMA della dichiarazione buona: e'
# l'important o la cascata a farle vincere, non l'ordine.
PIEDE = ".sidebar-piede {\n  flex: none;"
COLONNA = "flex-direction: column;"

CASI = [
    (PIEDE, f".sidebar-piede {{\n  flex: 1 !important;\n  flex: none;",
     "non si comprime", True, "!important"),
    (PIEDE, f".sidebar-piede {{\n  flex: 1 ! important;\n  flex: none;",
     "non si comprime", True, "! important (spazio)"),
    (PIEDE, f".sidebar-piede {{\n  flex: 1 !IMPORTANT;\n  flex: none;",
     "non si comprime", True, "!IMPORTANT (maiuscolo)"),
    (PIEDE, f".sidebar-piede {{\n  flex: 1 !Important;\n  flex: none;",
     "non si comprime", True, "!Important (misto)"),
    (PIEDE, f".sidebar-piede {{\n  flex: 1  !  important ;\n  flex: none;",
     "non si comprime", True, "!  important  (spazi)"),
    (PIEDE, f".sidebar-piede {{\n  flex: none;\n  FLEX: 1;",
     "non si comprime", True, "FLEX: nome maiuscolo"),
    (PIEDE, f".sidebar-piede {{\n  flex: none;\n  Flex: 1;",
     "non si comprime", True, "Flex: nome misto"),
    (COLONNA, "flex-direction: COLUMN;",
     "colonna flex", False, "COLUMN: valore maiuscolo"),
    (COLONNA, "flex-direction: Column;",
     "colonna flex", False, "Column: valore misto"),
]


def gate():
    return subprocess.run([sys.executable, str(GATE)],
                          capture_output=True, text=True, cwd=QUI)


originale = CSS.read_text()

print("Il gate legge il CSS come lo legge un browser?\n")

falliti = 0
try:
    for ancora, sost, riga, visto_atteso, etichetta in CASI:
        if ancora not in originale:
            print(f"  {etichetta:<28} ANCORA ASSENTE: non dimostra niente")
            falliti += 1
            continue
        CSS.write_text(originale.replace(ancora, sost, 1))
        r = gate()
        accusa = any(riga in l and "MANCA" in l for l in r.stdout.splitlines())
        # non basta l'uscita 1: il gate deve NOMINARE la riga giusta,
        # altrimenti un rosso per altra causa passa per prova riuscita
        ok = (accusa and r.returncode == 1) if visto_atteso else not accusa
        if ok:
            esito = "visto" if visto_atteso else "tollerato"
        else:
            esito = "FALSO VERDE" if visto_atteso else "FALSO ROSSO"
        print(f"  {etichetta:<28} {esito}")
        falliti += not ok
finally:
    CSS.write_text(originale)

integro = gate().returncode == 0
print(f"\n  {'a CSS integro tace':<28} {'si' if integro else 'NO'}")
falliti += not integro

print()
if falliti:
    print(f"  {falliti} casi sbagliati: il gate e il browser leggono CSS diversi.")
    sys.exit(1)
print("  Nessuna forma valida di CSS sfugge al gate ne' viene accusata a torto.")
