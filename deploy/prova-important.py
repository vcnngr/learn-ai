#!/usr/bin/env python3
"""
Il gate di layout riconosce !important in tutte le forme valide?

    python3 deploy/prova-important.py

La sintassi CSS ammette spazio fra `!` e `important` ed e'
case-insensitive: `! important` e `!IMPORTANT` sono validi quanto
`!important`. Un confronto letterale sulla stringa "!important" li
manca tutti — tre falsi verdi su tre varianti provate, prima che una
review lo segnalasse.

Ogni variante viene iniettata PRIMA della dichiarazione buona: nel
browser vince l'important, quindi l'invariante e' violato e il gate
deve accorgersene.
"""

import pathlib
import re
import subprocess
import sys

QUI = pathlib.Path(__file__).resolve().parent.parent
CSS = QUI / "corso/assets/style.css"
ANCORA = ".sidebar-piede {\n  flex: none;"

VARIANTI = [
    "flex: 1 !important;",
    "flex: 1 ! important;",
    "flex: 1 !IMPORTANT;",
    "flex: 1 !Important;",
    "flex: 1  !  important ;",
]

originale = CSS.read_text()
if ANCORA not in originale:
    print(f"  ANCORA ASSENTE in {CSS.name}: la prova non dimostra niente")
    sys.exit(1)

print("Il gate riconosce !important in ogni forma valida?\n")
falliti = 0
try:
    for v in VARIANTI:
        CSS.write_text(originale.replace(ANCORA, f".sidebar-piede {{\n  {v}\n  flex: none;", 1))
        r = subprocess.run([sys.executable, str(QUI / "deploy/prova-layout.py")],
                           capture_output=True, text=True, cwd=QUI)
        riga = [l for l in r.stdout.splitlines() if "non si comprime" in l]
        visto = bool(riga) and "flex vale" in riga[0] and r.returncode == 1
        print(f"  {v:<28} {'visto' if visto else 'FALSO VERDE'}")
        falliti += not visto
finally:
    CSS.write_text(originale)

# e a CSS integro deve tacere
r = subprocess.run([sys.executable, str(QUI / "deploy/prova-layout.py")],
                   capture_output=True, text=True, cwd=QUI)
integro = r.returncode == 0
print(f"\n  {'a CSS integro tace':<28} {'si' if integro else 'NO'}")
falliti += not integro

print()
if falliti:
    print(f"  {falliti} casi non visti: il gate accetta un !important che il browser applica.")
    sys.exit(1)
print("  Nessuna forma valida di !important sfugge al gate.")
