#!/usr/bin/env python3
"""
parity_conti.py — le due strade danno lo stesso conto?

    python3 corso/tools/parity_conti.py

Le pagine mostrano i numeri del profilo di riferimento, verificati da
verifica.py contro l'output dei lab. Accanto, assets/conti.js ricalcola
gli stessi conti per il profilo del LETTORE, nel browser.

Sono due implementazioni della stessa formula, in due linguaggi. Se
divergono, la pagina mostra al lettore un numero che nessun lab produce
- ed e' il difetto che tutto questo corso esiste per impedire.

E' il controllo di parity di M02 e M20 applicato al corso stesso:
stessa funzione per due strade, esito BINARIO, nessuna tolleranza da
negoziare. Qui i conti sono aritmetica esatta su interi e potenze di
due, quindi l'uguaglianza deve essere esatta - non "entro l'1%".

Esce 1 se una qualunque voce diverge. Sta in RILASCIO.md.
"""

import json
import pathlib
import shutil
import subprocess
import sys

QUI = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QUI / "labs"))

NODE = shutil.which("node")
if not NODE:
    print("  SALTATO: node non e' installato, e serve per eseguire conti.js.")
    print("  Il controllo non e' stato fatto: NON e' passato.")
    sys.exit(2)


# ----------------------------------------------------------------------
# 1. la strada Python: la funzione del lab, importata invece che ricopiata

def calcola_python(P, metodo, opt, lora_frac):
    """Riusa `calcola` di lab_06_1_memoria.py senza eseguirne il main.

    Il file e' uno script, non un modulo: lo leggo ed estraggo la sola
    funzione. Ricopiarla qui vorrebbe dire avere TRE implementazioni da
    tenere allineate invece di due, e la terza sarebbe invisibile.
    """
    sorgente = (QUI / "labs" / "lab_06_1_memoria.py").read_text()
    inizio = sorgente.index("def calcola(")
    fine = sorgente.index("def stampa(")
    ns = {}
    exec(sorgente[inizio:fine], ns)
    return ns["calcola"](P, metodo, opt, lora_frac)


# ----------------------------------------------------------------------
# 2. la strada JS

JS = """
const CONTI = require(%s);
// con `node -e` lo script NON compare in argv: gli argomenti partono
// dall'indice 1, non dal 2. Con slice(2) si perde il primo e tutto
// scala di uno - il primo difetto che questo parity ha trovato.
const [P, metodo, opt, frac] = process.argv.slice(1);
const voci = CONTI.voci(Number(P), metodo, opt, Number(frac));
console.log(JSON.stringify(voci.map(v => [v.nome, v.byte])));
"""


def calcola_js(P, metodo, opt, lora_frac):
    script = JS % json.dumps(str(QUI / "assets" / "conti.js"))
    r = subprocess.run(
        [NODE, "-e", script, repr(P), metodo, opt, repr(lora_frac)],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit("  conti.js non e' eseguibile: vedi l'errore sopra.")
    return json.loads(r.stdout)


# ----------------------------------------------------------------------
CASI = [
    # (parametri, metodo, optimizer, frazione addestrabili)
    (27e9,  "full",  "adamw",     0.00491),
    (27e9,  "lora",  "adamw",     0.00491),
    (27e9,  "qlora", "adamw",     0.00491),
    (27e9,  "lora",  "adamw8bit", 0.00491),
    (27e9,  "full",  "sgd",       0.00491),
    # profili di altri regimi, per non verificare un caso solo
    (8e9,   "lora",  "adamw",     0.00045),   # 8B, q,v a r=16
    (70e9,  "qlora", "adamw8bit", 0.0100),    # 70B, LoRA largo
    (1.5e9, "full",  "adamw",     0.00491),   # modellino
]

print("Parity fra assets/conti.js e labs/lab_06_1_memoria.py\n")
print(f"  {'caso':<34} {'voci':>5} {'esito':>10}")

diversi = 0
dettagli = []
for P, metodo, opt, frac in CASI:
    py = calcola_python(P, metodo, opt, frac)
    js = calcola_js(P, metodo, opt, frac)
    etichetta = f"{P:.3g} {metodo}/{opt}"

    if len(py) != len(js):
        diversi += 1
        dettagli.append(f"{etichetta}: {len(py)} voci in Python, {len(js)} in JS")
        print(f"  {etichetta:<34} {len(py):>5} {'DIVERSO':>10}")
        continue

    scarti = []
    for (nome_py, byte_py, _spieg), (nome_js, byte_js) in zip(py, js):
        if nome_py != nome_js:
            scarti.append(f"nome: {nome_py!r} vs {nome_js!r}")
        elif byte_py != byte_js:
            scarti.append(f"{nome_py}: {byte_py!r} vs {byte_js!r}")
    if scarti:
        diversi += 1
        dettagli.extend(f"{etichetta} -> {s}" for s in scarti)
    print(f"  {etichetta:<34} {len(py):>5} {'OK' if not scarti else 'DIVERSO':>10}")

print()
if diversi:
    print(f"  {diversi} casi su {len(CASI)} DIVERGONO:\n")
    for d in dettagli:
        print(f"    {d}")
    print("""
  Le due strade non calcolano piu' la stessa cosa. Finche' e' cosi', il
  ricalcolo mostrato al lettore nelle pagine e' un numero che nessun lab
  produce. Allinea conti.js a lab_06_1_memoria.py - o viceversa, se e'
  il lab a essere rimasto indietro - e rilancia.""")
    sys.exit(1)

print(f"""  Tutti i {len(CASI)} casi coincidono, byte per byte.

  Uguaglianza ESATTA, non entro tolleranza: questi conti sono
  moltiplicazioni fra interi e potenze di due, e in doppia precisione
  non c'e' arrotondamento da concedere. Se un giorno comparisse uno
  scarto minuscolo, non sarebbe rumore numerico: sarebbe una delle due
  formule cambiata.""")
