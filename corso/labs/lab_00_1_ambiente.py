#!/usr/bin/env python3
"""
Lab 00.1 — L'ambiente: cosa hai, cosa manca, cosa potrai eseguire.   CPU.

    python3 corso/labs/lab_00_1_ambiente.py

Non installa niente e non cambia niente. Guarda, misura e dice.

Il corso promette che ogni numero viene da una misura. Questo lab applica
la stessa regola all'ambiente: invece di dirti "serve Python recente e
torch", ti dice quale Python e quale torch HAI, cosa ci puoi fare, e
quali sezioni del corso non gireranno su questa macchina.
"""

import importlib
import platform
import sys


def sez(t):
    print(f"\n{'─' * 70}\n{t}\n{'─' * 70}")


ESITO = {"bloccanti": [], "avvisi": []}


# ----------------------------------------------------------------------
sez("1. Interprete e piattaforma")

py = sys.version_info
print(f"  {'python':<22} {platform.python_version()}")
print(f"  {'eseguibile':<22} {sys.executable}")
print(f"  {'sistema':<22} {platform.system()} {platform.release()}")
print(f"  {'architettura':<22} {platform.machine()}")

if py < (3, 9):
    ESITO["bloccanti"].append(
        f"python {platform.python_version()}: i lab usano sintassi da 3.9 in su")
    print("\n  PROBLEMA: serve Python 3.9 o superiore.")
else:
    print("\n  Python va bene: i lab non usano niente di piu' recente di 3.9.")


# ----------------------------------------------------------------------
sez("2. I pacchetti")

# (modulo, a cosa serve, bloccante da quale modulo del corso)
PACCHETTI = [
    ("torch", "tutto: tensori, autograd, addestramento", "M01"),
    ("numpy", "i lab di algebra e i conti su array", "M01"),
    ("safetensors", "aprire un checkpoint e leggerne i pesi", "M04"),
]

print(f"  {'pacchetto':<16} {'versione':<14} {'serve a'}")
trovati = {}
for nome, scopo, da_modulo in PACCHETTI:
    try:
        m = importlib.import_module(nome)
        v = getattr(m, "__version__", "?")
        trovati[nome] = m
        print(f"  {nome:<16} {v:<14} {scopo}")
    except ImportError:
        print(f"  {nome:<16} {'ASSENTE':<14} {scopo}")
        ESITO["bloccanti"].append(f"{nome} assente: serve da {da_modulo} in poi")

if len(trovati) < len(PACCHETTI):
    mancanti = [n for n, _, _ in PACCHETTI if n not in trovati]
    print(f"\n  Per installare quelli che mancano:")
    print(f"    python3 -m pip install {' '.join(mancanti)}")
    print("""
  Se pip fallisce con 'no matching distribution', NON e' un errore tuo:
  significa che per la tua combinazione di Python, sistema operativo e
  architettura quel pacchetto non e' pubblicato. Succede davvero - vedi
  la sezione 5.""")


# ----------------------------------------------------------------------
sez("3. Il minimo che deve funzionare")

if "torch" in trovati:
    torch = trovati["torch"]
    x = torch.tensor([2.0], requires_grad=True)
    y = (x ** 3).sum()
    y.backward()
    atteso = 3 * 2.0 ** 2
    ok = abs(x.grad.item() - atteso) < 1e-6
    print(f"  y = x^3 con x = 2  ->  dy/dx misurato {x.grad.item():.1f},"
          f" atteso {atteso:.1f}   {'OK' if ok else 'DIVERSO'}")
    print(f"  dtype di default        {torch.get_default_dtype()}")
    print(f"  thread usati da torch   {torch.get_num_threads()}")
    if not ok:
        ESITO["bloccanti"].append("autograd non produce il gradiente atteso")
    print("""
  Se questa riga dice OK, tutta la Parte I e la Parte II del corso
  girano su questa macchina. Sono le prime ~25 ore.""")
else:
    print("  Salto: torch non e' installato.")


# ----------------------------------------------------------------------
sez("4. Acceleratori")

if "torch" in trovati:
    torch = trovati["torch"]
    cuda = torch.cuda.is_available()
    mps = getattr(torch.backends, "mps", None)
    mps_ok = bool(mps and mps.is_available())
    print(f"  {'CUDA disponibile':<24} {'SI' if cuda else 'NO'}")
    if cuda:
        print(f"  {'GPU visibili':<24} {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"    [{i}] {p.name}, {p.total_memory / 1e9:.0f} GB")
        print(f"  {'versione CUDA di torch':<24} {torch.version.cuda}")
    print(f"  {'MPS (Apple) disponibile':<24} {'SI' if mps_ok else 'NO'}")

    if not cuda:
        ESITO["avvisi"].append(
            "senza CUDA: M07 sez. 4-5 e M18 sez. 5 non sono eseguibili")
        print("""
  Senza CUDA il corso resta utilizzabile quasi per intero. Due sezioni
  si fermano, e si fermano PULITE - con un messaggio esplicito, non con
  un errore:

    M07 sezioni 4-5   picco di memoria e gradient checkpointing su GPU
    M18 sezione 5     memoria per rank e scaling con torchrun

  Non stampano numeri stimati al posto di quelli misurati: sarebbero
  inventati, ed e' esattamente cio' che il corso insegna a non fare.
  Tutto il resto - inclusa l'ARITMETICA della memoria in M06 e la teoria
  del parallelismo in M18 - gira qui.

  NOTA: MPS non sostituisce CUDA per quei due lab. Usano torchrun e
  NCCL, che su MPS non ci sono.""")
else:
    print("  Salto: torch non e' installato.")


# ----------------------------------------------------------------------
sez("5. Le API che dipendono dalla versione")

print("""  Alcuni lab hanno due strade: una che usa un'API recente di PyTorch e
  una di riserva. Non e' pigrizia - e' che il corso deve dire QUALE
  strada ha eseguito, perche' le due possono dare numeri diversi.
""")

if "torch" in trovati:
    torch = trovati["torch"]
    import torch.nn.functional as F

    def ha_parametro(fn, nome):
        """`scaled_dot_product_attention` e' un builtin: inspect.signature
        spesso non lo descrive. Ricado sul docstring, e se manca anche
        quello dico 'non determinabile' invece di tirare a indovinare."""
        import inspect
        try:
            return nome in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            pass
        doc = getattr(fn, "__doc__", None)
        return (nome in doc) if doc else None

    API = [
        ("F.rms_norm",
         hasattr(F, "rms_norm"),
         "M09 confronta la tua RMSNorm con quella di libreria",
         "il lab usa la formula di riferimento scritta a mano"),
        ("scaled_dot_product_attention(enable_gqa=)",
         ha_parametro(F.scaled_dot_product_attention, "enable_gqa"),
         "M09 verifica la GQA con il kernel nativo",
         "il lab espande le teste K/V a mano"),
    ]
    def etichetta(p):
        return "SI" if p is True else ("NO" if p is False else "?")

    print(f"  {'API':<44} {'presente':<10} {'se assente'}")
    for nome, presente, _a_cosa, fallback in API:
        print(f"  {nome:<44} {etichetta(presente):<10} {fallback}")

    assenti = [n for n, p, _, _ in API if p is not True]
    if assenti:
        ESITO["avvisi"].append(
            f"{len(assenti)} API recenti assenti: i lab usano la strada di riserva")
        print(f"""
  Le strade di riserva sono corrette e verificate: producono lo stesso
  risultato matematico. Quello che NON puoi fare, con questa versione,
  e' verificare il ramo nativo - e infatti il corso lo dichiara
  'non verificabile' invece di dare per scontato che funzioni.""")


# ----------------------------------------------------------------------
sez("6. Il profilo: adattare il corso al tuo hardware")

print("""  Il file corso/assets/profilo.js descrive l'hardware, il modello di
  riferimento e il caso di studio. Serve a due cose, e conviene sapere
  quale fa e quale NON fa:

    FA         aggiorna l'intestazione delle pagine, e ti da' i valori da
               passare ai lab che accettano parametri, per esempio:
                 python3 lab_06_1_memoria.py --params 8e9 --gpus 4 --hbm 80

    NON FA     non ricalcola i numeri gia' scritti nelle pagine. I 54 GB,
               i 432 GB e i conteggi LoRA restano quelli del profilo di
               riferimento. Vanno letti come un CASO SVOLTO: il metodo e'
               tuo, i numeri sono suoi.

  Confonderle e' facile e costa caro: cambieresti 'parametri' aspettando
  che le tabelle seguano, e resterebbero ferme.

  I campi da toccare per primi, in ordine di impatto sui conti:
    modello.parametri     entra in ogni conto di memoria (M06)
    gpu.memoriaGB         decide cosa ci sta e cosa no
    gpu.quante            cambia il batch effettivo (M08) e M18
    caso.lora             la configurazione dell'adattatore: da li' esce
                          la frazione di addestrabili usata da M06 e M17""")


# ----------------------------------------------------------------------
sez("7. Verdetto")

if ESITO["bloccanti"]:
    print("  BLOCCANTI - vanno risolti prima di cominciare:")
    for b in ESITO["bloccanti"]:
        print(f"    - {b}")
else:
    print("  Nessun problema bloccante: puoi cominciare da M01.")

if ESITO["avvisi"]:
    print("\n  AVVISI - il corso gira, con queste sezioni non eseguibili:")
    for a in ESITO["avvisi"]:
        print(f"    - {a}")
else:
    print("\n  Nessun avviso: anche le sezioni GPU e i rami nativi sono eseguibili.")

print(f"""
  Un ambiente diverso dal mio non e' un problema da risolvere: e' un
  dato da registrare. Quando un numero non torna, la prima domanda e'
  'quale versione ho?', e questo lab e' la risposta. In M15 lo stesso
  principio diventa il MANIFEST di un addestramento.""")

print()
