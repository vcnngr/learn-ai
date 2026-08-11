#!/usr/bin/env python3
"""
Lab 06.1 — Calcolatore di memoria per-GPU.   CPU (e' solo aritmetica).

    python3 corso/labs/lab_06_1_memoria.py
    python3 corso/labs/lab_06_1_memoria.py --params 27e9 --method lora --gpus 2

Fai il conto A MANO prima di lanciarlo. Poi confronta.
Questo script non misura niente: applica la formula. La misura vera e' in M07,
e serve proprio a scoprire di quanto la formula sottostima.
"""

import argparse

GB = 1e9

ap = argparse.ArgumentParser()
ap.add_argument("--params", type=float, default=None, help="numero di parametri, es. 27e9")
ap.add_argument("--method", default=None, choices=["full", "lora", "qlora"])
ap.add_argument("--gpus", type=int, default=2)
ap.add_argument("--hbm", type=float, default=141.0, help="GB per GPU")
ap.add_argument("--optimizer", default="adamw", choices=["adamw", "adamw8bit", "sgd"])
# 0.00491 non e' un numero rotondo scelto a occhio: e' la frazione che
# produce la configurazione LoRA del caso di studio - tutti e sette i
# moduli a r=32 su un denso da 27B, cioe' 132.644.864 parametri.
# La conta la fa lab_14_1_lora.py sezione 1; la config sta in
# assets/profilo.js sotto caso.lora. Con un'altra config, passa --lora-frac.
ap.add_argument("--lora-frac", type=float, default=0.00491,
                help="frazione di parametri addestrabili con LoRA "
                     "(default: il caso di studio, 7 moduli a r=32; tipico 0.1-1%%)")
args = ap.parse_args()


def sez(t):
    print(f"\n{'─' * 72}\n{t}\n{'─' * 72}")


def calcola(P, metodo, opt, lora_frac):
    """Ritorna le voci di memoria in byte. Assunzioni dichiarate esplicitamente."""
    if metodo == "full":
        w_dtype, addestrabili = 2, P            # pesi bf16
    elif metodo == "lora":
        w_dtype, addestrabili = 2, P * lora_frac
    else:                                        # qlora
        w_dtype, addestrabili = 0.5, P * lora_frac   # base quantizzata a 4 bit

    stati = {"adamw": 8, "adamw8bit": 2, "sgd": 4}[opt]   # byte per parametro ADDESTRABILE

    voci = [
        ("pesi (base)",            P * w_dtype,
         f"{P:.3g} par x {w_dtype} byte"),
        ("gradienti",              addestrabili * 2,
         f"{addestrabili:.3g} par addestrabili x 2 byte (bf16)"),
        ("stati optimizer",        addestrabili * stati,
         f"{addestrabili:.3g} x {stati} byte ({opt})"),
    ]
    if metodo == "full":
        voci.append(("copia master fp32", P * 4, f"{P:.3g} x 4 byte (se l'impl. la tiene)"))
    else:
        voci.append(("copia master fp32", addestrabili * 4,
                     f"solo gli addestrabili: {addestrabili:.3g} x 4"))
    return voci


def stampa(P, metodo, opt, gpus, hbm, lora_frac):
    voci = calcola(P, metodo, opt, lora_frac)
    tot = sum(v[1] for v in voci)
    print(f"\n  metodo={metodo}  optimizer={opt}  parametri={P:.4g}"
          + (f"  addestrabili={lora_frac:.3%}" if metodo != "full" else ""))
    print(f"\n  {'voce':<22} {'GB':>9}   come si ottiene")
    for nome, byte, spieg in voci:
        print(f"  {nome:<22} {byte / GB:>9.1f}   {spieg}")
    print(f"  {'-' * 22} {'-' * 9}")
    print(f"  {'STATICO totale':<22} {tot / GB:>9.1f}   (senza attivazioni)")

    print(f"\n  Su {gpus} GPU da {hbm:.0f} GB ciascuna:")
    print(f"    replicato su ogni GPU (DDP)  -> {tot / GB:>7.1f} GB per GPU"
          f"   {'ENTRA' if tot / GB < hbm else 'NON ENTRA'}")
    sharded = tot / gpus
    print(f"    partizionato (ZeRO-3, ideale)-> {sharded / GB:>7.1f} GB per GPU"
          f"   {'ENTRA' if sharded / GB < hbm else 'NON ENTRA'}")
    print(f"    [somma su tutte le GPU: {tot / GB:.1f} GB - NON e' un pool unico]")
    print("    [riga ZeRO-3 = limite teorico: assume divisione perfetta e ignora")
    print("     i buffer di comunicazione. Se i pesi base sono congelati, se e come")
    print("     vengano partizionati dipende dall'implementazione: verificalo.]")
    return tot


sez("1. Le voci, e perche' esistono")

print("""  Cinque voci. Le prime quattro sono STATICHE: ci sono sempre, non dipendono
  dal batch. La quinta dipende da batch e lunghezza di sequenza.

    1. PESI            il modello.                        -> M04
    2. GRADIENTI       stessa forma dei pesi ADDESTRABILI. -> M02
    3. STATI OPTIMIZER AdamW tiene due momenti per parametro addestrabile,
                       tipicamente in fp32 perche' sono accumulatori. -> M05
    4. COPIA MASTER    alcune implementazioni tengono una copia fp32 dei
                       pesi addestrabili. Dipende dall'implementazione:
                       verificalo, non assumerlo.
    5. ATTIVAZIONI     gli ingressi di ogni strato, trattenuti per il
                       backward. Crescono con batch x seq len. -> M02, M07

  NOTA sul punto 2: i gradienti seguono i parametri ADDESTRABILI, non tutti
  i parametri. E' la ragione per cui LoRA cambia tutto il conto.""")


sez("2. Il caso che ti riguarda: 27B")

P = 27e9
if args.params is None:
    for metodo in ("full", "lora", "qlora"):
        stampa(P, metodo, args.optimizer, args.gpus, args.hbm, args.lora_frac)
else:
    stampa(args.params, args.method or "full", args.optimizer,
           args.gpus, args.hbm, args.lora_frac)


sez("3. Le attivazioni: l'unica voce che TU controlli")

print("  Stima grezza per un denso, con gradient checkpointing DISATTIVO:\n")
print("    attivazioni ~ batch x seq_len x hidden x n_layer x k x byte")
print("    dove k dipende da quante uscite intermedie ogni strato trattiene.\n")
print("  k non e' una costante universale: cambia con l'architettura e con")
print("  quali kernel sono fusi. Per questo la formula da' un ORDINE DI")
print("  GRANDEZZA, non un numero da fidarsi. La misura vera e' in M07.\n")

hidden, layers, bpp = 5120, 46, 2      # proporzioni tipiche di un denso ~27B
print(f"  Esempio con hidden={hidden}, layers={layers}, bf16, k=10 (stima):\n")
print(f"  {'batch':>6} {'seq':>7} {'GB attivazioni':>16}")
for batch in (1, 4, 8):
    for seq in (2048, 8192):
        att = batch * seq * hidden * layers * 10 * bpp
        print(f"  {batch:>6} {seq:>7} {att / GB:>16.1f}")

print("\n  Raddoppia il batch: raddoppia. Raddoppia la sequenza: raddoppia.")
print("  E' l'unica voce su cui puoi intervenire senza cambiare metodo.")
print("  Il gradient checkpointing la riduce molto, pagando in tempo. -> M07")


sez("4. Gli errori che rendono il conto inutile")

print("""  1. SOMMARE LE GPU. Non hai 282 GB: hai due volte 141 GB. Il conto si fa
     PER GPU, e cosa sta su quale GPU dipende dalla strategia (M18).

  2. CONTARE TUTTO NELLO STESSO DTYPE. "Training in bf16" descrive i pesi.
     Gli stati optimizer sono tipicamente fp32: 8 byte per parametro
     addestrabile, non 4. -> M05

  3. DARE I GRADIENTI A TUTTI I PARAMETRI. Con LoRA la base e' congelata:
     niente gradienti, niente stati optimizer. E' li' che sta il risparmio,
     non nei pesi (quelli restano tutti in memoria).

  4. DIMENTICARE CHE LA FORMULA SOTTOSTIMA. Mancano sempre: workspace
     temporanei dei kernel, frammentazione dell'allocatore, buffer di
     comunicazione, i logit (batch x seq x VOCABOLARIO, che con un
     vocabolario grande NON e' trascurabile).

  Regola pratica: se il conto dice che ci stai per un pelo, non ci stai.""")


sez("5. Adesso tocca a te")

print("""  Prima di lanciare qualunque addestramento, scrivi su carta:

    parametri totali          ______
    parametri addestrabili    ______   (con LoRA: quanti davvero?)
    byte/par pesi             ______
    byte/par gradienti        ______   (solo addestrabili)
    byte/par stati optimizer  ______   (solo addestrabili)
    copia master fp32?        si / no  (verificato, non assunto)
    -------------------------------
    statico per GPU           ______
    attivazioni stimate       ______
    -------------------------------
    totale                    ______   vs 141 GB

  Poi lancia, misura, e confronta. Lo scarto e' l'informazione utile:
  ti dice quale termine del tuo modello mentale e' sbagliato. -> M07""")

print()
