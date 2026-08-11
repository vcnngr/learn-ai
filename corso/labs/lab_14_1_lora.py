#!/usr/bin/env python3
"""
Lab 14.1 — LoRA, QLoRA, DoRA: cosa cambia ciascun parametro.   CPU.

    python3 corso/labs/lab_14_1_lora.py

Ogni affermazione viene verificata numericamente. Il merge non e' spiegato:
e' dimostrato con max|diff|.
"""

import numpy as np


def sez(t):
    print(f"\n{'─' * 72}\n{t}\n{'─' * 72}")


rng = np.random.default_rng(0)

# ----------------------------------------------------------------------
sez("1. Quanti parametri aggiunge LoRA, contati")

def par_lora(shape, r):
    out, inn = shape
    return r * (inn + out)


# proporzioni di un denso ~27B
H, INTER, L, V = 5120, 13824, 46, 128000
MODULI = {
    "q_proj": (H, H), "k_proj": (H // 4, H), "v_proj": (H // 4, H), "o_proj": (H, H),
    "gate_proj": (INTER, H), "up_proj": (INTER, H), "down_proj": (H, INTER),
}
TOT = 27e9

print(f"  Modello di riferimento: hidden={H}, intermedio={INTER}, {L} strati\n")
print(f"  {'target_modules':<34} {'moduli/strato':>13} {'r=16':>12} {'% del tot':>10}")
scelte = [
    ("q,v (il default piu' comune)", ["q_proj", "v_proj"]),
    ("q,k,v,o (tutta l'attention)", ["q_proj", "k_proj", "v_proj", "o_proj"]),
    ("solo MLP", ["gate_proj", "up_proj", "down_proj"]),
    ("tutti e sette", list(MODULI)),
]
for nome, mods in scelte:
    p = sum(par_lora(MODULI[m], 16) for m in mods) * L
    print(f"  {nome:<34} {len(mods):>13} {p:>12,} {100 * p / TOT:>9.3f}%")

CASO_R = 32
p_caso = sum(par_lora(MODULI[m], CASO_R) for m in MODULI) * L
print(f"\n  {'CASO DI STUDIO del corso':<34} {len(MODULI):>13} "
      f"{p_caso:>12,} {100 * p_caso / TOT:>9.3f}%   <- r={CASO_R}")
print(f"""
  L'ultima riga e' la configurazione dell'adattatore che attraversa il
  corso: tutti e sette i moduli a r={CASO_R} (alpha={2 * CASO_R}). E' da QUI che esce
  la frazione di addestrabili usata da M06 e M17 per contare la memoria:
  {100 * p_caso / TOT:.3f}%, non un valore assunto. Se cambi r o target_modules,
  quella frazione cambia e i GB statici di M06 vanno rifatti:
    python3 lab_06_1_memoria.py --params 27e9 --method lora --lora-frac {p_caso / TOT:.5f}
  La configurazione sta in assets/profilo.js, sotto caso.lora.""")

print(f"\n  Formula: r x (in + out) per modulo, per strato.")
print(f"  Nota la riga 'solo MLP': l'MLP ha meno moduli dell'attention ma")
print(f"  matrici molto piu' grandi, quindi pesa di piu'. E' la stessa")
print(f"  asimmetria misurata in M04 su un checkpoint vero.")


# ----------------------------------------------------------------------
sez("2. rank e alpha: cosa cambia CIASCUNO")

print("""  LoRA calcola:   W_eff = W + (alpha / r) * B @ A

  Due parametri, due effetti DIVERSI:
    r      quanto e' espressivo l'aggiornamento (il rango della matrice)
    alpha  quanto forte e' l'aggiornamento (un fattore di scala)

  Il fattore e' alpha/r, non alpha. Conseguenza che sorprende:""")

r_vals = [4, 8, 16, 32, 64]
print(f"\n  {'r':>5} {'alpha':>7} {'scala alpha/r':>15}  effetto")
for r in r_vals:
    print(f"  {r:>5} {16:>7} {16 / r:>15.3f}  alpha FISSO: piu' r, piu' DEBOLE l'aggiornamento")
print()
for r in r_vals:
    print(f"  {r:>5} {2 * r:>7} {2:>15.3f}  alpha = 2r: scala COSTANTE, cambia solo l'espressivita'")

print("""
  Se aumenti r lasciando alpha fisso, stai facendo DUE cose insieme:
  aumenti l'espressivita' E riduci la scala. Se poi confronti i due run
  non sai a quale delle due attribuire la differenza.

  Convenzione diffusa: tenere alpha = 2r (o alpha = r) proprio per
  isolare l'effetto del rango. Non e' una regola magica: e' un modo di
  cambiare una variabile alla volta.""")


# ----------------------------------------------------------------------
sez("3. Cosa puo' e non puo' rappresentare un rango r")

W0 = rng.standard_normal((256, 256))
target = rng.standard_normal((256, 256)) * 0.1          # l'aggiornamento "ideale", rango pieno

U, S, Vh = np.linalg.svd(target)
print("  Un aggiornamento LoRA di rango r puo' rappresentare al PIU' una")
print("  matrice di rango r. Quanto bene approssima un bersaglio di rango")
print("  pieno? La miglior approssimazione possibile e' quella dell'SVD:\n")
print(f"  {'r':>5} {'errore relativo':>17} {'varianza spiegata':>19}")
tot_en = (S ** 2).sum()
for r in (1, 4, 16, 64, 128, 256):
    ric = (U[:, :r] * S[:r]) @ Vh[:r]
    err = np.linalg.norm(ric - target) / np.linalg.norm(target)
    print(f"  {r:>5} {err:>17.4f} {100 * (S[:r] ** 2).sum() / tot_en:>18.1f}%")

print("""
  Su un bersaglio CASUALE l'energia e' distribuita su tutti i ranghi, e
  un rango basso approssima male. Ma un aggiornamento utile di fine-tuning
  non e' casuale: se il compito e' ristretto, l'aggiornamento necessario
  tende ad avere pochi ranghi dominanti, ed e' l'ipotesi su cui LoRA si
  regge.

  ATTENZIONE: e' un'IPOTESI sul tuo compito, non un teorema. Se il tuo
  fine-tuning richiede un cambiamento diffuso - una lingua nuova, un
  dominio molto lontano - un rango basso non ce la fa, e nessun
  learning rate lo compensa. Il segnale osservabile: aumentando r la
  metrica migliora ancora invece di appiattirsi. -> M12 per misurarlo""")


# ----------------------------------------------------------------------
sez("4. Il merge, dimostrato")

IN, OUT, R, ALPHA = 128, 256, 8, 16
W = rng.standard_normal((OUT, IN)) * 0.05
A = rng.standard_normal((R, IN)) * 0.02
B = rng.standard_normal((OUT, R)) * 0.02                # dopo l'addestramento B non e' piu' zero
x = rng.standard_normal((32, IN))

scala = ALPHA / R
y_adapter = x @ W.T + scala * ((x @ A.T) @ B.T)
W_merged = W + scala * (B @ A)
y_merged = x @ W_merged.T

d = np.abs(y_adapter - y_merged).max()
print(f"  base + adattatore attivo   vs   pesi fusi")
print(f"    max|diff| = {d:.3e}   {'OK' if d < 1e-4 else 'DIVERSO'}")
print(f"    forma di W: {W.shape}, invariata dopo il merge: {W_merged.shape}")
print("""
  Il merge non e' un'approssimazione: in aritmetica esatta e' la stessa
  funzione, riscritta. Ma NON e' identico bit per bit: lo scarto stampato
  sopra non e' zero, perche' le due strade moltiplicano nello stesso
  ordine matematico e in un ordine di macchina diverso. La verifica
  corretta e' 'scarto entro una tolleranza dichiarata per questo dtype'
  (qui float64), non 'uguaglianza esatta'. In bf16 lo stesso scarto
  cresce di tre ordini di grandezza -> M20.

  Dopo il merge l'adattatore NON esiste piu' come oggetto separato -
  i suoi effetti sono dentro W.

  Conseguenza pratica: un modello fuso non si puo' piu' 'spegnere' ne'
  scambiare con un altro adattatore. Se ti serve poter alternare piu'
  adattatori sulla stessa base, NON fondere.""")


# ----------------------------------------------------------------------
sez("5. L'inizializzazione: perche' B parte da zero")

A0 = rng.standard_normal((R, IN)) * 0.02
B0 = np.zeros((OUT, R))
y_base = x @ W.T
y_init = x @ W.T + scala * ((x @ A0.T) @ B0.T)
print(f"  al passo 0, con B = 0:  max|diff| dalla base = "
      f"{np.abs(y_base - y_init).max():.3e}")
print("""
  Zero esatto. Il modello con adattatore appena inizializzato E' il modello
  base: e' quello che rende possibile il controllo di parita' di M02.

  Se A e B fossero ENTRAMBI zero, il gradiente sarebbe zero e l'adattatore
  non imparerebbe mai nulla. Se fossero entrambi casuali, il modello
  partirebbe gia' degradato. Uno casuale e uno a zero e' l'unica
  combinazione che da' sia partenza neutra sia gradiente non nullo.""")

# Gradienti calcolati a mano (niente autograd: serve solo algebra).
# L = media(y^2),  y = xW^T + s (xA^T) B^T
def gradienti(A_, B_):
    Z = x @ A_.T                       # [N, R]
    y = x @ W.T + scala * (Z @ B_.T)   # [N, OUT]
    dL_dy = 2 * y / y.size
    dB = scala * dL_dy.T @ Z           # [OUT, R]
    dZ = scala * dL_dy @ B_            # [N, R]
    dA = dZ.T @ x                      # [R, IN]
    return dA, dB

dA0, dB0 = gradienti(A0, B0)
print(f"  gradiente su B all'init: norma {np.linalg.norm(dB0):.4f}"
      f"  -> non nullo, l'adattatore impara")
print(f"  gradiente su A all'init: norma {np.linalg.norm(dA0):.4f}"
      f"  -> nullo finche' B e' 0, poi si sblocca")

dA1, dB1 = gradienti(A0, B)          # dopo qualche passo B non e' piu' zero
print(f"  con B non nullo:  grad A = {np.linalg.norm(dA1):.4f}   grad B ="
      f" {np.linalg.norm(dB1):.4f}")


# ----------------------------------------------------------------------
sez("6. QLoRA: quanto costa quantizzare la base")

def quant_blocchi(t, bit=4, blocco=64):
    """Quantizzazione simmetrica per blocchi (semplificata)."""
    f = t.flatten()
    pad = (-len(f)) % blocco
    f = np.concatenate([f, np.zeros(pad)])
    b = f.reshape(-1, blocco)
    sc = np.maximum(np.abs(b).max(1, keepdims=True), 1e-8)
    livelli = 2 ** (bit - 1) - 1
    q = np.clip(np.round(b / sc * livelli), -livelli, livelli)
    return ((q / livelli) * sc).flatten()[:t.size].reshape(t.shape)


Wbig = rng.standard_normal((1024, 1024)) * 0.05
print(f"  {'bit':>5} {'errore relativo':>17} {'byte/parametro':>16}")
for bit in (8, 4, 3):
    wq = quant_blocchi(Wbig, bit)
    err = np.linalg.norm(wq - Wbig) / np.linalg.norm(Wbig)
    # bit dati + una scala fp16 ogni 64 valori
    bpp = bit / 8 + 2 / 64
    print(f"  {bit:>5} {err:>17.4f} {bpp:>16.3f}")

xq = rng.standard_normal((16, 1024))
y_pieno = xq @ Wbig.T
y_q = xq @ quant_blocchi(Wbig, 4).T
print(f"\n  effetto sull'USCITA con 4 bit: errore relativo "
      f"{np.linalg.norm(y_q - y_pieno) / np.linalg.norm(y_pieno):.4f}")

print("""
  La quantizzazione della base introduce un errore che NON sparisce:
  l'adattatore lo compensa in parte, ma parte da un modello leggermente
  diverso da quello che avresti con la base piena.

  Il baratto e' esplicito: in M06 QLoRA portava lo statico da 55.9 a
  15.4 GB. Se quei 40 GB ti servono, l'errore e' il prezzo. Se NON ti
  servono, stai pagando un degrado per niente.

  Nota che le scale non sono gratis: a 4 bit con blocchi da 64 il costo
  reale e' 0.531 byte/parametro, non 0.5. Su 27 miliardi fa ~840 MB di
  sole scale.""")


# ----------------------------------------------------------------------
sez("7. DoRA: cosa cambia nella parametrizzazione")

print("""  LoRA aggiorna la matrice cosi' com'e':
      W_eff = W + (alpha/r) * B @ A

  DoRA la scompone prima in DIREZIONE e MAGNITUDINE, e le aggiorna
  separatamente: una norma per colonna diventa un parametro a se',
  e l'aggiornamento a basso rango agisce sulla sola direzione.

  Non e' 'LoRA con un altro rango': e' una parametrizzazione diversa,
  con un numero di parametri leggermente maggiore.""")

W2 = rng.standard_normal((OUT, IN)) * 0.05
m = np.linalg.norm(W2, axis=0, keepdims=True)               # magnitudine per colonna
d_dir = W2 / m                                  # direzione (colonne normalizzate)
print(f"\n  W {W2.shape} -> magnitudine {m.shape} + direzione {d_dir.shape}")
print(f"  ricostruzione m * direzione == W ?  max|diff| = "
      f"{np.abs(m * d_dir - W2).max():.3e}")
print(f"  parametri extra rispetto a LoRA: {m.size:,} (uno per colonna)")

print("""
  I dettagli implementativi (come si normalizza, cosa e' addestrabile,
  come si fonde) dipendono dalla libreria e dalla versione: verificali
  sul tuo stack invece di assumerli da questa pagina.

  Il punto che resta vero e' architetturale: DoRA cambia COSA viene
  parametrizzato, non solo quanto. Confrontare DoRA e LoRA "a parita' di
  rango" non e' un confronto a parita' di capacita'.""")


# ----------------------------------------------------------------------
sez("8. La domanda che conta per il tuo LoRA")

print("""  Con quello che hai misurato qui, questi sono i controlli concreti:

  1. Quanti parametri addestrabili hai davvero?
     Contali: r x (in+out) x moduli x strati. Se sono lo 0.01% del
     modello e il compito e' lontano dal dominio base, il rango puo'
     essere il vincolo.

  2. Hai cambiato r fra due esperimenti lasciando alpha fisso?
     Allora hai cambiato anche la scala, e il confronto non isola nulla.

  3. target_modules include l'MLP?
     Solo attention tocca una frazione molto piu' piccola del modello.

  4. Il controllo di parita' di M02 passa?
     Con B inizializzato a zero DEVE dare scarto nullo. Se non lo da',
     il problema e' la configurazione dell'adattatore, non i dati.

  5. Stai usando QLoRA senza averne bisogno?
     Se lo statico ci sta in memoria senza quantizzare, stai pagando
     un errore di quantizzazione in cambio di niente.""")

print()
