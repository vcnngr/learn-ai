#!/usr/bin/env python3
"""
Lab 08.1 — Optimizer: cosa tengono in memoria e cosa cambiano.   CPU.

    python3 corso/labs/lab_08_1_optimizer.py

Tutto misurato: gli stati vengono contati dall'optimizer vero, non stimati.
"""

import math

import torch
import torch.nn as nn


def sez(t):
    print(f"\n{'─' * 72}\n{t}\n{'─' * 72}")


torch.manual_seed(0)

# ----------------------------------------------------------------------
sez("1. Quanti byte tiene un optimizer, CONTATI (non stimati)")

def misura_stati(costruttore, nome):
    m = nn.Linear(2000, 2000, bias=False)          # 4.000.000 parametri
    P = sum(p.numel() for p in m.parameters())
    opt = costruttore(m.parameters())
    m(torch.randn(8, 2000)).pow(2).mean().backward()
    opt.step()                                      # gli stati nascono qui
    byte, tensori, nomi = 0, 0, set()
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v) and v.numel() > 1:
                byte += v.numel() * v.element_size()
                tensori += 1
                nomi.add(k)
    return P, byte, tensori, sorted(nomi)

print(f"  {'optimizer':<22} {'stati/par':>10} {'byte/par':>9} {'tensori':>8}  nomi")
for nome, cost in [
    ("SGD (senza momento)", lambda p: torch.optim.SGD(p, lr=0.01)),
    ("SGD + momento",       lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9)),
    ("Adam",                lambda p: torch.optim.Adam(p, lr=0.01)),
    ("AdamW",               lambda p: torch.optim.AdamW(p, lr=0.01)),
]:
    P, byte, nt, nomi = misura_stati(cost, nome)
    print(f"  {nome:<22} {nt:>10} {byte / P:>9.1f} {nt:>8}  {', '.join(nomi) if nomi else '-'}")

print(f"\n  Modello di prova: {4_000_000:,} parametri, fp32 (4 byte l'uno).")
print("  AdamW tiene DUE tensori per parametro: exp_avg (il momento) e")
print("  exp_avg_sq (il momento quadratico).")

print("\n  MA il dtype degli stati NON e' una costante: SEGUE quello dei pesi.")
print("  Non lo assumo, lo chiedo all'optimizer:\n")
print(f"  {'dtype dei pesi':<18} {'dtype degli stati':>18} {'byte/parametro':>16}")
for dt in (torch.float32, torch.bfloat16, torch.float16):
    mm = nn.Linear(256, 256, bias=False).to(dt)
    oo = torch.optim.AdamW(mm.parameters(), lr=1e-3)
    mm(torch.randn(4, 256, dtype=dt)).pow(2).mean().backward()
    oo.step()
    dts = {v.dtype for st in oo.state.values() for k, v in st.items()
           if torch.is_tensor(v) and v.numel() > 1}
    byte = sum(v.element_size() for st in oo.state.values()
               for k, v in st.items() if torch.is_tensor(v) and v.numel() > 1)
    print(f"  {str(dt).replace('torch.',''):<18} "
          f"{', '.join(str(d).replace('torch.','') for d in dts):>18} {byte:>16}")

print("""
  Con pesi bf16 l'AdamW di PyTorch crea stati bf16: 4 byte per parametro,
  NON 8. E non crea nessuna copia master fp32 da solo.

  Gli 8 byte/parametro e la copia master appartengono a una configurazione
  di MIXED PRECISION costruita apposta - quella che molti framework di
  addestramento impostano per te - non ad 'AdamW' come algoritmo.

  Conseguenza per il conto di M06: byte_stati e byte_master NON sono
  costanti da memorizzare. Sono cose da MISURARE sul proprio stack, con
  esattamente il codice qui sopra.""")


# ----------------------------------------------------------------------
sez("2. Perche' Adam esiste: la stessa discesa su una valle stretta")

def valle(p):
    """Superficie molto piu' ripida lungo x che lungo y (mal condizionata)."""
    return 10.0 * p[0] ** 2 + 0.1 * p[1] ** 2

print("  f(x,y) = 10x^2 + 0.1y^2   -> pendenza 100 volte diversa fra i due assi")
print("  Partenza (1, 1). Obiettivo (0, 0). 60 passi.\n")
print(f"  {'optimizer':<22} {'lr':>7} {'x finale':>12} {'y finale':>12} {'distanza':>11}")
for nome, cost, lr in [
    ("SGD",           lambda p, lr: torch.optim.SGD(p, lr=lr), 0.05),
    ("SGD + momento",  lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9), 0.05),
    ("Adam",          lambda p, lr: torch.optim.Adam(p, lr=lr), 0.05),
]:
    p = torch.tensor([1.0, 1.0], requires_grad=True)
    opt = cost([p], lr)
    for _ in range(60):
        loss = valle(p)
        loss.backward()
        opt.step()
        opt.zero_grad()
    d = p.detach().norm().item()
    print(f"  {nome:<22} {lr:>7} {p[0].item():>12.6f} {p[1].item():>12.6f} {d:>11.6f}")

print("\n  SGD e' costretto a un lr piccolo dall'asse RIPIDO, e su quello PIATTO")
print("  non si muove quasi. Adam divide il passo per la scala del gradiente di")
print("  ciascun parametro: avanza su entrambi gli assi. E' quello che comprano")
print("  gli stati in piu' per parametro.")


# ----------------------------------------------------------------------
sez("3. Gradient accumulation: equivalente a un batch grande?")

def addestra(micro_batch, accum, lr, dividi, passi=40, seed=1):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(16, 64), nn.GELU(), nn.Linear(64, 1))
    opt = torch.optim.SGD(m.parameters(), lr=lr)
    torch.manual_seed(99)
    X = torch.randn(256, 16)
    Y = X @ torch.arange(16.).unsqueeze(1) / 16 + 0.1
    for passo in range(passi):
        opt.zero_grad()
        for a in range(accum):
            i0 = ((passo * accum + a) * micro_batch) % 128
            l = ((m(X[i0:i0 + micro_batch]) - Y[i0:i0 + micro_batch]) ** 2).mean()
            # la media va presa sul batch EFFETTIVO, non sul micro-batch:
            (l / accum if dividi else l).backward()
        opt.step()
    f = ((m(X) - Y) ** 2).mean().item()
    return "DIVERGE" if (f != f or f > 1e4) else f"{f:.5f}"

print("  Stessa quantita' di dati per passo, tre modi di ottenerla.\n")
print(f"  {'lr':>6}  {'batch 32':>12}  {'8x4 CON /accum':>16}  {'8x4 SENZA /accum':>18}")
for lr in (0.05, 0.1, 0.2, 0.5):
    print(f"  {lr:>6}  {addestra(32, 1, lr, True):>12}"
          f"  {addestra(8, 4, lr, True):>16}  {addestra(8, 4, lr, False):>18}")

print("""
  Due letture, entrambe importanti:

  1. Le prime due colonne sono IDENTICHE a ogni lr. L'accumulazione fatta
     bene e' equivalente a un batch grande: e' il modo standard di simulare
     batch che in memoria non ci starebbero.

     SOTTO QUALI CONDIZIONI: qui c'e' SGD, una loss additiva sugli esempi,
     micro-batch tutti da 8 e nessuna operazione che dipenda dal batch.
     L'equivalenza si rompe con normalizzazioni o stati dipendenti dal
     batch, con dropout o altro consumo di RNG, e soprattutto con sequenze
     MASCHERATE: li' il denominatore giusto e' il numero totale di token
     supervisionati del batch effettivo, non 'accum'.

  2. La terza colonna e' un altro addestramento. Dimenticare la divisione
     somma i gradienti invece di mediarli: con SGD e' come moltiplicare il
     learning rate per 'accum'. Con AdamW NO - il passo divide per la
     radice di exp_avg_sq, quindi il fattore viene in gran parte
     normalizzato via, e quel che resta passa da epsilon, clipping e
     weight decay. In generale e' un RISCALAMENTO DEL GRADIENTE, non del
     learning rate. A lr basso puo' persino sembrare che vada meglio
     (converge prima). A lr=0.2 la config corretta converge e quella
     sbagliata DIVERGE - stesso codice, una riga di differenza.

  E' un errore silenzioso: nessun messaggio, nessun avviso. Riscala il
  gradiente di un fattore che non hai scelto. -> M21""")


# ----------------------------------------------------------------------
sez("4. Warmup: perche' i primi passi sono diversi dagli altri")

def corri(lr, warmup, passi=60, seed=3, width=256):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(32, width), nn.GELU(),
                      nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    torch.manual_seed(7)
    X, Y = torch.randn(128, 32), torch.randn(128, 1)
    storia = []
    for p in range(passi):
        for g in opt.param_groups:
            g["lr"] = lr * min(1.0, (p + 1) / warmup) if warmup else lr
        loss = ((m(X) - Y) ** 2).mean()
        storia.append(loss.item())
        loss.backward()
        opt.step(); opt.zero_grad()
    return storia

print("  La loss FINALE non e' la metrica giusta: guarda il PICCO iniziale.\n")
print(f"  {'lr':>6}  {'senza warmup: picco':>21}  {'con warmup(15): picco':>23}")
for lr in (0.05, 0.1, 0.2, 0.5):
    a, b = corri(lr, 0), corri(lr, 15)
    print(f"  {lr:>6}  {max(a):>21.2f}  {max(b):>23.2f}")

print("""
  Partenza a loss ~1.0 in tutti i casi. Senza warmup, la loss SCHIZZA
  prima di riprendersi - e piu' alto il lr, piu' il picco e' violento.
  Con warmup i picchi restano ordini di grandezza piu' bassi.

  Il meccanismo: all'inizio gli stati di Adam sono VUOTI. Le sue stime
  della scala del gradiente sono costruite su uno o due campioni, quindi
  inaffidabili, e il passo che ne esce puo' essere enorme. Il warmup gli
  da' il tempo di riempirsi prima di fidarsi.

  Su un modello piccolo il picco si riassorbe. Su un modello grande quei
  primi passi possono spostare i pesi cosi' lontano da non tornare piu'
  indietro - e il run e' perso senza che il log mostri un errore.""")


# ----------------------------------------------------------------------
sez("5. Gradient clipping: cosa fa esattamente")

g = torch.tensor([3.0, 4.0])                     # norma 5
for soglia in (10.0, 5.0, 1.0):
    x = torch.zeros(2, requires_grad=True)
    x.grad = g.clone()
    prima = x.grad.norm().item()
    torch.nn.utils.clip_grad_norm_([x], soglia)
    print(f"  soglia {soglia:>5.1f}  norma prima {prima:.2f}  dopo {x.grad.norm().item():.4f}"
          f"   direzione invariata: {torch.allclose(x.grad / x.grad.norm(), g / g.norm())}")

print("\n  Il clipping RISCALA, non taglia componente per componente:")
print("  la DIREZIONE resta identica, cambia solo la lunghezza.")
print("  Serve contro i picchi isolati (un batch anomalo, un'esponenziale")
print("  della regola della catena - M02). Non corregge un lr sbagliato:")
print("  se clippa a ogni passo, il problema e' altrove.")

print()
