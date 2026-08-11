#!/usr/bin/env python3
"""
Lab 03.2 — Un addestramento completo, e perche' la training loss mente.  CPU.

    python3 corso/labs/lab_03_2_addestramento.py

Prima parte: recuperiamo coefficienti NOTI. Se il ciclo funziona, devono uscire.
Seconda parte: un modello che azzera la training loss e peggiora sul resto.
"""

import torch


def sez(titolo):
    print(f"\n{'─' * 68}\n{titolo}\n{'─' * 68}")


torch.manual_seed(0)

sez("1. Il ciclo, cinque passi, sempre gli stessi")

print("""    for batch in dati:
        pred = modello(batch.x)            # 1 avanti
        loss = criterio(pred, batch.y)     # 2 quanto sbaglia
        loss.backward()                    # 3 indietro: riempie i .grad
        optimizer.step()                   # 4 applica: w <- w - lr * grad
        optimizer.zero_grad()              # 5 azzera, o si accumulano

  Ogni addestramento che lancerai e' questo, con piu' strati intorno.""")


sez("2. Recuperare coefficienti NOTI: y = 3x + 2")

W_VERO, B_VERO, N = 3.0, 2.0, 200
x = torch.randn(N, 1)
y = W_VERO * x + B_VERO + torch.randn(N, 1) * 0.1        # rumore piccolo

w = torch.zeros(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
opt = torch.optim.SGD([w, b], lr=0.1)

print(f"  parto da w = 0, b = 0.  Verita': w = {W_VERO}, b = {B_VERO}\n")
print("  Ogni riga e' UNO stato: loss e pesi misurati PRIMA dello stesso step.")
print(f"\n  {'passo':>6}  {'loss':>10}  {'w':>8}  {'b':>8}")
for passo in range(201):
    pred = x @ w + b
    loss = ((pred - y) ** 2).mean()
    # fotografia dello stato corrente, prima di modificarlo
    stato = (loss.item(), w.item(), b.item())
    loss.backward()
    opt.step()
    opt.zero_grad()
    if passo % 40 == 0:
        print(f"  {passo:>6}  {stato[0]:>10.5f}  {stato[1]:>8.4f}  {stato[2]:>8.4f}")

print(f"\n  Trovati:  w = {w.item():.4f}   b = {b.item():.4f}")
print(f"  Veri:     w = {W_VERO}      b = {B_VERO}")
print(f"  Errore:   {abs(w.item() - W_VERO):.4f}      {abs(b.item() - B_VERO):.4f}")
print("\n  La loss non arriva a zero: c'e' rumore nei dati che nessun modello")
print("  puo' spiegare. Una loss che va a zero su dati rumorosi e' un allarme,")
print("  non un successo - lo vedi nella sezione 4.")


sez("3. Lo stesso ciclo senza zero_grad()")

w2 = torch.zeros(1, 1, requires_grad=True)
b2 = torch.zeros(1, requires_grad=True)
opt2 = torch.optim.SGD([w2, b2], lr=0.1)

storia = []
print(f"  {'passo':>6}  {'loss':>14}  {'|grad w|':>14}")
for passo in range(40):
    loss = ((x @ w2 + b2 - y) ** 2).mean()
    loss.backward()
    gn = w2.grad.abs().item()
    opt2.step()
    # zero_grad() OMESSO di proposito
    storia.append(loss.item())
    if passo < 8 or passo % 8 == 0:
        print(f"  {passo:>6}  {loss.item():>14.4f}  {gn:>14.4f}")

print(f"\n  Dopo 40 passi:  min {min(storia):.3f}   max {max(storia):.3f}")
print(f"  Confronto: col ciclo corretto la loss era {0.01124:.5f} e ferma.")

print("\n  ATTENZIONE al sintomo, perche' NON e' quello che ti aspetteresti:")
print("   * la loss non sale in modo monotono: OSCILLA, e continua a oscillare")
print("   * non esplode a NaN: resta in un intervallo, all'infinito")
print("   * a volte tocca valori BASSI (qui 0.04) - se guardi il log nel")
print("     momento sbagliato sembra che stia funzionando benissimo")
print("\n  In QUESTO esperimento il segno distintivo e' la mancata convergenza.")
print("  Ma non generalizzarlo a sintomo universale: con altri modelli, altri")
print("  learning rate o altri dati, l'accumulo puo' divergere a NaN o saturare.")
print("  Cio' che l'accumulo produce SEMPRE e' un aggiornamento piu' grande del")
print("  dovuto; come si manifesti nei log, no.")
print("\n  La verifica che non dipende dal sintomo: guarda il codice, oppure")
print("  stampa la norma del gradiente prima e dopo l'azzeramento.")
print("  Costa meno che indovinare dai log.")


sez("4. La training loss mente: memorizzazione")

# pochi punti, modello molto flessibile
N_TRAIN, GRADO = 12, 15
xt = torch.linspace(-1, 1, N_TRAIN).unsqueeze(1)
yt = torch.sin(3 * xt) + torch.randn(N_TRAIN, 1) * 0.1
xv = torch.linspace(-1, 1, 200).unsqueeze(1)              # tenuti da parte
yv = torch.sin(3 * xv)

def caratteristiche(x):
    return torch.cat([x ** k for k in range(GRADO + 1)], dim=1)

Ft, Fv = caratteristiche(xt), caratteristiche(xv)
theta = torch.zeros(GRADO + 1, 1, requires_grad=True)
opt3 = torch.optim.Adam([theta], lr=0.05)

print(f"  {N_TRAIN} punti di addestramento, polinomio di grado {GRADO}")
print(f"  ({GRADO + 1} parametri per {N_TRAIN} punti: piu' parametri che dati)\n")
print("  Entrambe le loss sono misurate sullo STESSO stato, prima dello step.\n")
print(f"  {'passo':>7}  {'loss ADDESTRAMENTO':>19}  {'loss TENUTI DA PARTE':>21}")
traccia = []
for passo in range(20001):
    loss_t = ((Ft @ theta - yt) ** 2).mean()
    if passo % 4000 == 0:
        with torch.no_grad():                      # stesso stato di loss_t
            loss_v = ((Fv @ theta - yv) ** 2).mean()
        traccia.append((passo, loss_t.item(), loss_v.item()))
        print(f"  {passo:>7}  {loss_t.item():>19.6f}  {loss_v.item():>21.6f}")
    loss_t.backward()
    opt3.step()
    opt3.zero_grad()

t0, tf = traccia[0], traccia[-1]
migliore = min(traccia, key=lambda r: r[2])
print(f"\n  addestramento: {t0[1]:.6f} -> {tf[1]:.6f}"
      f"   = {t0[1] / tf[1]:.0f}x meglio")
print(f"  tenuti da parte: minimo {migliore[2]:.6f} al passo {migliore[0]},"
      f" finale {tf[2]:.6f}")
print(f"                   = {tf[2] / migliore[2]:.1f}x PEGGIO del suo minimo")

print("\n  La colonna di sinistra scende sempre. Quella di destra no.")
print("  Guardando SOLO la training loss questo addestramento sembra ottimo.")
print("\n  E' il motivo per cui la training loss non e' una misura di qualita':")
print("  dice quanto il modello si adatta ai dati che ha gia' visto, che e'")
print("  esattamente cio' che NON ti interessa. -> M12")

print()
