#!/usr/bin/env python3
"""
Lab 03.1 — La discesa del gradiente, e la soglia esatta del learning rate.  CPU.

    python3 corso/labs/lab_03_1_discesa.py

Su f(x) = x^2 la soglia di divergenza si calcola ESATTAMENTE.
Prima la deriviamo, poi la misuriamo. Devono coincidere.
"""


def sez(titolo):
    print(f"\n{'─' * 68}\n{titolo}\n{'─' * 68}")


def f(x):
    return x ** 2


def grad(x):
    return 2 * x


sez("1. La regola di aggiornamento, un passo alla volta")

x, lr = 5.0, 0.1
print(f"  f(x) = x^2,  df/dx = 2x,  parto da x = {x},  lr = {lr}\n")
print(f"  {'passo':>5}  {'x':>10}  {'f(x)':>10}  {'gradiente':>10}  {'x - lr*grad':>12}")
for passo in range(6):
    g = grad(x)
    nuovo = x - lr * g
    print(f"  {passo:>5}  {x:>10.5f}  {f(x):>10.5f}  {g:>10.5f}  {nuovo:>12.5f}")
    x = nuovo

print("\n  Gradiente positivo -> x diminuisce. Il segno fa il lavoro da solo.")


sez("2. La soglia, derivata a mano")

print("""  x_nuovo = x - lr * 2x = x * (1 - 2*lr)

  Ogni passo moltiplica x per il fattore (1 - 2*lr). Quindi:

    |1 - 2*lr| < 1   ->  converge
    |1 - 2*lr| = 1   ->  oscilla per sempre, ampiezza costante
    |1 - 2*lr| > 1   ->  diverge

  Risolvendo:  0 < lr < 1  converge.  lr = 0.5 -> fattore 0, un solo passo.
  lr = 1 -> fattore -1, rimbalza. lr > 1 -> esplode.""")


sez("3. La misura: 40 passi da x = 5")

print(f"  {'lr':>6}  {'fattore':>9}  {'x finale':>16}  {'previsione':>14}  esito")
for lr in (0.01, 0.1, 0.5, 0.9, 1.0, 1.01, 1.1):
    fattore = 1 - 2 * lr
    x = 5.0
    esploso = False
    for _ in range(40):
        x = x - lr * grad(x)
        if abs(x) > 1e12:
            esploso = True
            break
    a = abs(fattore)
    previsione = "converge" if a < 1 else ("oscilla" if a == 1 else "diverge")
    if esploso:
        esito = "DIVERGE"
        xs = "overflow"
    else:
        xs = f"{x:.6g}"
        esito = "diverge" if abs(x) > 5 else ("oscilla" if abs(x) > 1e-6 and a == 1 else "converge")
    print(f"  {lr:>6.2f}  {fattore:>9.2f}  {xs:>16}  {previsione:>14}  {esito}")

print("\n  Previsione e misura coincidono su tutte le righe, soglia inclusa.")
print("  lr = 0.5 arriva a zero in UN passo: e' il fattore che vale esattamente 0.")


sez("4. Perche' 'lr troppo basso' non e' innocuo")

print(f"  {'lr':>7}  {'passi per |x| < 0.001':>22}  {'rapporto col precedente':>24}")
prec = None
for lr in (0.5, 0.4, 0.1, 0.01, 0.001):
    x, n = 5.0, 0
    while abs(x) > 1e-3 and n < 10_000_000:
        x = x - lr * grad(x)
        n += 1
    rap = f"x{n / prec:.1f}" if prec else "-"
    print(f"  {lr:>7.3f}  {n:>22,}  {rap:>24}")
    prec = n

print("\n  Confronta quanto COSTA una riduzione, rapportata a quanto e' grande:")
print("   * da 0.01 a 0.001: lr diviso per 10   -> passi x10.1")
print("     regime lontano dall'ottimo: costo inversamente proporzionale.")
print("   * da 0.5 a 0.4:    lr diviso per 1.25 -> passi x6")
print("     vicino all'ottimo la sensibilita' e' molto piu' alta: una")
print("     riduzione otto volte piu' piccola costa quasi quanto l'altra.")
print("\n  Su un addestramento vero non conosci l'ottimo e non puoi calcolarlo.")
print("  Ma sai che stare un po' sotto costa in proporzione, mentre stare un po'")
print("  sopra fa divergere: il rischio NON e' simmetrico. Per questo si tende")
print("  a cercare il piu' alto che resta stabile, non il piu' sicuro. -> M08")


sez("5. Cosa NON si generalizza da qui")

print("""  f(x) = x^2 e' convessa, a una variabile, con soglia esatta calcolabile.
  Una rete neurale non e' niente di tutto questo:

    - milioni di dimensioni, ognuna con curvatura diversa
    - la superficie cambia forma a ogni batch (i dati sono diversi)
    - nessuna soglia chiusa: il lr utile si trova sperimentalmente

  Cosa SI porta via, e vale ovunque:
    - troppo alto -> diverge, e lo vedi subito (loss a NaN o che sale)
    - troppo basso -> converge, ma il costo esplode in modo non lineare
    - esiste una soglia netta fra i due, non una transizione morbida

  Il warmup e gli schedule (M08) esistono perche' quella soglia si sposta
  durante l'addestramento.""")

print()
