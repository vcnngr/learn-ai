#!/usr/bin/env python3
"""
Lab 02.1 — Calcolare una derivata senza sapere le formule.   CPU.

    python3 corso/labs/lab_02_1_derivata.py

Metodo: sposta x di pochissimo, guarda di quanto cambia y, dividi.
E' esattamente il conto che hai fatto a mano in M02.
"""

import math


def sez(titolo):
    print(f"\n{'─' * 66}\n{titolo}\n{'─' * 66}")


def derivata_numerica(f, x, h=1e-6):
    """Derivata centrata: piu' accurata di (f(x+h)-f(x))/h a parita' di h."""
    return (f(x + h) - f(x - h)) / (2 * h)


sez("1. y = x^2, il conto fatto a mano")

f = lambda x: x ** 2
x = 3.0
h = 0.001
print(f"  x         = {x}")
print(f"  y = x^2   = {f(x)}")
print(f"  x + {h}  -> y = {f(x + h)}")
print(f"  variazione di y      = {f(x + h) - f(x):.6f}")
print(f"  diviso {h}          = {(f(x + h) - f(x)) / h:.4f}")
print(f"\n  formula esatta 2x    = {2 * x}")


sez("2. La derivata dipende dal PUNTO, non solo dalla funzione")

print(f"  {'x':>6}  {'numerica':>10}  {'esatta 2x':>10}")
for x in (0.0, 1.0, 3.0, 10.0, -4.0):
    print(f"  {x:>6.1f}  {derivata_numerica(f, x):>10.4f}  {2 * x:>10.1f}")
print("\n  A x=0 la derivata e' 0: punto piatto, muoversi non cambia niente.")
print("  Segno negativo a x=-4: aumentando x, y scende.")


sez("3. Funzioni che incontrerai davvero")

funzioni = [
    ("x^2",      lambda x: x ** 2,                    lambda x: 2 * x,                          "il caso base"),
    ("e^x",      math.exp,                            math.exp,                                 "dentro softmax"),
    ("log(x)",   math.log,                            lambda x: 1 / x,                          "dentro cross-entropy"),
    ("1/(1+e-x)", lambda x: 1 / (1 + math.exp(-x)),   lambda x: (lambda s: s * (1 - s))(1 / (1 + math.exp(-x))), "sigmoide"),
]
x = 1.5
print(f"  valutate in x = {x}\n")
print(f"  {'funzione':>11}  {'numerica':>11}  {'esatta':>11}  nota")
for nome, fn, dfn, nota in funzioni:
    print(f"  {nome:>11}  {derivata_numerica(fn, x):>11.6f}  {dfn(x):>11.6f}  {nota}")


sez("4. La regola della catena: le sensibilita' si MOLTIPLICANO")

# y = (3x + 1)^2   ->   u = 3x+1,  y = u^2
u = lambda x: 3 * x + 1
y_da_u = lambda u: u ** 2
y = lambda x: y_da_u(u(x))

x = 2.0
du_dx = derivata_numerica(u, x)
dy_du = derivata_numerica(y_da_u, u(x))
dy_dx_catena = dy_du * du_dx
dy_dx_diretta = derivata_numerica(y, x)

print(f"  y = (3x+1)^2  in x = {x}")
print(f"    du/dx              = {du_dx:.4f}")
print(f"    dy/du              = {dy_du:.4f}")
print(f"    prodotto (catena)  = {dy_dx_catena:.4f}")
print(f"    derivata diretta   = {dy_dx_diretta:.4f}   <- coincidono")


sez("5. Perche' i gradienti esplodono o svaniscono")

print("  60 strati = 60 fattori moltiplicati fra loro.\n")
print(f"  {'fattore medio':>14}  {'^60':>14}  cosa vedi")
for k in (1.2, 1.1, 1.0, 0.9, 0.8):
    v = k ** 60
    if v > 100:
        nota = "gradienti enormi -> pesi via -> loss NaN"
    elif v < 0.01:
        nota = "gradienti nulli -> i primi strati non imparano"
    else:
        nota = "sano"
    print(f"  {k:>14.1f}  {v:>14.4f}  {nota}")

print("\n  Normalizzazione, collegamenti residui e gradient clipping esistono")
print("  per tenere quel prodotto vicino a 1. Non sono trucchi: sono difese")
print("  contro un'esponenziale. -> M08")

print()
