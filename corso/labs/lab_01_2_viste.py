#!/usr/bin/env python3
"""
Lab 01.2 — Viste contro copie.   Gira su CPU.

    python3 corso/labs/lab_01_2_viste.py

Una vista cambia i metadati. Una copia alloca byte nuovi.
Su un tensore da 3 elementi non cambia nulla. Su uno da 40 GB e' tutto.
"""

import time
import torch


def sez(titolo):
    print(f"\n{'─' * 62}\n{titolo}\n{'─' * 62}")


sez("1. Stesso blocco di byte, o no?")

a = torch.zeros(4096, 4096)
b = a.t()          # trasposta
c = a.view(-1)     # appiattito
d = a.clone()      # copia vera

mb = a.numel() * a.element_size() / 1e6
print(f"a  shape={list(a.shape)}  = {mb:.1f} MB")
for nome, t in [("a.t()", b), ("a.view(-1)", c), ("a.clone()", d)]:
    stessa = t.data_ptr() == a.data_ptr()
    print(f"  {nome:12} shape={str(list(t.shape)):16} stessi byte di a? {stessa}"
          f"   {'-> 0 byte nuovi' if stessa else f'-> {mb:.1f} MB nuovi'}")


sez("2. Stessi byte = modificarne uno modifica l'altro")

x = torch.zeros(2, 3)
v = x.t()            # vista
x[0, 0] = 99.0
print(f"x[0,0] = 99  ->  v[0,0] = {v[0, 0].item()}    (la vista vede la modifica)")

y = torch.zeros(2, 3)
k = y.clone()        # copia
y[0, 0] = 99.0
print(f"y[0,0] = 99  ->  k[0,0] = {k[0, 0].item()}     (la copia no: byte separati)")


sez("3. Gli stride: il meccanismo sotto")

a = torch.zeros(3, 4)
print(f"a      shape={list(a.shape)}  stride={a.stride()}  contiguo={a.is_contiguous()}")
print(f"a.t()  shape={list(a.t().shape)}  stride={a.t().stride()}  contiguo={a.t().is_contiguous()}")
print("\nstride = di quanti elementi saltare per avanzare di 1 lungo quell'asse.")
print("Trasporre = scambiare gli stride. I byte non si toccano.")
print("Non contiguo = alcune operazioni dovranno copiare per procedere.")


sez("4. Quanto costa davvero una copia")

for lato in (2048, 4096, 8192):
    t = torch.zeros(lato, lato)
    mb = t.numel() * t.element_size() / 1e6

    t0 = time.perf_counter()
    for _ in range(20):
        _ = t.t()
    dt_vista = (time.perf_counter() - t0) / 20 * 1e6      # microsecondi

    t0 = time.perf_counter()
    for _ in range(5):
        _ = t.clone()
    dt_copia = (time.perf_counter() - t0) / 5 * 1e3        # millisecondi

    print(f"[{lato},{lato}] = {mb:7.1f} MB   vista: {dt_vista:7.1f} us   copia: {dt_copia:7.2f} ms")

print("\nLa vista e' costante: non dipende dalla dimensione, non tocca i byte.")
print("La copia cresce con i byte: e' limitata dalla banda di memoria.")


sez("5. Il conto che rifarai su un modello vero")

for lato in (1024, 4096, 11008):
    t = torch.zeros(lato, 4096)
    print(f"  [{lato:>5}, 4096] float32 = {t.numel():>12,} elementi"
          f" x 4 byte = {t.numel() * 4 / 1e6:8.1f} MB")

print("\nUn modello e' qualche centinaio di questi. Sommarli e' M04.")
print()
