#!/usr/bin/env python3
"""
Lab 05.1 — Dove finiscono i bit, e cosa si rompe.   CPU.

    python3 corso/labs/lab_05_1_dtype.py

Non "bf16 e' meglio": si guarda dove vanno i bit e cosa smette di funzionare.
"""

import struct

import torch


def sez(t):
    print(f"\n{'─' * 70}\n{t}\n{'─' * 70}")


DTYPES = [
    ("float32",  torch.float32,  1, 8, 23),
    ("bfloat16", torch.bfloat16, 1, 8, 7),
    ("float16",  torch.float16,  1, 5, 10),
]

sez("1. Come sono spesi i bit")

print(f"  {'dtype':>9} {'byte':>5} {'segno':>6} {'esponente':>10} {'mantissa':>9}   a cosa serve")
note = {
    "float32":  "riferimento",
    "bfloat16": "esponente di fp32, mantissa tagliata",
    "float16":  "esponente ridotto, mantissa piu' larga di bf16",
}
for nome, dt, s, e, m in DTYPES:
    print(f"  {nome:>9} {torch.finfo(dt).bits // 8:>5} {s:>6} {e:>10} {m:>9}   {note[nome]}")

print("\n  ESPONENTE = quanto grande o piccolo puoi essere  (il RANGE)")
print("  MANTISSA  = quante cifre significative hai        (la PRECISIONE)")
print("\n  bf16 e fp16 occupano gli stessi 2 byte, ma li spendono in modo opposto.")


sez("2. Range e precisione, misurati")

print(f"  {'dtype':>9} {'valore max':>14} {'min normale':>14} {'eps':>12}")
for nome, dt, *_ in DTYPES:
    fi = torch.finfo(dt)
    print(f"  {nome:>9} {fi.max:>14.4g} {fi.tiny:>14.4g} {fi.eps:>12.4g}")

rap = torch.finfo(torch.bfloat16).eps / torch.finfo(torch.float16).eps
print("\n  eps = il piu' piccolo passo relativo rappresentabile.")
print(f"  bf16 ha lo STESSO range di fp32 (esponente a 8 bit) ma eps {rap:.0f} volte")
print("  peggiore di fp16. fp16 e' piu' preciso, in un intervallo molto piu' stretto.")
print(f"  ({torch.finfo(torch.bfloat16).eps:.7f} contro {torch.finfo(torch.float16).eps:.7f})")


sez("3. Il punto di rottura di fp16")

print("  Un valore che in bf16 sta comodo, in fp16 non esiste:\n")
for v in (1e4, 6.5e4, 1e5, 1e8, 1e30):
    r = []
    for nome, dt, *_ in DTYPES:
        x = torch.tensor(v, dtype=torch.float32).to(dt).float().item()
        r.append(f"{nome}={'inf' if x == float('inf') else f'{x:.4g}'}")
    print(f"  {v:>8.0e}   {'   '.join(r)}")

print(f"\n  fp16 satura a {torch.finfo(torch.float16).max:.0f}. Sopra: infinito, e da li'")
print("  in poi ogni operazione produce NaN.")
print("\n  Perche' ti riguarda: durante il backward i gradienti possono superare")
print("  quella soglia. In fp16 diventano inf -> NaN -> run buttato. In bf16 no.")
print("  Per questo fp16 richiede il LOSS SCALING (moltiplicare la loss per un")
print("  fattore grande prima del backward) e bf16 no. Non e' che bf16 sia")
print("  'migliore': ha barattato precisione per range, e nel training il range conta di piu'.")


sez("4. Quanto si perde in precisione")

print("  Lo stesso numero, arrotondato a ciascun tipo:\n")
print(f"  {'valore esatto':>18} {'float32':>16} {'bfloat16':>16} {'float16':>16}")
for v in (3.14159265358979, 1.00390625, 0.1, 65.4321):
    riga = f"  {v:>18.14g}"
    for _, dt, *_ in DTYPES:
        riga += f" {torch.tensor(v, dtype=torch.float64).to(dt).double().item():>16.10g}"
    print(riga)

print("\n  bf16 tiene circa 3 cifre decimali significative. Sembra poco, e lo e'")
print("  per un calcolo scientifico. Per un peso di una rete basta: quello che")
print("  conta e' la direzione dell'aggiornamento, non la sua terza cifra.")


sez("5. Dove la bassa precisione si rompe DAVVERO: l'accumulazione")

N = 100_000
piccolo = 1e-3
print(f"  Sommo {N:,} volte il valore {piccolo}. Risultato esatto: {N * piccolo:.1f}\n")
print(f"  {'accumulatore':>14} {'risultato':>14} {'errore':>12}")
for nome, dt, *_ in DTYPES:
    acc = torch.zeros((), dtype=dt)
    passo = torch.tensor(piccolo, dtype=dt)
    for _ in range(N):
        acc = acc + passo
    got = acc.float().item()
    print(f"  {nome:>14} {got:>14.4f} {100 * abs(got - N * piccolo) / (N * piccolo):>11.1f}%")

print("\n  Il meccanismo: quando l'accumulatore diventa grande rispetto al passo,")
print("  sommare il passo non cambia piu' nulla - viene arrotondato via.")
print("  L'accumulo si FERMA, e piu' iteri piu' l'errore cresce.")
print("\n  Ecco perche' le riduzioni (somme, medie, norme) e gli stati")
print("  dell'optimizer si tengono in fp32 anche quando i pesi sono in bf16.")
print("  'Training in bf16' non vuol dire che TUTTO e' in bf16. -> M06")


sez("6. TF32 non e' un dtype di storage")

print("""  TF32 e' una MODALITA' di calcolo dei Tensor Core su input fp32:
  i valori restano fp32 in memoria, ma la moltiplicazione interna usa
  una mantissa ridotta. Non cambia quanti byte occupa un tensore.

  Non troverai mai un tensore "di tipo tf32": e' un interruttore.
  Verifica sul tuo stack come si attiva e se e' attivo per default -
  dipende dalla versione di PyTorch e cambia nel tempo:

      torch.backends.cuda.matmul.allow_tf32
      torch.backends.cudnn.allow_tf32

  Conseguenza pratica: due macchine con lo stesso codice e lo stesso
  dtype possono dare numeri leggermente diversi se questo flag differisce.
  E' una delle cause di un controllo di parita' che fallisce di poco (M02).""")

print()
