#!/usr/bin/env python3
"""
Lab 01.1 — Forme dei tensori.   Gira su CPU, nessuna GPU necessaria.

    python3 corso/labs/lab_01_1_forme.py

Fai la previsione PRIMA di lanciarlo (vedi M01). Serve lo scarto, non la precisione.
"""

import warnings

import torch


def sez(titolo):
    print(f"\n{'─' * 62}\n{titolo}\n{'─' * 62}")


sez("1. Le quattro forme fondamentali")

esempi = [
    ("scalare",  torch.tensor(3.14),          "la loss è sempre così"),
    ("vettore",  torch.zeros(768),            "i pesi di una normalizzazione"),
    ("matrice",  torch.zeros(4096, 4096),     "una matrice di attention"),
    ("3D",       torch.zeros(8, 512, 4096),   "attivazioni: 8 frasi x 512 token x 4096"),
]
for nome, t, nota in esempi:
    print(f"{nome:9} shape={str(list(t.shape)):22} dim={t.dim()}  numel={t.numel():>12,}   {nota}")


sez("2. La regola del prodotto:  [a,b] @ [b,c] -> [a,c]")

a = torch.zeros(3, 4)
b = torch.zeros(4, 2)
print(f"a={list(a.shape)}  b={list(b.shape)}  ->  a@b = {list((a @ b).shape)}")
print("Le due dimensioni interne (4 e 4) si saldano e spariscono.")

try:
    torch.zeros(3, 4) @ torch.zeros(5, 2)
except RuntimeError as e:
    print(f"\n[3,4] @ [5,2] fallisce subito, ed e' una buona notizia:\n  {e}")


sez("3. Byte: il mattone dell'aritmetica della memoria")

a = torch.zeros(3, 4)
print(f"dtype           = {a.dtype}")
print(f"numel()         = {a.numel()}          (quanti numeri)")
print(f"element_size()  = {a.element_size()}           (byte per numero)")
print(f"totale          = {a.numel() * a.element_size()} byte")

print("\nLo stesso conto su scala reale:")
for nome, n_par, byte_par in [
    ("modello 27B in float32", 27e9, 4),
    ("modello 27B in bfloat16", 27e9, 2),
]:
    gb = n_par * byte_par / 1e9
    print(f"  {nome:26} = {n_par:.2e} x {byte_par} byte = {gb:7.1f} GB   (una H200 ha 141 GB)")
print("\nSolo i pesi. Senza gradienti, senza optimizer, senza attivazioni. -> M06.")


sez("4. La dimensione batch viene trascinata")

x = torch.zeros(8, 512, 4096)
W = torch.zeros(4096, 11008)
print(f"x={list(x.shape)}  @  W={list(W.shape)}  ->  {list((x @ W).shape)}")
print("L'8 davanti resta intatto: e' per questo che raddoppiare il batch")
print("raddoppia la memoria delle attivazioni.")


sez("5. nn.Linear salva [out_features, in_features]")

layer = torch.nn.Linear(in_features=4096, out_features=11008)
print(f"nn.Linear(in=4096, out=11008)")
print(f"  .weight.shape = {list(layer.weight.shape)}   <- out, in: INVERTITO")
print(f"  .bias.shape   = {list(layer.bias.shape)}")
print("\nIn M04 leggerai 'mlp.up_proj.weight' con shape [11008, 4096]:")
print("significa 'prende 4096 numeri, ne produce 11008'.")


sez("6. Broadcasting: utile, e fabbrica di bug muti")

a = torch.zeros(3, 4)
print(f"[3,4] + [4]    -> {list((a + torch.zeros(4)).shape)}   (il vettore ripetuto su ogni riga)")
print(f"[3,4] + [3,1]  -> {list((a + torch.zeros(3, 1)).shape)}   (la colonna ripetuta su ogni colonna)")
try:
    a + torch.zeros(3)
except RuntimeError:
    print("[3,4] + [3]    -> ERRORE (3 non si allinea con 4, si confronta da destra)")

print("\nIl caso che fa male davvero:")
pred = torch.zeros(32, 1)
targ = torch.zeros(32)
diff = pred - targ
print(f"  pred={list(pred.shape)}  targ={list(targ.shape)}  ->  diff={list(diff.shape)}")
print(f"  Volevi 32 differenze. Ne hai {diff.numel()}.")

F = torch.nn.functional
print("\n  Quanto sei avvisato dipende da COME scrivi la loss:")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    F.mse_loss(pred, targ)
    print(f"    F.mse_loss(pred, targ)        -> {len(w)} warning")
    for x in w:
        print(f"       {x.category.__name__}: {str(x.message)[:88]}...")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = (pred - targ).pow(2).mean()
    print(f"    (pred - targ).pow(2).mean()   -> {len(w)} warning   <- MUTO")
print("    La funzione di libreria sa che sono 'input' e 'target', e avvisa.")
print("    L'aritmetica grezza no: li' il broadcasting e' legittimo.")
print("    In un fine-tuning reale la loss e' quasi sempre scritta a mano,")
print("    con maschere e pesi per token. Cioe': nel caso muto.")
print("\n    E comunque Python stampa ogni warning UNA volta: al passo 0,")
print("    dentro migliaia di righe di log, non lo vede nessuno.")

print("\n  ATTENZIONE: la difesa ovvia NON funziona.")
buona = F.mse_loss(torch.zeros(32), targ)
rotta = F.mse_loss(pred, targ)
print(f"    loss corretta  shape = {list(buona.shape)}  <- scalare")
print(f"    loss col bug   shape = {list(rotta.shape)}  <- scalare LO STESSO")
print("    La riduzione a media nasconde il bug invece di rivelarlo.")

print("\n  Cosa funziona davvero: controllare le shape PRIMA della riduzione.")
print(f"    reduction='none' sul caso rotto -> {list(F.mse_loss(pred, targ, reduction='none').shape)}")
print("    oppure: assert pred.shape == targ.shape")

print()
