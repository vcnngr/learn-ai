#!/usr/bin/env python3
"""
Lab 15.1 — Cosa rende un addestramento riproducibile.   CPU.

    python3 corso/labs/lab_15_1_riproducibile.py

Ogni fonte di non-determinismo viene isolata e MISURATA una alla volta.
"""

import hashlib
import json
import os
import platform
import random

import torch
import torch.nn as nn


def sez(t):
    print(f"\n{'─' * 72}\n{t}\n{'─' * 72}")


# ----------------------------------------------------------------------
def addestra(seed_pesi, seed_dati, passi=60, mescola=True):
    """Ritorna i pesi finali. Due semi SEPARATI, per isolarne gli effetti."""
    torch.manual_seed(seed_pesi)
    m = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 1))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)

    g = torch.Generator().manual_seed(12345)          # dati sempre uguali
    X = torch.randn(256, 16, generator=g)
    Y = X @ torch.arange(16.0).unsqueeze(1) / 16 + 0.1

    r = random.Random(seed_dati)
    ordine = list(range(0, 256, 32))
    for _ in range(passi):
        if mescola:
            r.shuffle(ordine)
        for i in ordine:
            loss = ((m(X[i:i + 32]) - Y[i:i + 32]) ** 2).mean()
            loss.backward(); opt.step(); opt.zero_grad()
    return torch.cat([p.detach().flatten() for p in m.parameters()])


def scarto(a, b):
    return (a - b).abs().max().item()


sez("1. Cosa cambia se cambia UN SOLO ingrediente")

base = addestra(seed_pesi=0, seed_dati=0)
casi = [
    ("tutto identico",                addestra(0, 0)),
    ("stesso seme, seconda esecuzione", addestra(0, 0)),
    ("seme dei PESI diverso",         addestra(1, 0)),
    ("ordine dei DATI diverso",       addestra(0, 1)),
]
print(f"  {'variante':<34} {'max|scarto| dai pesi base':>28}")
for nome, w in casi:
    print(f"  {nome:<34} {scarto(base, w):>28.6e}")

print("""
  Le prime due righe sono ZERO: qui, su CPU e con tutti i semi fissati,
  due esecuzioni danno pesi identici bit per bit. La riproducibilita' NON
  e' un ideale irraggiungibile - e' una proprieta' che ottieni se
  controlli gli ingressi.

  ATTENZIONE al limite: questo vale per QUESTO ambiente. Su GPU restano
  kernel non deterministici, e anche forzando le varianti deterministiche
  contano versione, hardware e backend. Il seme e' condizione necessaria,
  non sufficiente. -> sezione 5

  Le altre due mostrano che due semi DIVERSI producono modelli diversi.
  Ovvio detto cosi'. Il punto e' che sono DUE semi, non uno: se ne fissi
  solo uno, credi di aver reso deterministico un run che non lo e'.""")


sez("2. L'errore piu' comune: fissare un seme solo")

print("  Fisso solo il seme dei pesi, lascio libero l'ordine dei dati:\n")
print(f"  {'esecuzione':<20} {'scarto dalla prima':>22}")

# Il seme dei DATI e' "libero" nel senso che cambia a ogni esecuzione del
# ciclo - ma lo estraggo da un generatore seminato, non da quello globale.
# Altrimenti questo lab, proprio nel modulo sulla riproducibilita', non
# sarebbe riproducibile e i numeri stampati qui non tornerebbero mai.
semi_liberi = random.Random(0)
prima = addestra(7, semi_liberi.randrange(10**6))
for i in range(3):
    w = addestra(7, semi_liberi.randrange(10**6))
    print(f"  {'#' + str(i + 1):<20} {scarto(prima, w):>22.6e}")

print("""
  Stesso torch.manual_seed(), risultati diversi. Chi vede questo di solito
  conclude 'il training e' intrinsecamente non deterministico'.

  Non lo e'. Semplicemente l'ordine dei dati e' una SECONDA sorgente di
  casualita', spesso gestita da un generatore diverso (quello del
  DataLoader, o del sampler). Va fissato esplicitamente.""")


sez("3. L'inventario completo di cosa va registrato")

MANIFEST = {
    "modello_base":      {"nome": "org/modello", "revisione": "sha-esatto"},
    "dataset":           {"percorso": "dati/train.jsonl", "sha256": None, "righe": None},
    "tokenizer":         {"nome": "org/modello", "revisione": "sha-esatto",
                          "chat_template_sha256": None},
    "adattatore":        {"tipo": "lora", "r": 16, "alpha": 32,
                          "target_modules": ["q_proj", "v_proj"], "dropout": 0.0},
    "ottimizzazione":    {"optimizer": "adamw", "lr": 1e-4, "scheduler": "cosine",
                          "warmup": 100, "clip_norm": 1.0},
    "batch":             {"micro": 4, "accumulazioni": 8, "gpu": 2, "effettivo": 64},
    "sequenza":          {"max_len": 2048, "packing": False, "troncamento": "destra"},
    "precisione":        {"pesi": "bf16", "stati_optimizer": "fp32", "tf32": None},
    "semi":              {"pesi": 0, "ordine_dati": 0, "dropout": 0},
    "ambiente":          {"python": platform.python_version(),
                          "torch": torch.__version__,
                          "piattaforma": platform.platform(),
                          "cuda": torch.version.cuda},
    "loss":              {"maschera": "solo_risposta", "ignore_index": -100},
    "inferenza_eval":    {"temperatura": 0.0, "max_new_tokens": 256,
                          "template": "quello del tokenizer"},
}

print("  Un manifest che rende un run rifacibile deve contenere almeno questo:\n")
for k, v in MANIFEST.items():
    print(f"  {k:<18} {json.dumps(v, ensure_ascii=False)[:76]}")

mancanti = [f"{k}.{kk}" for k, v in MANIFEST.items()
            if isinstance(v, dict) for kk, vv in v.items() if vv is None]
print(f"\n  Campi ancora da riempire in questo esempio: {len(mancanti)}")
for m in mancanti:
    print(f"    - {m}")
print("""
  Nota quali sono: gli SHA dei dati e del template, e il flag tf32.
  Sono esattamente quelli che nessuno registra e che poi rendono
  impossibile capire perche' due run 'identici' differiscono. -> M02, M05""")


sez("4. Perche' serve l'impronta dei DATI, non il percorso")

d1 = b'{"prompt":"A","risposta":"B"}\n{"prompt":"C","risposta":"D"}\n'
d2 = b'{"prompt":"A","risposta":"B"}\n{"prompt":"C","risposta":"D "}\n'   # uno spazio
for nome, d in [("dataset originale", d1), ("dopo una modifica invisibile", d2)]:
    print(f"  {nome:<32} sha256 {hashlib.sha256(d).hexdigest()[:16]}...  "
          f"{len(d)} byte")
print(f"\n  differiscono di UN carattere -> impronte completamente diverse.")
print("""
  'dati/train.jsonl' non identifica niente: il file cambia e il nome resta.
  L'impronta si', ed e' la differenza fra 'ho rifatto lo stesso esperimento'
  e 'credo di aver rifatto lo stesso esperimento'.""")


sez("5. Il determinismo completo costa, e va deciso")

print("""  Sui kernel GPU esistono operazioni non deterministiche: due esecuzioni
  identiche possono differire di poco per l'ordine delle riduzioni in
  parallelo. Si possono forzare varianti deterministiche:

      torch.use_deterministic_algorithms(True)
      torch.backends.cudnn.deterministic = True
      # piu' variabili d'ambiente specifiche della libreria

  COSTO: alcune operazioni diventano piu' lente, e alcune sollevano
  un errore invece di procedere (non esiste variante deterministica).
  Il rallentamento dipende dal modello e dalle operazioni usate:
  misuralo, non assumerlo.

  QUANDO SERVE DAVVERO:
    - stai facendo un CONFRONTO DI PARITA' (M02): li' serve, altrimenti
      il rumore numerico si confonde con il difetto che stai cercando
    - stai inseguendo un bug che appare a intermittenza

  QUANDO NON SERVE:
    - un addestramento di produzione. Ti basta poter RIFARE il run in
      modo statisticamente equivalente, non bit per bit. Registra i semi
      e il manifest, e accetta lo scarto numerico residuo.

  Confondere i due casi porta a pagare un rallentamento permanente per
  una garanzia che non ti serviva.""")


sez("6. La checklist prima di lanciare")

CHECK = [
    ("il manifest e' scritto su file, non nella memoria di chi lancia", True),
    ("sha256 del dataset registrato", None),
    ("revisione esatta del modello base (non 'main')", None),
    ("template renderizzato salvato come testo, non solo il nome", None),
    ("semi fissati: pesi, ordine dati, dropout", None),
    ("baseline misurata PRIMA (M12) sullo stesso insieme di eval", None),
    ("controllo di parita' a passo zero eseguito (M02)", None),
    ("maschera della loss ispezionata token per token (M10)", None),
]
print("  Otto voci. Le ultime tre sono quelle che questo corso ha aggiunto\n"
      "  e che di solito non compaiono in nessuna checklist:\n")
for testo, stato in CHECK:
    print(f"    [{'x' if stato else ' '}] {testo}")

print("""
  Nessuna di queste costa piu' di dieci minuti. Insieme sono la
  differenza fra un esperimento e un aneddoto: se il run rende poco,
  con il manifest puoi cercare la causa; senza, puoi solo rifarlo
  diverso e sperare.""")

print()
