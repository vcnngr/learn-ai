#!/usr/bin/env python3
"""
Lab 04.1 — Aprire un modello e contarne i parametri a mano.   CPU.

    python3 corso/labs/lab_04_1_anatomia.py                  # checkpoint sintetico
    python3 corso/labs/lab_04_1_anatomia.py /path/al/modello  # un modello vero

Senza argomenti costruisce un checkpoint safetensors realistico in una
directory temporanea, cosi' il lab gira ovunque. Con un path, ispeziona
quello: funziona su qualunque directory contenente file .safetensors.

L'header si legge senza caricare i pesi: 8 byte + un JSON.
"""

import json
import struct
import sys
import tempfile
from pathlib import Path


def sez(titolo):
    print(f"\n{'─' * 70}\n{titolo}\n{'─' * 70}")


# ----------------------------------------------------------------------
# Lettura dell'header: nessuna dipendenza, nessun peso caricato in RAM
# ----------------------------------------------------------------------

def leggi_header(path):
    """Restituisce (header_dict, byte_header, dimensione_file)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]     # uint64 little-endian
        header = json.loads(f.read(n))
    return header, n, path.stat().st_size


BYTE_PER_DTYPE = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}


# ----------------------------------------------------------------------
# Checkpoint sintetico: proporzioni realistiche di un denso stile Llama
# ----------------------------------------------------------------------

def crea_checkpoint(dest):
    import torch
    from safetensors.torch import save_file

    cfg = dict(hidden=512, inter=1376, layers=4, vocab=8000,
               q_heads=8, kv_heads=2)
    hd = cfg["hidden"] // cfg["q_heads"]                  # head_dim = 64
    t = {}
    t["model.embed_tokens.weight"] = torch.zeros(cfg["vocab"], cfg["hidden"])
    for i in range(cfg["layers"]):
        p = f"model.layers.{i}."
        t[p + "self_attn.q_proj.weight"] = torch.zeros(cfg["q_heads"] * hd, cfg["hidden"])
        t[p + "self_attn.k_proj.weight"] = torch.zeros(cfg["kv_heads"] * hd, cfg["hidden"])
        t[p + "self_attn.v_proj.weight"] = torch.zeros(cfg["kv_heads"] * hd, cfg["hidden"])
        t[p + "self_attn.o_proj.weight"] = torch.zeros(cfg["hidden"], cfg["q_heads"] * hd)
        t[p + "mlp.gate_proj.weight"] = torch.zeros(cfg["inter"], cfg["hidden"])
        t[p + "mlp.up_proj.weight"] = torch.zeros(cfg["inter"], cfg["hidden"])
        t[p + "mlp.down_proj.weight"] = torch.zeros(cfg["hidden"], cfg["inter"])
        t[p + "input_layernorm.weight"] = torch.zeros(cfg["hidden"])
        t[p + "post_attention_layernorm.weight"] = torch.zeros(cfg["hidden"])
    t["model.norm.weight"] = torch.zeros(cfg["hidden"])
    # NIENTE lm_head: questo modello ha embedding legati (tied)
    t = {k: v.to(torch.bfloat16) for k, v in t.items()}
    save_file(t, str(dest / "model.safetensors"))
    (dest / "config.json").write_text(json.dumps({
        "hidden_size": cfg["hidden"], "intermediate_size": cfg["inter"],
        "num_hidden_layers": cfg["layers"], "vocab_size": cfg["vocab"],
        "num_attention_heads": cfg["q_heads"], "num_key_value_heads": cfg["kv_heads"],
        "tie_word_embeddings": True, "torch_dtype": "bfloat16",
    }, indent=2))
    return cfg


# ----------------------------------------------------------------------

arg = sys.argv[1] if len(sys.argv) > 1 else None
if arg:
    modello = Path(arg).expanduser()
    tmp = None
    print(f"Ispeziono: {modello}")
else:
    tmp = tempfile.TemporaryDirectory()
    modello = Path(tmp.name)
    cfg = crea_checkpoint(modello)
    print(f"Nessun path fornito: costruito un checkpoint sintetico in {modello}")
    print("Per usarne uno vero:  python3 lab_04_1_anatomia.py /path/al/modello")

shard = sorted(modello.glob("*.safetensors"))
if not shard:
    print(f"\nNessun file .safetensors in {modello}. Esco.")
    sys.exit(1)


sez("1. L'header: 8 byte di lunghezza, poi un JSON. I pesi restano su disco.")

# TUTTI gli shard, non solo il primo: un modello grande e' sempre spezzato
# in piu' file, e leggerne uno solo produce totali falsi.
per_shard = []
for s in shard:
    hh, nb, dm = leggi_header(s)
    per_shard.append((s, hh, nb, dm))

dim_tot = sum(x[3] for x in per_shard)
head_tot = sum(x[2] for x in per_shard)
n_tensori = sum(len([k for k in x[1] if k != "__metadata__"]) for x in per_shard)

if len(shard) > 1:
    print(f"  ATTENZIONE: il checkpoint e' SHARDATO in {len(shard)} file.")
    print(f"  Leggerne uno solo darebbe totali falsi. Li aggrego tutti.\n")
    print(f"  {'file':<44} {'byte':>16} {'tensori':>9}")
    for s, hh, nb, dm in per_shard:
        nt = len([k for k in hh if k != "__metadata__"])
        print(f"  {s.name:<44} {dm:>16,} {nt:>9}")
    idx = modello / "model.safetensors.index.json"
    print(f"\n  indice presente ({idx.name})? {'si' if idx.exists() else 'no'}")
    if idx.exists():
        m = json.loads(idx.read_text())
        print(f"  l'indice mappa {len(m.get('weight_map', {})):,} tensori sui rispettivi shard")
else:
    print(f"  file            {shard[0].name}   (shard unico)")

print(f"\n  dimensione totale   {dim_tot:,} byte")
print(f"  header totali       {head_tot:,} byte di JSON")
print(f"  tensori totali      {n_tensori}")

h0 = per_shard[0][1]
primo = next(k for k in h0 if k != "__metadata__")
print(f"\n  voce grezza dell'header, per '{primo}':")
print(f"    {json.dumps(h0[primo])}")
print("\n  dtype, shape, e [inizio, fine] come offset di byte nel blob. Nient'altro.")
print("  Nota: gli offset sono relativi al PROPRIO shard, non al modello intero.")


sez("2. Tutti i tensori: nome, forma, tipo, byte")

righe, tot_par, tot_byte = [], 0, 0
visti = {}
for s, hh, _, _ in per_shard:
    for nome, meta in hh.items():
        if nome == "__metadata__":
            continue
        if nome in visti:                      # stesso nome in due shard: anomalia
            print(f"  [!] '{nome}' compare in piu' shard ({visti[nome]} e {s.name})")
            continue
        visti[nome] = s.name
        shape = meta["shape"]
        par = 1
        for d in shape:
            par *= d
        bpp = BYTE_PER_DTYPE.get(meta["dtype"], 0)
        righe.append((nome, shape, meta["dtype"], par, par * bpp))
        tot_par += par
        tot_byte += par * bpp

righe.sort(key=lambda r: r[0])

mostra = righe if len(righe) <= 24 else righe[:12] + [None] + righe[-6:]
print(f"  {'nome':<44} {'forma':>16} {'dtype':>6} {'parametri':>12}")
for r in mostra:
    if r is None:
        print(f"  {'... (' + str(len(righe) - 18) + ' tensori omessi)':<44}")
        continue
    nome, shape, dt, par, _ = r
    n = nome if len(nome) <= 44 else "..." + nome[-41:]
    print(f"  {n:<44} {str(shape):>16} {dt:>6} {par:>12,}")


sez("3. Il conto a mano contro i byte sul disco")

somma = sum(r[4] for r in righe)
prefissi = 8 * len(per_shard)                  # un uint64 per ogni shard
residuo = dim_tot - somma - head_tot - prefissi
print(f"  parametri totali              {tot_par:>15,}")
print(f"  byte dei tensori (somma)      {somma:>15,}")
print(f"  + header JSON ({len(per_shard)} shard)      {head_tot:>15,}")
print(f"  + prefissi di lunghezza       {prefissi:>15,}   (8 byte per shard)")
print(f"  {'=' * 46}")
print(f"  atteso                        {somma + head_tot + prefissi:>15,}")
print(f"  dimensione reale sul disco    {dim_tot:>15,}")
print(f"  residuo non spiegato          {residuo:>15,}")
if residuo == 0:
    print("\n  Zero. Ogni file e' ESATTAMENTE prefisso + header + tensori.")
    print("  Nessuna compressione, nessun padding, nessun metadato nascosto.")
else:
    print(f"\n  {residuo:,} byte non spiegati: padding di allineamento fra i tensori.")
    print("  Normale. Quello che conta e' che sia piccolo rispetto al totale:")
    print(f"  qui e' il {100 * residuo / dim_tot:.4f}% del file.")
print("  I parametri logici spiegano i byte fisici.")


sez("4. Dove finiscono i parametri")

def gruppo(nome):
    if "embed" in nome:
        return "embedding"
    if "lm_head" in nome:
        return "testa di uscita"
    if "self_attn" in nome:
        return "attention"
    if "mlp" in nome:
        return "MLP"
    if "norm" in nome:
        return "normalizzazioni"
    return "altro"

agg = {}
for nome, _, _, par, byte in righe:
    g = gruppo(nome)
    a = agg.setdefault(g, [0, 0, 0])
    a[0] += par
    a[1] += byte
    a[2] += 1

print(f"  {'gruppo':<18} {'tensori':>8} {'parametri':>14} {'quota':>7}")
for g, (par, byte, n) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
    print(f"  {g:<18} {n:>8} {par:>14,} {100 * par / tot_par:>6.1f}%")

print("\n  In un modello vero l'MLP domina: e' tipicamente la quota maggiore.")
print("  Le normalizzazioni sono vettori, non matrici: quota trascurabile,")
print("  ma sono quelle che tengono stabile l'addestramento (M02).")


sez("5. Tied embeddings: quando lm_head non c'e'")

ha_head = any("lm_head" in r[0] for r in righe)
emb = next((r for r in righe if "embed" in r[0]), None)
print(f"  lm_head presente nel checkpoint?   {'si' if ha_head else 'NO'}")
if emb:
    print(f"  embed_tokens                       forma {emb[1]}, {emb[3]:,} parametri")
if not ha_head and emb:
    print(f"\n  Il modello ha comunque bisogno di proiettare a vocabolario in uscita.")
    print(f"  Riusa la matrice degli embedding: e' quello che si chiama tying.")
    print(f"  Conseguenza sul conteggio:")
    print(f"    parametri LOGICI (unici)         {tot_par:>14,}")
    print(f"    parametri se contassi la testa   {tot_par + emb[3]:>14,}")
    print(f"    differenza                       {emb[3]:>14,}  ({100 * emb[3] / tot_par:.1f}%)")
    print("\n  Ecco perche' il numero dichiarato e i byte sul disco possono non")
    print("  coincidere: dipende da cosa stai contando. Controlla sempre")
    print("  'tie_word_embeddings' nella config prima di fidarti di un totale.")

cfgp = modello / "config.json"
if cfgp.exists():
    c = json.loads(cfgp.read_text())
    print(f"\n  config.json dichiara: tie_word_embeddings = {c.get('tie_word_embeddings')}")


sez("6. Leggere le forme di attention: GQA")

qs = [r for r in righe if "q_proj" in r[0]]
ks = [r for r in righe if "k_proj" in r[0]]
if qs and ks:
    q, k = qs[0], ks[0]
    print(f"  q_proj.weight   forma {str(q[1]):>14}")
    print(f"  k_proj.weight   forma {str(k[1]):>14}")
    if q[1][0] != k[1][0]:
        rap = q[1][0] / k[1][0]
        print(f"\n  Le uscite differiscono di un fattore {rap:.0f}.")
        print(f"  NON e' un errore: e' Grouped-Query Attention. Piu' teste di query")
        print(f"  condividono le stesse teste di key/value. Qui {rap:.0f} query per gruppo.")
        print(f"  Riduce i parametri di K e V, e soprattutto la KV cache in inferenza.")
        print(f"  La dimensione di Q resta piena. -> M09")
    else:
        print("\n  Stessa forma: attention classica a teste multiple (MHA), niente GQA.")
else:
    print("  Nessun q_proj/k_proj riconosciuto: naming diverso da quello atteso.")
    print("  Non e' un problema del modello, e' che il naming NON e' universale.")

if tmp:
    tmp.cleanup()
print()
