#!/usr/bin/env python3
"""
Lab 09.1 — Scrivere un transformer da zero e verificarlo contro il riferimento.  CPU.

    python3 corso/labs/lab_09_1_transformer.py

Ogni pezzo che scrivo viene CONFRONTATO numericamente con l'implementazione
di PyTorch. Se non coincide entro tolleranza, il lab lo dice.
"""

import math

import torch
import torch.nn.functional as F


def sez(t):
    print(f"\n{'─' * 72}\n{t}\n{'─' * 72}")


def confronta(nome, mio, rif, tol=1e-5):
    d = (mio - rif).abs().max().item()
    ok = d < tol
    print(f"  {nome:<44} max|diff| = {d:.3e}   {'OK' if ok else 'DIVERSO'}")
    return ok


torch.manual_seed(0)
B, S, H, TESTE = 2, 6, 64, 4
D = H // TESTE                                    # head_dim = 16

# ----------------------------------------------------------------------
sez("1. Attention: la formula, riga per riga")

print("""    punteggi = Q @ K.T / sqrt(d)      quanto ogni token guarda ogni altro
    pesi     = softmax(punteggi)      normalizzati a 1 per riga
    uscita   = pesi @ V               media pesata dei valori

  sqrt(d) NON e' cosmetico: senza, i punteggi crescono con d e il softmax
  satura. Lo misuriamo fra due sezioni.""")

q = torch.randn(B, TESTE, S, D)
k = torch.randn(B, TESTE, S, D)
v = torch.randn(B, TESTE, S, D)


def mia_attention(q, k, v, causale=False):
    punteggi = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if causale:
        S_ = q.shape[-2]
        maschera = torch.triu(torch.ones(S_, S_, dtype=torch.bool), diagonal=1)
        punteggi = punteggi.masked_fill(maschera, float("-inf"))
    return punteggi.softmax(-1) @ v


print()
confronta("attention piena  vs  F.scaled_dot_product_attention",
          mia_attention(q, k, v), F.scaled_dot_product_attention(q, k, v))
confronta("attention causale vs  ... (is_causal=True)",
          mia_attention(q, k, v, causale=True),
          F.scaled_dot_product_attention(q, k, v, is_causal=True))


# ----------------------------------------------------------------------
sez("2. La maschera causale: cosa impedisce, esattamente")

S_ = 5
m = torch.triu(torch.ones(S_, S_, dtype=torch.bool), diagonal=1)
print("  maschera (True = VIETATO guardare):\n")
print("        " + "  ".join(f"k{j}" for j in range(S_)))
for i in range(S_):
    print(f"    q{i}  " + "   ".join("X" if m[i, j] else "." for j in range(S_)))

pesi = mia_attention(q[:, :, :S_], k[:, :, :S_], v[:, :, :S_], causale=True)
p = (q[:1, :1, :S_] @ k[:1, :1, :S_].transpose(-2, -1) / math.sqrt(D))
p = p.masked_fill(m, float("-inf")).softmax(-1)[0, 0]
print(f"\n  Pesi di attention della prima testa (righe = query):")
for i in range(S_):
    print("    " + "  ".join(f"{p[i, j]:.2f}" for j in range(S_)))
print("\n  Triangolo superiore esattamente 0: il token i non vede i successivi.")
print(f"  Ogni riga somma a 1: {p.sum(-1).tolist()}")
print("\n  ATTENZIONE: la maschera causale e' cio' che rende sensato addestrare")
print("  su tutta la sequenza in una volta. Senza, ogni token vedrebbe la")
print("  risposta e la loss crollerebbe a zero - il caso di M02.")


# ----------------------------------------------------------------------
sez("3. Perche' si divide per sqrt(d)")

print(f"  {'d':>6} {'std punteggi SENZA /sqrt(d)':>28} {'CON':>10} {'max softmax':>13}")
for d in (8, 64, 512, 4096):
    qq, kk = torch.randn(1, 1, 64, d), torch.randn(1, 1, 64, d)
    grezzi = qq @ kk.transpose(-2, -1)
    scalati = grezzi / math.sqrt(d)
    print(f"  {d:>6} {grezzi.std().item():>28.2f} {scalati.std().item():>10.2f}"
          f" {grezzi.softmax(-1).max().item():>13.4f}")

print("\n  Senza la divisione la deviazione standard dei punteggi cresce come")
print("  sqrt(d). A d grande il softmax diventa quasi un massimo secco: un solo")
print("  token prende peso ~1 e gli altri ~0. Il gradiente attraverso un softmax")
print("  saturo e' quasi nullo -> quello strato smette di imparare. -> M02")


# ----------------------------------------------------------------------
sez("4. GQA: quanto si risparmia, contato")

print(f"  hidden={H*16}, teste query=32\n")
Hbig, QH = H * 16, 32
d = Hbig // QH
print(f"  {'kv_heads':>9} {'q per gruppo':>13} {'par K+V':>12} {'vs MHA':>9} {'KV cache/token':>16}")
for kvh in (32, 8, 4, 1):
    par_kv = 2 * (kvh * d) * Hbig
    par_mha = 2 * (QH * d) * Hbig
    cache = 2 * kvh * d * 2                        # K e V, bf16
    print(f"  {kvh:>9} {QH // kvh:>13} {par_kv:>12,} {100 * par_kv / par_mha:>8.1f}%"
          f" {cache:>13,} B")

print("\n  MHA = 32 kv_heads (nessuna condivisione). MQA = 1.")
print("  Il risparmio sui PARAMETRI e' reale ma modesto sul totale del modello")
print("  (K e V sono 2 delle 7 matrici per strato). Il risparmio grosso e' la")
print("  KV CACHE in inferenza, che scala con kv_heads x lunghezza x batch.")
print("  Q resta a dimensione piena: su di lui non si risparmia niente.")

# verifica che GQA calcoli davvero la stessa cosa di MHA con K,V ripetuti
kvh = 2
qg = torch.randn(B, TESTE, S, D)
kg = torch.randn(B, kvh, S, D)
vg = torch.randn(B, kvh, S, D)
rip = TESTE // kvh
kg_rip, vg_rip = kg.repeat_interleave(rip, dim=1), vg.repeat_interleave(rip, dim=1)
mio_gqa = mia_attention(qg, kg_rip, vg_rip)

print()
# riferimento sempre disponibile: SDPA standard sulle teste K/V gia' ripetute
confronta("GQA (4 teste q, 2 kv) vs SDPA su K/V ripetute",
          mio_gqa, F.scaled_dot_product_attention(qg, kg_rip, vg_rip))

# riferimento aggiuntivo, solo se la versione di torch lo supporta
try:
    rif_nativo = F.scaled_dot_product_attention(qg, kg, vg, enable_gqa=True)
    confronta("  ... e vs il percorso nativo enable_gqa=True", mio_gqa, rif_nativo)
except TypeError:
    print(f"  {'  ... enable_gqa non disponibile in torch ' + torch.__version__:<44}"
          f" (aggiunto in una versione successiva)")
    print("      Il confronto sopra resta valido: ripetere K/V e fare attention")
    print("      normale E' la definizione di GQA, non un'approssimazione.")


# ----------------------------------------------------------------------
sez("5. RoPE: la proprieta' che lo rende utile")

def rope(x, pos, base=10000.0):
    """x: [.., seq, d]. Ruota coppie di dimensioni di un angolo pos*theta."""
    d = x.shape[-1]
    inv = base ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)
    ang = pos.unsqueeze(-1) * inv                    # [seq, d/2]
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


d = 32
a, b = torch.randn(d), torch.randn(d)
print("  Proprieta' attesa: <RoPE(a,m), RoPE(b,n)> dipende SOLO da (m-n).\n")
print(f"  {'m':>4} {'n':>4} {'m-n':>5} {'prodotto scalare':>18}")
for m_, n_ in [(0, 0), (5, 5), (100, 100), (3, 1), (7, 5), (101, 99), (1, 3)]:
    pa = rope(a.unsqueeze(0), torch.tensor([float(m_)]))[0]
    pb = rope(b.unsqueeze(0), torch.tensor([float(n_)]))[0]
    print(f"  {m_:>4} {n_:>4} {m_ - n_:>5} {(pa @ pb).item():>18.6f}")

print("\n  Righe con la stessa differenza danno lo stesso valore, a posizioni")
print("  assolute molto diverse. E' questo che si intende con 'RoPE codifica")
print("  la posizione RELATIVA': non c'e' un vettore di posizione sommato,")
print("  c'e' una rotazione il cui effetto sul prodotto scalare dipende solo")
print("  dalla distanza.")
print("\n  NOTA: (3,1) e (1,3) danno valori DIVERSI: il segno della differenza")
print("  conta. La proprieta' e' 'funzione di m-n', non 'di |m-n|'.")

print("\n  Cosa NON segue da qui: che il modello funzioni oltre la lunghezza")
print("  su cui e' stato addestrato. La formula si valuta a qualunque posizione,")
print("  ma le rotazioni a frequenze mai viste in addestramento producono")
print("  configurazioni fuori distribuzione. Estendere il contesto richiede")
print("  interventi espliciti (interpolazione, riaddestramento) e va misurato.")


# ----------------------------------------------------------------------
sez("6. RMSNorm contro LayerNorm")

def mia_rmsnorm(x, peso, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * peso

x = torch.randn(B, S, H)
w = torch.randn(H)
print()
# F.rms_norm esiste solo dalle versioni piu' recenti: riferimento esplicito
# come fallback, cosi' il confronto vale su qualunque versione.
def rmsnorm_riferimento(x, peso, eps=1e-6):
    var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return (x.to(torch.float32) * torch.rsqrt(var + eps)).to(x.dtype) * peso

if hasattr(F, "rms_norm"):
    confronta("mia RMSNorm  vs  F.rms_norm (nativa)",
              mia_rmsnorm(x, w), F.rms_norm(x, (H,), w, eps=1e-6))
else:
    print(f"  F.rms_norm non disponibile in torch {torch.__version__}:"
          f" uso la formula esplicita")
    confronta("mia RMSNorm  vs  formula di riferimento",
              mia_rmsnorm(x, w), rmsnorm_riferimento(x, w))

ln = F.layer_norm(x, (H,))
rn = mia_rmsnorm(x, torch.ones(H))
print(f"\n  media dopo LayerNorm  {ln.mean().item():+.2e}   (centra: media -> 0)")
print(f"  media dopo RMSNorm    {rn.mean().item():+.2e}   (NON centra)")
print(f"  norma quadratica media dopo RMSNorm: {rn.pow(2).mean().item():.4f}  (-> 1)")
print("\n  RMSNorm normalizza solo la SCALA, non sottrae la media. Meno")
print("  operazioni, nessun parametro di bias. La differenza di costo e'")
print("  piccola per tensore ma si moltiplica per il numero di strati.")


# ----------------------------------------------------------------------
sez("7. L'MLP con SwiGLU: perche' tre matrici e non due")

def mio_swiglu(x, Wg, Wu, Wd):
    return (F.silu(x @ Wg) * (x @ Wu)) @ Wd

inter = int(H * 8 / 3)
Wg, Wu, Wd = torch.randn(H, inter), torch.randn(H, inter), torch.randn(inter, H)
y = mio_swiglu(x, Wg, Wu, Wd)
print(f"  ingresso {list(x.shape)} -> uscita {list(y.shape)}")
print(f"  hidden={H}  intermedio={inter}\n")
print("  gate = silu(x @ Wg)   fa da INTERRUTTORE, valori in (-0.28, +inf)")
print("  up   = x @ Wu         il segnale vero")
print("  down = (gate*up) @ Wd riporta alla dimensione del modello")
print(f"\n  frazione di gate negativi: {(F.silu(x @ Wg) < 0).float().mean().item():.3f}")
print("  Un MLP classico ha 2 matrici. SwiGLU ne ha 3, e per pareggiare i")
print("  parametri si riduce la dimensione intermedia (spesso a ~2/3 x 4H).")
print(f"  Qui: 8/3 x H = {inter}, invece di 4 x H = {4 * H}.")


# ----------------------------------------------------------------------
sez("8. Un blocco completo, confrontato col riferimento")

class MioBlocco(torch.nn.Module):
    def __init__(self, h, teste):
        super().__init__()
        self.h, self.teste, self.d = h, teste, h // teste
        self.wq = torch.nn.Linear(h, h, bias=False)
        self.wk = torch.nn.Linear(h, h, bias=False)
        self.wv = torch.nn.Linear(h, h, bias=False)
        self.wo = torch.nn.Linear(h, h, bias=False)
        self.n1 = torch.nn.Parameter(torch.ones(h))
        self.n2 = torch.nn.Parameter(torch.ones(h))
        inter = int(h * 8 / 3)
        self.wg = torch.nn.Linear(h, inter, bias=False)
        self.wu = torch.nn.Linear(h, inter, bias=False)
        self.wd = torch.nn.Linear(inter, h, bias=False)

    def forward(self, x):
        b, s, _ = x.shape
        r = x
        x = mia_rmsnorm(x, self.n1)
        q = self.wq(x).view(b, s, self.teste, self.d).transpose(1, 2)
        k = self.wk(x).view(b, s, self.teste, self.d).transpose(1, 2)
        v = self.wv(x).view(b, s, self.teste, self.d).transpose(1, 2)
        a = mia_attention(q, k, v, causale=True)
        x = r + self.wo(a.transpose(1, 2).reshape(b, s, self.h))
        r = x
        x = mia_rmsnorm(x, self.n2)
        return r + self.wd(F.silu(self.wg(x)) * self.wu(x))


blocco = MioBlocco(H, TESTE)
x = torch.randn(B, S, H)
y = blocco(x)
print(f"  ingresso {list(x.shape)} -> uscita {list(y.shape)}   forma conservata: {x.shape == y.shape}")

# stesso blocco, ma con l'attention di riferimento al posto della mia
def blocco_riferimento(m, x):
    b, s, _ = x.shape
    r = x
    xn = mia_rmsnorm(x, m.n1)
    q = m.wq(xn).view(b, s, m.teste, m.d).transpose(1, 2)
    k = m.wk(xn).view(b, s, m.teste, m.d).transpose(1, 2)
    v = m.wv(xn).view(b, s, m.teste, m.d).transpose(1, 2)
    a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    x = r + m.wo(a.transpose(1, 2).reshape(b, s, m.h))
    r = x
    xn = mia_rmsnorm(x, m.n2)
    return r + m.wd(F.silu(m.wg(xn)) * m.wu(xn))


print()
confronta("blocco intero con la MIA attention vs quella di riferimento",
          y, blocco_riferimento(blocco, x))

n_par = sum(p.numel() for p in blocco.parameters())
print(f"\n  parametri del blocco: {n_par:,}")
print(f"    attention (4 matrici h x h)      {4 * H * H:>10,}  {400 * H * H / n_par:>5.1f}%")
print(f"    MLP (3 matrici)                  {3 * H * inter:>10,}  {300 * H * inter / n_par:>5.1f}%")
print(f"    norme (2 vettori)                {2 * H:>10,}  {200 * H / n_par:>5.1f}%")

# confronto onesto col checkpoint di M04, che usa GQA
h4, i4, qh4, kvh4, d4 = 512, 1376, 8, 2, 64
att4 = (qh4 * d4) * h4 * 2 + (kvh4 * d4) * h4 * 2      # q,o pieni + k,v ridotti
mlp4 = 3 * i4 * h4
tot4 = att4 + mlp4 + 2 * h4
print(f"\n  Confronto col checkpoint di M04 (che usa GQA 8q/2kv), per strato:")
print(f"    attention  {100 * att4 / tot4:>5.1f}%   contro {400 * H * H / n_par:.1f}% qui")
print(f"    MLP        {100 * mlp4 / tot4:>5.1f}%   contro {300 * H * inter / n_par:.1f}% qui")
print("\n  L'MLP domina in entrambi, ma NON nella stessa proporzione: questo")
print("  blocco non ha GQA, quindi K e V sono a dimensione piena e l'attention")
print("  pesa di piu'. E' GQA a spostare il rapporto, ed e' un esempio di")
print("  perche' le percentuali di un modello non si trasferiscono a un altro.")

print()
