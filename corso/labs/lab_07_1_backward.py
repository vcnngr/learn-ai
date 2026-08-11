#!/usr/bin/env python3
"""
Lab 07.1 — Cosa resta in memoria fra forward e backward.

    python3 corso/labs/lab_07_1_backward.py

Le sezioni 1-3 girano su CPU (Mac).
Le sezioni 4-5 richiedono CUDA: senza, si fermano con un messaggio esplicito
invece di stampare numeri finti.
"""

import math
import pathlib
import platform
import sys

import torch
import torch.nn as nn


def sez(t):
    print(f"\n{'─' * 72}\n{t}\n{'─' * 72}")


HA_CUDA = torch.cuda.is_available()

# ======================================================================
# ATTESI — registrare una misura e riconfrontarla altrove.
#
#   python3 lab_07_1_backward.py --registra    salva le misure di QUESTO
#                                              ambiente in labs/expected/
#   python3 lab_07_1_backward.py --confronta   le rimette a fianco di
#                                              quelle appena misurate
#
# A cosa serve: le sezioni 4-5 non sono mai state eseguite su GPU da chi
# ha scritto il corso. Chi ha l'hardware non ha un riferimento per sapere
# se l'output che ottiene e' quello atteso o un difetto del lab. Con un
# file di attesi, "non verificato" diventa "verificabile da chiunque".
#
# Il blocco e' RIPETUTO in ogni lab che lo usa invece di stare in un
# modulo condiviso: i lab del corso sono autonomi, si copiano e si
# lanciano da soli, e un import fra lab romperebbe quella promessa.
# Trenta righe duplicate sono il prezzo dichiarato di quella scelta.
# ======================================================================

REGISTRA = "--registra" in sys.argv
CONFRONTA = "--confronta" in sys.argv
ATTESI = pathlib.Path(__file__).resolve().parent / "expected"
TOLLERANZA = 0.10          # 10% relativo, DICHIARATA: memoria e tempi
                           # variano con versione, driver e carico.


def _ambiente():
    return (f"torch={torch.__version__} "
            f"cuda={torch.version.cuda or '-'} "
            f"gpu={torch.cuda.get_device_name(0) if HA_CUDA else '-'} "
            f"piattaforma={platform.system()}-{platform.machine()}")


def _file(nome):
    return ATTESI / f"{pathlib.Path(__file__).stem}__{nome}.txt"


def registra(nome, misure):
    """misure: dict chiave -> valore numerico."""
    if not REGISTRA:
        return
    ATTESI.mkdir(exist_ok=True)
    f = _file(nome)
    righe = [f"# ambiente: {_ambiente()}",
             f"# tolleranza dichiarata: {TOLLERANZA:.0%} relativo"]
    righe += [f"{k}\t{v!r}" for k, v in misure.items()]
    f.write_text("\n".join(righe) + "\n")
    print(f"\n  REGISTRATO in {f.relative_to(ATTESI.parent.parent)}"
          f"  ({len(misure)} valori)")


def confronta(nome, misure):
    if not CONFRONTA:
        return
    f = _file(nome)
    if not f.exists():
        print(f"""
  NESSUN ATTESO REGISTRATO per '{nome}'.
  Il file {f.name} non esiste ancora: nessuno ha mai eseguito questa
  sezione in un ambiente da cui valesse la pena registrarla.

  Se il tuo output ti sembra sensato, registralo e diventa il
  riferimento per chi viene dopo:
      python3 {pathlib.Path(__file__).name} --registra""")
        return
    testo = f.read_text().splitlines()
    intestazione = [r for r in testo if r.startswith("#")]
    attesi = {}
    for r in testo:
        if r.startswith("#") or "\t" not in r:
            continue
        k, v = r.split("\t", 1)
        try:
            attesi[k] = float(v)
        except ValueError:
            pass

    print("\n  CONFRONTO con l'atteso registrato")
    for r in intestazione:
        print(f"    {r}")
    print(f"\n  {'grandezza':<22} {'misurato':>12} {'atteso':>12} {'scarto':>9}")
    fuori = 0
    for k, v in misure.items():
        a = attesi.get(k)
        if a is None:
            print(f"  {k:<22} {v:>12.4f} {'assente':>12} {'-':>9}")
            continue
        rel = abs(v - a) / abs(a) if a else (0.0 if v == a else 1.0)
        segno = "OK" if rel <= TOLLERANZA else "FUORI"
        fuori += segno == "FUORI"
        print(f"  {k:<22} {v:>12.4f} {a:>12.4f} {rel:>8.1%} {segno}")
    solo_attesi = [k for k in attesi if k not in misure]
    for k in solo_attesi:
        print(f"  {k:<22} {'non misurato':>12} {attesi[k]:>12.4f} {'-':>9}")
    print(f"""
  {fuori} valori fuori dalla tolleranza dichiarata del {TOLLERANZA:.0%}.

  Uno scarto NON significa che uno dei due sia sbagliato: significa che
  i due ambienti differiscono, e l'intestazione qui sopra dice in cosa.
  Un tempo che cambia e' normale. Una MEMORIA che cambia di molto no:
  quella e' aritmetica, e vale la pena capire perche'.""")

sez("0. Ambiente")
print(f"  torch            {torch.__version__}")
print(f"  CUDA disponibile {HA_CUDA}")
if HA_CUDA:
    print(f"  device           {torch.cuda.get_device_name(0)}")
    print(f"  GPU visibili     {torch.cuda.device_count()}")
else:
    print("  -> sezioni 1-3 girano lo stesso (CPU).")
    print("  -> sezioni 4-5 si fermeranno: misurano memoria GPU reale.")


# ----------------------------------------------------------------------
sez("1. Il grafo: chi ha prodotto cosa")

x = torch.randn(4, 8, requires_grad=True)
W1 = torch.randn(8, 16, requires_grad=True)
W2 = torch.randn(16, 4, requires_grad=True)

h = x @ W1
a = torch.relu(h)
y = a @ W2
loss = y.pow(2).mean()

for nome, t in [("x", x), ("W1", W1), ("h", h), ("a", a), ("y", y), ("loss", loss)]:
    fn = t.grad_fn.__class__.__name__ if t.grad_fn else ("FOGLIA" if t.requires_grad else "-")
    print(f"  {nome:>5}  shape={str(list(t.shape)):<10} requires_grad={str(t.requires_grad):<5} grad_fn={fn}")

print("\n  grad_fn = l'operazione che ha prodotto quel tensore.")
print("  Seguendola all'indietro si ricostruisce tutto il percorso. E' il GRAFO.")
print("  Le FOGLIE sono cio' che hai creato tu: solo loro avranno .grad. -> M02")


# ----------------------------------------------------------------------
sez("2. Cosa il backward TRATTIENE davvero")

print("  Non lo decido io: lo si CHIEDE al grafo. Ogni grad_fn espone gli")
print("  attributi _saved_* con cio' che ha trattenuto.\n")

xx = torch.randn(4, 8, requires_grad=True)
WW = torch.randn(8, 16, requires_grad=True)
casi = [
    ("y = x @ W",   xx @ WW),
    ("y = relu(x)", torch.relu(xx)),
    ("y = x * 2",   xx * 2),
    ("y = x.exp()", xx.exp()),
]
print(f"  {'operazione':<14} {'grad_fn':<18} tensori trattenuti")
for espr, out in casi:
    gf = out.grad_fn
    tenuti = []
    for a in dir(gf):
        if not a.startswith("_saved") or a.endswith(("_sym_sizes", "_sym_strides")):
            continue
        v = getattr(gf, a)
        if torch.is_tensor(v):
            tenuti.append(f"{a[7:]}{list(v.shape)}")
    print(f"  {espr:<14} {gf.__class__.__name__:<18} {', '.join(tenuti) if tenuti else 'NIENTE'}")

print("""
  Come si leggono queste righe:

    x @ W    trattiene ENTRAMBI gli ingressi: 'self' (=x) per calcolare
             dL/dW, e 'mat2' (=W) per calcolare dL/dx. E' la formula di M02.
             W e' gia' un parametro persistente; x no -> x e' il costo nuovo.

    relu(x)  trattiene 'result', cioe' l'USCITA, con la stessa dimensione
             dell'ingresso. Concettualmente al backward basterebbe sapere
             DOVE l'ingresso era positivo - un bit per elemento - ma
             l'implementazione tiene il tensore intero. Costa quanto x.

    x * 2    trattiene solo 'other', che e' lo scalare 2: shape [], cioe'
             zero dimensioni, un singolo numero. NON trattiene x. La derivata
             e' la costante 2, quindi l'ingresso non serve. Costo: nullo in
             pratica, per quanto grande sia x.

    x.exp()  trattiene 'result', perche' la derivata di e^x e' e^x stesso:
             l'uscita gia' calcolata E' il gradiente locale.

  Conseguenza pratica: le operazioni NON costano memoria allo stesso modo.
  Una moltiplicazione per una costante e' gratis nel backward; un matmul
  trattiene il suo ingresso. E' li' che sta il costo. -> M02, M06""")


# ----------------------------------------------------------------------
sez("3. Il costo cresce con batch x sequenza (misurato su CPU)")

def byte_trattenuti(loss):
    """Cammina il grafo di autograd e somma i byte di STORAGE che restano
    pinnati dai tensori trattenuti.

    Due dettagli che rendono il conto corretto invece di plausibile:
      - si conta storage.nbytes(), NON numel()*element_size() della vista:
        una vista piccola puo' tenere in vita uno storage grande, ed e' lo
        storage a occupare memoria;
      - alcuni nodi salvano LISTE/TUPLE di tensori, non solo tensori singoli.
    """
    visti_nodi, visti_storage, tot = set(), set(), 0

    def considera(v):
        nonlocal tot
        if torch.is_tensor(v):
            try:
                st = v.untyped_storage()
            except Exception:
                return
            p = st.data_ptr()
            if p and p not in visti_storage:
                visti_storage.add(p)
                tot += st.nbytes()
        elif isinstance(v, (list, tuple)):
            for e in v:
                considera(e)

    da_fare = [loss.grad_fn]
    while da_fare:
        nodo = da_fare.pop()
        if nodo is None or nodo in visti_nodi:
            continue
        visti_nodi.add(nodo)
        for a in dir(nodo):
            if not a.startswith("_saved") or a.endswith(("_sym_sizes", "_sym_strides")):
                continue
            considera(getattr(nodo, a, None))
        for succ, _ in getattr(nodo, "next_functions", ()):
            da_fare.append(succ)
    return tot, len(visti_nodi), len(visti_storage)


def rete_lineare(batch, seq, hidden, layers):
    """Solo strati lineari + relu: nessuna attention, nessun termine seq^2."""
    x = torch.randn(batch, seq, hidden, requires_grad=True)
    pesi = [torch.randn(hidden, hidden, requires_grad=True) for _ in range(layers)]
    for W in pesi:
        x = torch.relu(x @ W)
    return x.pow(2).mean()


H, L = 256, 8
pesi_mb = L * H * H * 4 / 1e6            # i pesi sono trattenuti anche loro: costante

print("  Rete di soli strati lineari + relu (nessuna attention).")
print("  I byte NON sono sommati a mano: sono letti dal grafo di autograd.\n")
print(f"  I pesi trattenuti sono un termine COSTANTE: {pesi_mb:.2f} MB")
print("  (non dipendono da batch ne' da seq). Li separo, altrimenti nascondono")
print("  l'andamento della parte che varia.\n")
print(f"  {'batch':>6} {'seq':>6} {'batch*seq':>10} {'MB totali':>11} {'- pesi':>9} {'= attivaz.':>11}")
base = None
for b, s in [(1, 128), (2, 128), (1, 256), (4, 256), (2, 512)]:
    byte, nodi, nt = byte_trattenuti(rete_lineare(b, s, H, L))
    att = byte / 1e6 - pesi_mb
    if base is None:
        base = att
    print(f"  {b:>6} {s:>6} {b * s:>10} {byte / 1e6:>11.2f} {pesi_mb:>9.2f} {att:>11.2f}"
          f"   ({att / base:.0f}x la prima riga)")

print("""
  Guarda l'ultima colonna, non la prima: le attivazioni raddoppiano quando
  raddoppia batch*seq, e le righe con lo stesso batch*seq coincidono.
  Il totale grezzo NON raddoppia, perche' contiene la costante dei pesi.

  Qui posso separare le due componenti perche' la costante la CONOSCO
  (numero di pesi x byte). Su un modello vero fai lo stesso: misuri il
  termine statico da solo - modello + optimizer, un passo minimo, nessun
  forward vero - e lo sottrai. Non estrapolare da piu' batch: il picco
  non e' garantito lineare, e sotto batch=1 non si scende.

  ATTENZIONE - questo vale per QUESTA rete, fatta di soli strati lineari.
  Non e' una legge generale: l'attention classica costruisce una matrice
  [batch, teste, seq, seq], che cresce col QUADRATO della sequenza. In quel
  caso raddoppiare seq quadruplica quel termine, mentre raddoppiare il batch
  lo raddoppia soltanto: batch e sequenza NON sono piu' intercambiabili.
  Le implementazioni tipo FlashAttention non materializzano quella matrice
  e riportano il termine a lineare - quindi dipende dal kernel che usi.""")

# Dimostrazione del termine quadratico, sulla stessa infrastruttura di misura
print("\n  Lo stesso conto con attention esplicita (matrice seq x seq materializzata):\n")
print(f"  {'batch':>6} {'seq':>6} {'batch*seq':>10} {'MB trattenuti':>15}")


def rete_attention(batch, seq, hidden, teste=4):
    x = torch.randn(batch, seq, hidden, requires_grad=True)
    Wq = torch.randn(hidden, hidden, requires_grad=True)
    Wk = torch.randn(hidden, hidden, requires_grad=True)
    Wv = torch.randn(hidden, hidden, requires_grad=True)
    d = hidden // teste
    q = (x @ Wq).view(batch, seq, teste, d).transpose(1, 2)
    k = (x @ Wk).view(batch, seq, teste, d).transpose(1, 2)
    v = (x @ Wv).view(batch, seq, teste, d).transpose(1, 2)
    punteggi = (q @ k.transpose(-2, -1)) / math.sqrt(d)      # [b, teste, seq, seq]
    pesi = punteggi.softmax(-1)
    return (pesi @ v).pow(2).mean()


_mis_cpu = {}
for b, s in [(1, 128), (2, 128), (1, 256), (4, 256), (2, 512)]:
    byte, _, _ = byte_trattenuti(rete_attention(b, s, 256))
    _mis_cpu[f"attn_b{b}_s{s}_MB"] = byte / 1e6
    print(f"  {b:>6} {s:>6} {b * s:>10} {byte / 1e6:>15.2f}")

print("\n  Confronta le righe con batch*seq uguale: adesso NON coincidono piu'.")
print("  La sequenza costa piu' del batch, ed e' il termine quadratico.")

registra("cpu", _mis_cpu)
confronta("cpu", _mis_cpu)


# ----------------------------------------------------------------------
sez("4. Gradient checkpointing: memoria vs tempo (SERVE CUDA)")

if not HA_CUDA:
    print("""  FERMO: questa sezione misura memoria GPU reale con
    torch.cuda.max_memory_allocated()
  e su CPU non esiste un equivalente confrontabile.

  NON stampo numeri stimati: sarebbero inventati, e questo corso non lo fa.

  Lancia questo script sul nodo con le H200:
      python3 lab_07_1_backward.py

  Cosa misurera':
    - picco di memoria con e senza gradient checkpointing
    - secondi per step nei due casi
    - il rapporto fra i due, che e' il vero trade-off

  PRIMA di lanciarlo, scrivi la tua previsione:
    memoria risparmiata   ____%      tempo in piu'   ____%""")
else:
    # torch.utils.checkpoint NON e' accessibile con il solo 'import torch':
    # va importato esplicitamente, altrimenti AttributeError.
    import torch.utils.checkpoint as ckpt

    class Blocco(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.a = nn.Linear(h, 4 * h)
            self.b = nn.Linear(4 * h, h)

        def forward(self, x):
            return x + self.b(torch.nn.functional.gelu(self.a(x)))

    class Rete(nn.Module):
        def __init__(self, h, n, ckpt):
            super().__init__()
            self.blocchi = nn.ModuleList([Blocco(h) for _ in range(n)])
            self.ckpt = ckpt

        def forward(self, x):
            for b in self.blocchi:
                if self.ckpt:
                    x = ckpt.checkpoint(b, x, use_reentrant=False)
                else:
                    x = b(x)
            return x

    import time
    H, N, B, S = 1024, 12, 4, 512
    print(f"  hidden={H} blocchi={N} batch={B} seq={S}\n")
    print(f"  {'modalita':>16} {'picco GB':>10} {'s/step':>9}")
    ris = {}
    for nome, ckpt in [("senza ckpt", False), ("con ckpt", True)]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        m = Rete(H, N, ckpt).cuda()
        opt = torch.optim.SGD(m.parameters(), lr=1e-4)
        x = torch.randn(B, S, H, device="cuda")
        for _ in range(3):                       # warm-up
            m(x).pow(2).mean().backward()
            opt.step(); opt.zero_grad()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(10):
            m(x).pow(2).mean().backward()
            opt.step(); opt.zero_grad()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 10
        picco = torch.cuda.max_memory_allocated() / 1e9
        ris[nome] = (picco, dt)
        print(f"  {nome:>16} {picco:>10.2f} {dt:>9.4f}")
        del m, opt, x

    (p0, t0_), (p1, t1_) = ris["senza ckpt"], ris["con ckpt"]
    print(f"\n  memoria: {100 * (1 - p1 / p0):>5.1f}% risparmiata")
    print(f"  tempo:   {100 * (t1_ / t0_ - 1):>5.1f}% in piu'")
    print("\n  Confronta col tuo pronostico. Lo scarto dice quale termine del tuo")
    print("  modello mentale era sbagliato.")

    _mis_gpu = {
        "picco_senza_ckpt_GB": p0,
        "picco_con_ckpt_GB": p1,
        "sec_per_step_senza_ckpt": t0_,
        "sec_per_step_con_ckpt": t1_,
        "memoria_risparmiata_pct": 100 * (1 - p1 / p0),
        "tempo_in_piu_pct": 100 * (t1_ / t0_ - 1),
    }


# ----------------------------------------------------------------------
sez("5. Formula contro misura (SERVE CUDA)")

if not HA_CUDA:
    print("""  FERMO: stesso motivo.

  Sul nodo, questa sezione confrontera' la stima di M06
      batch x seq x hidden x layers x k x byte
  con il picco reale misurato, per ricavare il k EFFETTIVO del tuo stack.

  E' il numero che rende utilizzabile la formula di M06: una volta che
  conosci il k della tua architettura e della tua versione di libreria,
  le previsioni successive diventano affidabili.""")
else:
    H, N, B, S = 1024, 12, 4, 512
    stima_k1 = B * S * H * N * 2
    print(f"  ingredienti: batch={B} seq={S} hidden={H} layers={N} bf16")
    print(f"  termine base (k=1): {stima_k1 / 1e9:.3f} GB")
    p0 = ris["senza ckpt"][0]
    print(f"  picco misurato senza ckpt: {p0:.3f} GB")
    print(f"  -> k effettivo ~ {p0 * 1e9 / stima_k1:.1f}")
    print("\n  ATTENZIONE: questo k include anche pesi, gradienti e stati")
    print("  optimizer, non solo le attivazioni. Per isolarle, rifai la misura")
    print("  variando SOLO il batch: la pendenza e' il costo per unita' di batch.")

    _mis_gpu["k_effettivo"] = p0 * 1e9 / stima_k1
    registra("gpu", _mis_gpu)
    confronta("gpu", _mis_gpu)

if not HA_CUDA and (REGISTRA or CONFRONTA):
    print(f"""
  --registra / --confronta sulle sezioni GPU: SALTATI, manca CUDA.
  Le sezioni 4-5 non hanno prodotto misure, quindi non c'e' niente da
  registrare ne' da confrontare. La sezione 3, che gira su CPU, e'
  stata trattata regolarmente.

  Stato degli attesi su questa macchina:
    {_file('cpu').name}  {'presente' if _file('cpu').exists() else 'assente'}
    {_file('gpu').name}  {'presente' if _file('gpu').exists() else 'assente'}""")

print()
