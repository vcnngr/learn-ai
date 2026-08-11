#!/usr/bin/env python3
"""
Lab 18.1 — Parallelismo: cosa partiziona ciascuna strategia.

    python3 corso/labs/lab_18_1_parallelismo.py            # aritmetica, CPU
    torchrun --nproc_per_node=2 labs/lab_18_1_parallelismo.py --misura

Le sezioni 1-4 sono aritmetica e girano ovunque.
La sezione 5 MISURA su GPU reali: senza CUDA si ferma con un messaggio
esplicito invece di stampare numeri inventati.
"""

import argparse
import os

GB = 1e9

ap = argparse.ArgumentParser()
ap.add_argument("--misura", action="store_true", help="esegue la parte GPU")
ap.add_argument("--params", type=float, default=27e9)
ap.add_argument("--gpu", type=int, default=2)
ap.add_argument("--hbm", type=float, default=141.0)
ap.add_argument("--registra", action="store_true",
                help="salva le misure di questo ambiente in labs/expected/")
ap.add_argument("--confronta", action="store_true",
                help="rimette gli attesi registrati a fianco dei misurati")
args = ap.parse_args()

# ======================================================================
# ATTESI — vedi lab_07_1_backward.py per il perche' di questo blocco.
# E' duplicato di proposito: i lab del corso sono autonomi, si copiano e
# si lanciano da soli, e un import fra lab romperebbe quella promessa.
# ======================================================================

import pathlib
import platform

ATTESI = pathlib.Path(__file__).resolve().parent / "expected"
TOLLERANZA = 0.10          # 10% relativo, DICHIARATA


def _file(nome):
    return ATTESI / f"{pathlib.Path(__file__).stem}__{nome}.txt"


def _ambiente(torch_mod):
    return (f"torch={torch_mod.__version__} "
            f"cuda={torch_mod.version.cuda or '-'} "
            f"gpu={torch_mod.cuda.get_device_name(0) if torch_mod.cuda.is_available() else '-'} "
            f"piattaforma={platform.system()}-{platform.machine()}")


def registra(nome, misure, torch_mod):
    if not args.registra:
        return
    ATTESI.mkdir(exist_ok=True)
    f = _file(nome)
    righe = [f"# ambiente: {_ambiente(torch_mod)}",
             f"# tolleranza dichiarata: {TOLLERANZA:.0%} relativo"]
    righe += [f"{k}\t{v!r}" for k, v in misure.items()]
    f.write_text("\n".join(righe) + "\n")
    print(f"\n  REGISTRATO in labs/expected/{f.name}  ({len(misure)} valori)")


def confronta(nome, misure):
    if not args.confronta:
        return
    f = _file(nome)
    if not f.exists():
        print(f"""
  NESSUN ATTESO REGISTRATO per '{nome}'.
  Il file {f.name} non esiste: questa sezione non e' mai stata eseguita
  in un ambiente da cui valesse la pena registrarla. Se il tuo output ti
  sembra sensato, registralo e diventa il riferimento per chi viene dopo:
      torchrun --nproc_per_node={args.gpu} {pathlib.Path(__file__).name} --misura --registra""")
        return
    testo = f.read_text().splitlines()
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
    for r in [x for x in testo if x.startswith("#")]:
        print(f"    {r}")
    print(f"\n  {'grandezza':<26} {'misurato':>12} {'atteso':>12} {'scarto':>9}")
    fuori = 0
    for k, v in misure.items():
        a = attesi.get(k)
        if a is None:
            print(f"  {k:<26} {v:>12.4f} {'assente':>12} {'-':>9}")
            continue
        rel = abs(v - a) / abs(a) if a else (0.0 if v == a else 1.0)
        segno = "OK" if rel <= TOLLERANZA else "FUORI"
        fuori += segno == "FUORI"
        print(f"  {k:<26} {v:>12.4f} {a:>12.4f} {rel:>8.1%} {segno}")
    print(f"""
  {fuori} valori fuori dalla tolleranza dichiarata del {TOLLERANZA:.0%}.
  Uno scarto non dice che uno dei due sia sbagliato: dice che i due
  ambienti differiscono, e l'intestazione qui sopra dice in cosa.""")


def sez(t):
    print(f"\n{'─' * 72}\n{t}\n{'─' * 72}")


P, N, HBM = args.params, args.gpu, args.hbm

# ----------------------------------------------------------------------
sez("1. Cosa partiziona ciascuna strategia")

# voci statiche per FULL fine-tuning, bf16 pesi + AdamW fp32 + master
VOCI = {"pesi": P * 2, "gradienti": P * 2, "stati optimizer": P * 8, "master fp32": P * 4}
TOT = sum(VOCI.values())

STRATEGIE = {
    "singola GPU":      {"pesi": 1, "gradienti": 1, "stati optimizer": 1, "master fp32": 1},
    "DDP":              {"pesi": 1, "gradienti": 1, "stati optimizer": 1, "master fp32": 1},
    "ZeRO-1":           {"pesi": 1, "gradienti": 1, "stati optimizer": N, "master fp32": N},
    "ZeRO-2":           {"pesi": 1, "gradienti": N, "stati optimizer": N, "master fp32": N},
    "ZeRO-3 / FSDP":    {"pesi": N, "gradienti": N, "stati optimizer": N, "master fp32": N},
    "tensor parallel":  {"pesi": N, "gradienti": N, "stati optimizer": N, "master fp32": N},
}

print(f"  Full fine-tuning di {P:.3g} parametri su {N} GPU da {HBM:.0f} GB.")
print(f"  Voci statiche totali: {TOT / GB:.1f} GB (M06)\n")
print(f"  {'strategia':<18} {'pesi':>7} {'grad':>7} {'stati':>7} {'master':>7}"
      f" {'GB/GPU':>9} {'entra?':>8}")
for nome, div in STRATEGIE.items():
    g = sum(VOCI[k] / div[k] for k in VOCI) / GB
    print(f"  {nome:<18} {'/' + str(div['pesi']):>7} {'/' + str(div['gradienti']):>7}"
          f" {'/' + str(div['stati optimizer']):>7} {'/' + str(div['master fp32']):>7}"
          f" {g:>9.1f} {('SI' if g < HBM else 'NO'):>8}")

print("""
  Leggi la riga DDP: divide per 1 OGNI voce. DDP non riduce la memoria,
  la REPLICA. Due GPU in DDP danno il doppio del throughput e ZERO byte
  in piu' per il modello.

  E' la conseguenza diretta del punto di M06: non hai la somma delle
  memorie finche' non scegli una strategia che partiziona davvero.""")


# ----------------------------------------------------------------------
sez("2. Cosa costa la comunicazione")

print("""  Ogni strategia che partiziona deve poi RIMETTERE INSIEME i pezzi.
  Quanto traffico genera, per passo:

    DDP              all-reduce dei gradienti     ~ 2 x (P x byte_grad)
    ZeRO-1           come DDP + scatter/gather degli stati
    ZeRO-2           reduce-scatter dei gradienti + all-gather dei pesi
    ZeRO-3 / FSDP    all-gather dei pesi PER OGNI STRATO, avanti e indietro
    tensor parallel  all-reduce DENTRO ogni strato, piu' volte per strato
""")

byte_grad = P * 2
print(f"  {'strategia':<18} {'traffico per passo (ordine)':>30}")
print(f"  {'DDP':<18} {2 * byte_grad / GB:>27.1f} GB")
print(f"  {'ZeRO-3 (indicativo)':<18} {3 * byte_grad / GB:>27.1f} GB")
print("""
  Ordini di grandezza, non valori esatti: il traffico reale dipende
  dall'implementazione, dal raggruppamento dei parametri e da quanto
  la libreria riesce a SOVRAPPORRE comunicazione e calcolo.

  Il punto che resta: piu' partizioni, piu' comunichi. NVLink e' veloce,
  ma non gratis - e la differenza fra 'entra' e 'entra ed e' veloce'
  si misura, non si deduce.""")


# ----------------------------------------------------------------------
sez("3. Le due domande da tenere separate")

print("""  1. CI STA?           -> aritmetica, si risponde su carta (M06)
  2. E' VELOCE?        -> misura, si risponde solo eseguendo

  Sono indipendenti. ZeRO-3 fa entrare quasi tutto e puo' essere
  molto piu' lento di ZeRO-1 che ci sta per un pelo. La strategia
  giusta e' la MENO aggressiva che fa entrare il modello, non la piu'
  aggressiva disponibile.

  Errore tipico: attivare ZeRO-3 'per sicurezza' quando ZeRO-1 bastava,
  e pagare comunicazione per una capienza che non serviva.""")


# ----------------------------------------------------------------------
sez("4. Cosa cambia davvero fra 2 GPU e un cluster")

print(f"""  Con {N} GPU su un solo nodo:

    - la comunicazione passa da NVLink: veloce e a bassa latenza
    - non esistono i problemi di rete fra nodi
    - il tensor parallel e' praticabile: richiede banda altissima
    - hai DUE punti di misura per lo scaling: non abbastanza per
      estrapolare una curva

  Su un cluster:

    - la comunicazione fra nodi passa dalla rete: ordini di grandezza
      piu' lenta di NVLink
    - il tensor parallel si tiene DENTRO il nodo, gli altri paralleli
      attraversano i nodi
    - compaiono guasti che qui non vedi: un nodo lento rallenta tutti,
      un nodo morto ferma il run

  Cosa puoi imparare qui: quale strategia fa entrare cosa, e come si
  MISURA lo scaling. Cosa NON puoi imparare: il comportamento a scala,
  che dipende da una topologia che non hai. -> lo dice anche l'indice""")


# ----------------------------------------------------------------------
sez("5. La misura vera (SERVE CUDA + torchrun)")

if not args.misura:
    print(f"""  Non eseguita: manca il flag --misura.

  Sul nodo con le GPU:

      torchrun --nproc_per_node={N} labs/lab_18_1_parallelismo.py --misura

  Cosa misurera':
    - memoria per rank sotto DDP, VOCE PER VOCE (pesi, gradienti, stati
      optimizer) coi dtype davvero costruiti, confrontata con M06
    - tempo per passo su 1 GPU e su {N}, e l'efficienza di scaling

  Cosa NON misurera':
    - quanta parte del tempo e' comunicazione. Il tempo per passo e' un
      totale: quel che manca al 100% e' overhead complessivo di scaling,
      e separarne la comunicazione richiede un profiler.

  PRIMA di lanciarlo, scrivi la tua previsione:
    memoria per rank con DDP   ____ GB   (suggerimento: guarda la tabella 1)
    efficienza di scaling 1->{N}  ____ %   (100% = tempo dimezzato)""")
else:
    try:
        import torch
        import torch.distributed as dist
        import torch.nn as nn
    except ImportError:
        raise SystemExit("  FERMO: serve PyTorch.")

    if not torch.cuda.is_available():
        raise SystemExit(
            "  FERMO: nessuna GPU CUDA visibile.\n"
            "  Questa sezione misura memoria e tempo REALI su GPU.\n"
            "  Non stampo numeri stimati: sarebbero inventati.")

    if "RANK" not in os.environ:
        raise SystemExit(
            "  FERMO: va lanciato con torchrun, non con python3.\n"
            f"  torchrun --nproc_per_node={N} labs/lab_18_1_parallelismo.py --misura")

    import time
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    mondo = dist.get_world_size()
    torch.cuda.set_device(rank)

    H, L, B, S = 2048, 8, 4, 512
    modello = nn.Sequential(*[nn.Sequential(nn.Linear(H, 4 * H), nn.GELU(),
                                            nn.Linear(4 * H, H)) for _ in range(L)]).cuda()
    ddp = nn.parallel.DistributedDataParallel(modello, device_ids=[rank])
    opt = torch.optim.AdamW(ddp.parameters(), lr=1e-4)
    x = torch.randn(B, S, H, device="cuda")

    for _ in range(5):
        ddp(x).pow(2).mean().backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize(); dist.barrier()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(20):
        ddp(x).pow(2).mean().backward(); opt.step(); opt.zero_grad()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 20
    picco = torch.cuda.max_memory_allocated() / GB

    par = sum(p.numel() for p in modello.parameters())

    # Non confronto due totali: confronto VOCE PER VOCE, sulla configurazione
    # che questo processo ha davvero costruito. Due decomposizioni sbagliate
    # possono sommare allo stesso totale (bf16+bf16+8+4 = 4+4+8+0 = 16 byte
    # per parametro), e l'accordo del totale non convaliderebbe niente.
    # Un backward in piu' SENZA zero_grad(): il ciclo qui sopra finisce con
    # zero_grad(), che con set_to_none=True lascia p.grad a None. Senza
    # questo passo la voce 'gradienti' misurerebbe zero.
    ddp(x).pow(2).mean().backward()

    b_pesi = sum(p.numel() * p.element_size() for p in modello.parameters())
    b_grad = sum(p.grad.numel() * p.grad.element_size()
                 for p in modello.parameters() if p.grad is not None)
    b_stati, nomi_stati = 0, {}
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                b_stati += v.numel() * v.element_size()
                nomi_stati[k] = str(v.dtype)

    if rank == 0:
        print(f"  parametri del modello di prova : {par:,}")
        print(f"  ranghi                          : {mondo}")
        print(f"  memoria di picco PER RANK       : {picco:.2f} GB")
        print(f"  secondi per passo               : {dt:.4f}")

        print(f"\n  Le quattro voci di M06, MISURATE su questa configurazione:")
        dt_pesi = str(next(modello.parameters()).dtype)
        print(f"    pesi              {b_pesi / GB:8.2f} GB   dtype {dt_pesi},"
              f" {b_pesi / par:.1f} byte/par")
        print(f"    gradienti         {b_grad / GB:8.2f} GB  "
              f" {b_grad / par:.1f} byte/par")
        print(f"    stati optimizer   {b_stati / GB:8.2f} GB  "
              f" {b_stati / par:.1f} byte/par   {nomi_stati}")
        print(f"    copia master fp32     0.00 GB   AdamW nativo non la crea:"
              f" se il tuo stack la crea, e' fuori da questo conto")
        somma = (b_pesi + b_grad + b_stati) / GB
        print(f"    ----------------- --------")
        print(f"    somma statica     {somma:8.2f} GB")
        print(f"    scarto picco - somma statica  : {picco - somma:+.2f} GB"
              f"   (attivazioni, frammentazione, buffer NCCL)")
        print("""
  ATTENZIONE a come si legge questo confronto. Il modello qui e' fp32 con
  AdamW nativo: 4 + 4 + 8 = 16 byte per parametro. La tabella di M06 e'
  costruita su un altro stack - pesi bf16, gradienti bf16, momenti fp32 e
  master fp32 - che fa 2 + 2 + 8 + 4 = 16 byte per parametro. STESSO
  TOTALE, decomposizione diversa. Se avessi confrontato solo i totali
  avrei visto un accordo che non convalida nessuna delle quattro voci.
  E' il motivo per cui le stampo separate: la formula di M06 si verifica
  voce per voce, sullo stack che stai davvero usando.""")

        print(f"\n  Rilancia con --nproc_per_node=1 e confronta i secondi per passo:")
        print(f"    efficienza = t(1 GPU) / ({mondo} x t({mondo} GPU)) x 100")
        print(f"    100% = tempo esattamente dimezzato.")
        print("""
  Quello che manca al 100% e' OVERHEAD COMPLESSIVO DI SCALING, non
  'comunicazione'. Dentro ci sono, indistinguibili da questa misura:
  il traffico collettivo, le sincronizzazioni, il carico sbilanciato fra
  rank, il cambiamento del calcolo per rank, e il costo di lanciare i
  kernel. Per attribuirne una quota alla comunicazione serve una misura
  separata - un profiler, o un run con i collettivi disattivati.""")

        _mis = {
            "parametri": float(par),
            "ranghi": float(mondo),
            "picco_per_rank_GB": picco,
            "sec_per_passo": dt,
            "byte_per_par_pesi": b_pesi / par,
            "byte_per_par_gradienti": b_grad / par,
            "byte_per_par_stati": b_stati / par,
            "somma_statica_GB": somma,
        }
        registra(f"gpu{mondo}", _mis, torch)
        confronta(f"gpu{mondo}", _mis)
    dist.destroy_process_group()

if not args.misura and (args.registra or args.confronta):
    print("""
  --registra / --confronta riguardano la sezione 5, che serve --misura
  e va lanciata con torchrun su GPU. Senza, non c'e' nessuna misura da
  registrare o confrontare: le sezioni 1-4 sono aritmetica, e il loro
  risultato non dipende dall'ambiente.""")

print()
