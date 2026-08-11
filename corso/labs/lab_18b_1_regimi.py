#!/usr/bin/env python3
"""
Lab 18b.1 — Lo stesso modello, quattro regimi di hardware.   CPU.

    python3 corso/labs/lab_18b_1_regimi.py
    python3 corso/labs/lab_18b_1_regimi.py --params 8e9
    python3 corso/labs/lab_18b_1_regimi.py --mostra-tetti

Fino a qui il corso ha svolto un caso: un modello su piu' schede discrete
in un nodo. Questo lab prende la STESSA aritmetica e la applica ai quattro
regimi in cui vi trovate davvero, per far vedere una cosa sola:

    cambiando regime non cambiano i byte. Cambia QUALE VINCOLO MORDE.

Ed e' il vincolo che morde a decidere cosa puoi fare, non la dimensione
del modello.
"""

import argparse

GB = 1e9

ap = argparse.ArgumentParser()
ap.add_argument("--params", type=float, default=27e9)
ap.add_argument("--lora-frac", type=float, default=0.00491)
ap.add_argument("--mostra-tetti", action="store_true",
                help="stampa il modello piu' grande che entra, per regime")

# --- il tuo hardware, per generare le previsioni da falsificare -------
ap.add_argument("--regime", choices=["gpu-singola", "multi-gpu", "unificata",
                                     "multi-nodo", "noleggio"],
                help="il TUO regime: genera le previsioni specifiche da testare")
ap.add_argument("--accel", type=int, default=1, help="quanti acceleratori hai")
ap.add_argument("--gb", type=float, help="GB per acceleratore (o del pool, se unificata)")
ap.add_argument("--metodo", choices=["full", "lora", "qlora"], default="lora")
args = ap.parse_args()


def sez(t):
    print(f"\n{'═' * 74}\n{t}\n{'═' * 74}")


# ----------------------------------------------------------------------
# I REGIMI.
#
# Le capienze qui sotto sono ESEMPI, non specifiche: servono a rendere il
# conto concreto. Sostituiscile con quelle della TUA macchina - sono il
# solo dato che devi procurarti, e sta nella scheda tecnica.
#
# Quello che NON e' un esempio, e vale per tutti, e' la colonna
# 'partiziona': dice se in quel regime la memoria si possa spezzare fra
# piu' acceleratori. E' una proprieta' STRUTTURALE del regime, e da sola
# decide meta' delle risposte.
REGIMI = [
    # (nome, n_acceleratori, GB ciascuno, pool unico?, partiziona?)
    ("GPU singola",        1,  48,  False, False),
    ("piu' GPU in un nodo", 2, 141,  False, True),
    ("memoria unificata",  1, 128,  True,  False),
    ("cluster multi-nodo", 8,  80,  False, True),
]

METODI = [
    # (nome, byte/par per i pesi, quota addestrabile)
    ("full",  2,   1.0),
    ("LoRA",  2,   args.lora_frac),
    ("QLoRA", 0.5, args.lora_frac),
]

# byte per parametro addestrabile: gradienti bf16 + AdamW fp32 + master fp32
# Stessa decomposizione di M06, stesse assunzioni dichiarate li'.
BYTE_ADDESTRABILE = 2 + 8 + 4


def statico(P, byte_pesi, quota):
    """Byte statici totali. E' la formula di M06, invariata."""
    return P * byte_pesi + P * quota * BYTE_ADDESTRABILE


def entra(tot_byte, n, gb_ciascuno, partiziona):
    """GB che ogni acceleratore deve reggere, e se ci sta.

    La divisione per n si applica SOLO dove il regime la permette. E'
    qui che i quattro casi si separano davvero: non nei byte, ma nel
    fatto che quel /n esista o no.
    """
    per_acceleratore = (tot_byte / n) if partiziona else tot_byte
    return per_acceleratore / GB, per_acceleratore / GB <= gb_ciascuno


# ======================================================================
sez(f"1. Lo stesso modello da {args.params:.3g} parametri, nei quattro regimi")

print("  Il numero e' i GB che ogni acceleratore deve reggere, nel caso")
print("  migliore che il regime consente (partizionamento ideale dove esiste).\n")

intest = f"  {'regime':<22} {'accel.':>7} {'GB cad.':>8}"
for nome, _, _ in METODI:
    intest += f" {nome:>12}"
print(intest)

for r_nome, n, gb, pool, part in REGIMI:
    riga = f"  {r_nome:<22} {n:>7} {gb:>8}"
    for _m_nome, byte_pesi, quota in METODI:
        tot = statico(args.params, byte_pesi, quota)
        per, ok = entra(tot, n, gb, part)
        riga += f" {per:>8.1f} {'SI' if ok else 'no':>3}"
    print(riga)

print("""
  Leggi le colonne, non le righe. Lo STESSO metodo cambia esito da un
  regime all'altro, e non perche' il modello sia diverso: perche' in un
  regime quel /n esiste e nell'altro no.

  La riga 'memoria unificata' e' quella che smentisce l'intuizione piu'
  radicata: un solo acceleratore, ma una capienza che su schede discrete
  richiederebbe di partizionare. Il modello ci sta. Il che NON vuol dire
  che ci giri bene - vedi la sezione 3.""")


# ======================================================================
if args.mostra_tetti:
    sez("2. Il modello piu' grande che entra, per regime e metodo")

    print("  Invertendo la formula: dato il tetto di memoria, quanti")
    print("  parametri ci stanno?\n")
    print(f"  {'regime':<22}" + "".join(f" {n:>14}" for n, _, _ in METODI))
    for r_nome, n, gb, pool, part in REGIMI:
        riga = f"  {r_nome:<22}"
        capienza = gb * GB * (n if part else 1)
        for _m, byte_pesi, quota in METODI:
            byte_per_par = byte_pesi + quota * BYTE_ADDESTRABILE
            riga += f" {capienza / byte_per_par / 1e9:>12.1f}B"
        print(riga)

    print("""
  Questi tetti sono il LIMITE SUPERIORE dell'aritmetica statica: non
  contano le attivazioni, che dipendono da batch e sequenza (M07), ne'
  i buffer di comunicazione, ne' quello che il sistema operativo tiene
  per se'. Un modello all'80% del tetto tipicamente non ci sta.

  Servono a rispondere a una domanda sola, e la rispondono bene:
  'e' anche solo pensabile?'""")


# ======================================================================
sez("3. Cosa l'aritmetica NON decide, regime per regime")

DECIDE = [
    ("GPU singola",
     "la capienza, e basta: se non ci sta, non ci sta",
     "niente da partizionare. Le leve sono quantizzare la base (M14),\n"
     "     ridurre gli addestrabili, o le attivazioni (batch, sequenza,\n"
     "     gradient checkpointing di M07). Sono le stesse leve di sempre,\n"
     "     ma qui sono le UNICHE."),

    ("piu' GPU in un nodo",
     "la capienza per scheda, dopo aver scelto la strategia (M18)",
     "quanto costa la comunicazione: dipende dalla sovrapposizione con\n"
     "     il calcolo, e si misura. E' il caso svolto del corso."),

    ("memoria unificata",
     "poco: la capienza e' grande e quasi mai il vincolo",
     "LA BANDA. La stessa memoria che ti fa entrare il modello e'\n"
     "     molto piu' lenta della HBM di una scheda discreta, e il\n"
     "     tempo per passo ne dipende direttamente. Nessuna formula di\n"
     "     questo corso lo stima: si misura il tempo per passo a batch\n"
     "     crescente, come in M07, e si guarda dove smette di scalare."),

    ("cluster multi-nodo",
     "la capienza aggregata, che e' quasi sempre sufficiente",
     "la RETE fra nodi e la CODA. L'all-gather di ZeRO-3 attraversa la\n"
     "     rete invece dell'interconnessione interna: la riga 'ideale'\n"
     "     della sezione 1 e' li' piu' ottimistica che altrove. E il\n"
     "     tempo di attesa in coda non e' tempo di calcolo, ma e' tempo."),

    ("GPU a noleggio",
     "la stessa aritmetica del regime che stai affittando",
     "il TRASFERIMENTO e l'EFFIMERO. Il dataset e i checkpoint vanno\n"
     "     spostati a ogni run, e l'istanza puo' sparire a meta'. Il\n"
     "     manifest di M15 smette di essere buona pratica e diventa la\n"
     "     condizione per non perdere il lavoro."),
]

for nome, decide, non_decide in DECIDE:
    print(f"\n  {nome.upper()}")
    print(f"     l'aritmetica decide:  {decide}")
    print(f"     NON decide:           {non_decide}")

print("""

  IL PUNTO DI TUTTO IL MODULO. Il corso ti ha insegnato a contare i byte,
  ed e' una competenza che si trasferisce identica ai quattro regimi:
  la formula non cambia mai.

  Quello che cambia e' se contare i byte sia la domanda GIUSTA. Su GPU
  singola lo e' quasi sempre. Su memoria unificata quasi mai. Sapere in
  quale regime sei serve a non passare una settimana a ottimizzare la
  memoria quando il tuo problema era la banda - o viceversa.""")


# ======================================================================
sez("4. Il regime cambia anche cosa vuol dire 'riproducibile'")

print("""  Tre cose che in M15 erano una riga del manifest, e nei regimi
  diversi diventano il problema principale:

  SU CLUSTER          il nodo che ti tocca cambia fra un run e l'altro.
                      Registra quale hai avuto: due nodi 'identici'
                      possono avere driver o topologie diverse.

  SU GPU A NOLEGGIO   l'immagine e' effimera. Se non registri versione
                      di libreria e driver PRIMA, dopo non c'e' piu' la
                      macchina a cui chiederlo.

  SU MEMORIA UNIFICATA  il tempo per passo dipende da cosa altro sta
                      usando la memoria, CPU compresa. Un run 'piu'
                      lento' puo' non avere niente a che fare col
                      modello.

  In tutti e tre i casi la contromisura e' quella di M15, applicata
  prima invece che dopo.""")


# ======================================================================
sez("5. Le previsioni che questo modulo fa sul TUO hardware")

if not args.regime:
    print("""  Questa sezione genera previsioni FALSIFICABILI per la tua macchina.
  Serve dirle quale hai:

    python3 lab_18b_1_regimi.py --regime unificata --accel 1 --gb 128 \\
                                --params 8e9 --metodo lora

    --regime   gpu-singola | multi-gpu | unificata | multi-nodo | noleggio
    --accel    quanti acceleratori
    --gb       GB per acceleratore (o del pool, se unificata)

  PERCHE' e' fatta cosi'. Questo corso ti da' la TEORIA: l'aritmetica dei
  byte, e la struttura dei quattro regimi. Non ti da' misure sul tuo
  hardware, perche' non le ho - e inventarle sarebbe il difetto che il
  corso insegna a non commettere.

  La divisione del lavoro e' esplicita:
    io dichiaro cio' che va dichiarato, e cosa non ho verificato;
    tu esegui, misuri, e se la previsione e' sbagliata la correggi.

  Una teoria che non dice come si potrebbe smentirla non e' una teoria:
  e' un'opinione con dei numeri accanto.""")
else:
    if not args.gb:
        raise SystemExit("  Serve anche --gb: quanti GB ha ciascun acceleratore.")

    byte_pesi, quota = dict(
        full=(2, 1.0), lora=(2, args.lora_frac), qlora=(0.5, args.lora_frac)
    )[args.metodo]
    partiziona = args.regime in ("multi-gpu", "multi-nodo", "noleggio")
    tot = statico(args.params, byte_pesi, quota)
    per, ok = entra(tot, args.accel, args.gb, partiziona)

    print(f"  Il tuo caso: {args.params:.3g} par, {args.metodo}, "
          f"{args.accel} x {args.gb:.0f} GB, regime '{args.regime}'\n")

    # ---- previsione 1: capienza ----------------------------------
    print("  PREVISIONE 1 — LA CAPIENZA STATICA")
    print(f"    prevedo:     {per:.1f} GB per acceleratore, quindi "
          f"{'ci sta' if ok else 'NON ci sta'}")
    print("    come testi:  lancia il tuo addestramento e guarda se parte")
    print(f"    mi smentisci se: {'va in OOM lo stesso' if ok else 'parte senza OOM'}")
    if ok:
        print("""    ATTENZIONE: 'ci sta' qui vuol dire che ci stanno le voci STATICHE.
                 Le attivazioni non sono in questo conto (M07), e crescono
                 con batch e sequenza. Un OOM a batch grande non mi smentisce.""")

    # ---- previsione 2: lo statico sottostima ----------------------
    print("\n  PREVISIONE 2 — IL PICCO MISURATO SARA' PIU' ALTO")
    print(f"    prevedo:     picco > {per:.1f} GB, per attivazioni, buffer e")
    print("                 quello che il sistema tiene per se'")
    print("    come testi:  torch.cuda.max_memory_allocated() a fine passo")
    print("                 (il codice e' in lab_07_1, sezione 4)")
    print("    mi smentisci se: il picco e' UGUALE o MINORE del conto statico")

    # ---- previsione 3: dipende dal regime -------------------------
    PREV = {
        "unificata": (
            "il tempo per passo smettera' di migliorare aumentando il batch,\n"
            "                 PRIMA che il calcolo sia saturo: sei limitato dalla banda",
            "misura secondi/passo a batch 1, 2, 4, 8, 16 e cerca dove si appiattisce",
            "il tempo scala col batch fino a saturare la memoria: allora il\n"
            "                     vincolo era il calcolo, e la mia previsione e' sbagliata"),
        "multi-nodo": (
            "l'efficienza di scaling fra NODI sara' sensibilmente peggiore\n"
            "                 di quella fra GPU dello stesso nodo",
            "misura t(1 nodo) e t(N nodi), poi t(1 GPU) e t(k GPU) dentro un nodo,\n"
            "                 e confronta le due efficienze",
            "le due efficienze sono simili: allora la rete non e' il collo di\n"
            "                     bottiglia che prevedo, e la sezione 3 va corretta"),
        "multi-gpu": (
            "con DDP il picco per rank sara' circa uguale al conto statico\n"
            "                 intero, NON diviso per il numero di schede",
            "torchrun a 1 e a N GPU, confronta i picchi per rank (lab_18_1 sez. 5)",
            "il picco per rank cala aumentando le GPU senza che tu abbia\n"
            "                     attivato ZeRO: allora DDP fa qualcosa che non so"),
        "gpu-singola": (
            "nessuna strategia di M18 ti aiutera': le uniche leve sono\n"
            "                 quantizzare, ridurre gli addestrabili, o le attivazioni",
            "prova ad attivare ZeRO/FSDP con un solo acceleratore e guarda\n"
            "                 se il picco cambia",
            "il picco cala: allora esiste una leva che questo modulo non conosce"),
        "noleggio": (
            "il tempo che perdi in trasferimento dati sara' confrontabile\n"
            "                 col tempo di calcolo, per run corti",
            "cronometra separatamente: upload dataset, download checkpoint,\n"
            "                 e il tempo di addestramento vero",
            "il trasferimento e' trascurabile: allora il tuo dataset e' piccolo\n"
            "                     o la banda e' migliore di quanto assumo"),
    }
    prev, come, smentisci = PREV[args.regime]
    print(f"\n  PREVISIONE 3 — QUELLO CHE MORDE NEL REGIME '{args.regime}'")
    print(f"    prevedo:     {prev}")
    print(f"    come testi:  {come}")
    print(f"    mi smentisci se: {smentisci}")

    # ---- come rimandare indietro il risultato ---------------------
    print(f"""

  SE UNA DI QUESTE E' SBAGLIATA, IL CORSO E' SBAGLIATO. Non e' una
  formula di cortesia: sono affermazioni che non ho verificato su
  hardware come il tuo, e sono dichiarate come tali in AMBIENTI.md.

  Compila e rimanda:

    regime           {args.regime}
    acceleratori     {args.accel} x {args.gb:.0f} GB
    modello/metodo   {args.params:.3g} par, {args.metodo}
    torch / driver   ..........  (lab_00_1_ambiente.py te li stampa)

    previsione 1 (capienza)   confermata / smentita ..........
    previsione 2 (picco)      confermata / smentita ..........
    previsione 3 (regime)     confermata / smentita ..........

    numeri misurati  ..........

  Una previsione smentita vale piu' di dieci confermate: quella corregge
  il corso, le altre lo lasciano dov'e'.""")

print()
