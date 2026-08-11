#!/usr/bin/env python3
"""
Lab 12.1 — Valutazione: quando una differenza e' reale.   CPU.

    python3 corso/labs/lab_12_1_eval.py

Simulazione: due modelli con una differenza VERA e NOTA. Poi si misura
quanto spesso una valutazione la trova, e quanto spesso mente.
"""

import math
import random
import statistics


def sez(t):
    print(f"\n{'─' * 72}\n{t}\n{'─' * 72}")


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# Verita' di riferimento: A e' peggiore di B. Lo sappiamo per costruzione.
ABILITA_A, ABILITA_B = 0.85, 1.05          # in scala logit


def genera(n, seed, appaiato=True):
    """Ogni esempio ha una difficolta' propria: alcuni sono facili per
    entrambi i modelli, altri difficili per entrambi. E' questa correlazione
    che rende potente il confronto appaiato."""
    r = random.Random(seed)
    a, b = [], []
    for _ in range(n):
        d = r.gauss(0, 1.2)                              # difficolta' dell'item
        a.append(1 if r.random() < sigmoid(ABILITA_A - d) else 0)
        if not appaiato:
            d = r.gauss(0, 1.2)                          # item DIVERSO per B
        b.append(1 if r.random() < sigmoid(ABILITA_B - d) else 0)
    return a, b


# accuratezze vere, stimate su un campione enorme
va, vb = genera(200_000, 0)
VERA_A, VERA_B = statistics.mean(va), statistics.mean(vb)
VERO_DELTA = VERA_B - VERA_A

sez("1. La verita' di riferimento (che nella realta' non hai)")
print(f"  accuratezza vera modello A   {VERA_A:.4f}")
print(f"  accuratezza vera modello B   {VERA_B:.4f}")
print(f"  differenza vera              {VERO_DELTA:+.4f}   ({100 * VERO_DELTA:+.2f} punti)")
print("\n  B e' davvero migliore. Ora vediamo quanto spesso una valutazione")
print("  di dimensione realistica riesce a dirlo - e quanto spesso mente.")


# ----------------------------------------------------------------------
sez("2. Un solo run non decide niente")

print(f"  Dieci valutazioni indipendenti da 100 esempi ciascuna:\n")
print(f"  {'run':>4} {'acc A':>8} {'acc B':>8} {'diff':>9}  conclusione ingenua")
inversioni = 0
for s in range(1, 11):
    a, b = genera(100, s)
    ma, mb = statistics.mean(a), statistics.mean(b)
    d = mb - ma
    if d <= 0:
        inversioni += 1
    concl = "B meglio" if d > 0 else ("PARI" if d == 0 else "A meglio  <- SBAGLIATO")
    print(f"  {s:>4} {ma:>8.3f} {mb:>8.3f} {d:>+9.3f}  {concl}")

print(f"\n  In {inversioni}/10 run la conclusione ingenua e' sbagliata o nulla,")
print("  pur essendoci una differenza vera. Con 100 esempi, un singolo")
print("  numero non e' una misura: e' un campione rumoroso.")


# ----------------------------------------------------------------------
sez("3. Bootstrap: dare un intervallo invece di un numero")

def bootstrap_ic(a, b, appaiato=True, n_ric=2000, seed=7):
    """Intervallo di confidenza al 95% sulla differenza di accuratezza."""
    r = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_ric):
        if appaiato:
            idx = [r.randrange(n) for _ in range(n)]
            diffs.append(statistics.mean(b[i] for i in idx)
                         - statistics.mean(a[i] for i in idx))
        else:
            ia = [r.randrange(n) for _ in range(n)]
            ib = [r.randrange(n) for _ in range(n)]
            diffs.append(statistics.mean(b[i] for i in ib)
                         - statistics.mean(a[i] for i in ia))
    diffs.sort()
    return diffs[int(0.025 * n_ric)], diffs[int(0.975 * n_ric)]


a, b = genera(500, 42)
lo, hi = bootstrap_ic(a, b)
print(f"  500 esempi, confronto appaiato (stessi esempi ai due modelli)")
print(f"    differenza osservata  {statistics.mean(b) - statistics.mean(a):+.4f}")
print(f"    IC 95%                [{lo:+.4f}, {hi:+.4f}]")
print(f"    esclude lo zero?      {'SI -> differenza reale' if lo > 0 or hi < 0 else 'NO -> non concludi'}")
print(f"\n  differenza vera: {VERO_DELTA:+.4f}   dentro l'intervallo? "
      f"{lo <= VERO_DELTA <= hi}")


# ----------------------------------------------------------------------
sez("4. Appaiato contro non appaiato: la stessa spesa, meno rumore")

print("  Stesso numero di esempi. A sinistra i due modelli vedono gli STESSI")
print("  esempi; a destra esempi diversi.\n")
print(f"  {'n':>6}  {'APPAIATO: IC 95%':>26} {'ampiezza':>9}  "
      f"{'NON APPAIATO: IC 95%':>26} {'ampiezza':>9}")
for n in (100, 250, 500, 1000):
    a1, b1 = genera(n, 11, appaiato=True)
    lo1, hi1 = bootstrap_ic(a1, b1, appaiato=True)
    a2, b2 = genera(n, 11, appaiato=False)
    lo2, hi2 = bootstrap_ic(a2, b2, appaiato=False)
    print(f"  {n:>6}  [{lo1:+.4f}, {hi1:+.4f}]{'':>4} {hi1 - lo1:>9.4f}  "
          f"[{lo2:+.4f}, {hi2:+.4f}]{'':>4} {hi2 - lo2:>9.4f}")

print("""
  L'intervallo appaiato e' piu' stretto a parita' di esempi. Il motivo:
  buona parte del rumore viene dalla DIFFICOLTA' degli esempi, non dai
  modelli. Se entrambi vedono gli stessi esempi, quel rumore si cancella
  nella differenza.

  Conseguenza pratica: valutare i due modelli sullo stesso insieme non e'
  una comodita', e' quello che rende la misura utilizzabile.""")


# ----------------------------------------------------------------------
sez("5. Quanti esempi servono? Si misura, non si indovina")

print("  Per ogni dimensione: 200 esperimenti simulati. Conto in quanti")
print("  l'IC 95% appaiato esclude lo zero (cioe' la differenza viene")
print("  dichiarata reale) e in quanti punta nella direzione SBAGLIATA.\n")
print(f"  {'n':>6} {'trova la differenza':>21} {'direzione sbagliata':>21}")
for n in (50, 100, 250, 500, 1000, 2000):
    trovate = sbagliate = 0
    for s in range(200):
        a, b = genera(n, 1000 + s)
        lo, hi = bootstrap_ic(a, b, n_ric=400, seed=s)
        if lo > 0:
            trovate += 1
        elif hi < 0:
            sbagliate += 1
    print(f"  {n:>6} {100 * trovate / 200:>20.0f}% {100 * sbagliate / 200:>20.0f}%")

print(f"""
  LEGGI BENE LA COLONNA CENTRALE. Con {100 * VERO_DELTA:.1f} punti di differenza VERA:

    -  500 esempi la trovano nel 32% dei casi
    - 2000 esempi la trovano nel 74% dei casi

  Anche a 2000 esempi, un quarto delle volte concludi "non dimostrato"
  pur essendoci una differenza reale. Nessuna di queste dimensioni e'
  "affidabile": per superare il 90% ne servirebbero ancora di piu'.

  La colonna di destra: in QUESTI 200 esperimenti per dimensione non si e'
  mai verificata una conclusione invertita. Non e' una garanzia - un IC al
  95% PUO' escludere lo zero dal lato sbagliato, e con 0 casi su 200 il
  limite superiore al 95% su quel tasso resta circa 1.5% (regola del tre).

  La proprieta' vera e' piu' modesta e comunque preziosa: il tasso di
  conclusioni invertite e' LIMITATO e piccolo per costruzione, mentre nel
  confronto ingenuo della sezione 2 era 3 su 10. Non zero contro qualcosa:
  raro e controllato contro frequente e non controllato.

  Per una differenza piu' piccola servono molti piu' esempi: il costo
  cresce all'incirca col QUADRATO della finezza che vuoi risolvere.
  Dimezzare la differenza rilevabile costa circa quattro volte gli esempi.

  Quindi la domanda giusta non e' "quanti esempi bastano?" ma:
  "qual e' la differenza minima che mi interessa davvero?" - e da li'
  si ricava il resto.""")


# ----------------------------------------------------------------------
sez("6. LLM-as-judge: un bias piccolo ribalta il verdetto")

def giudica(n, bias_posizione, seed):
    """Il giudice confronta A e B a coppie. bias_posizione = probabilita'
    extra di preferire il candidato mostrato per PRIMO, a parita' di merito."""
    r = random.Random(seed)
    vinte_b = 0
    for _ in range(n):
        d = r.gauss(0, 1.2)
        qa = sigmoid(ABILITA_A - d)
        qb = sigmoid(ABILITA_B - d)
        p_b = qb / (qa + qb)                     # preferenza "onesta"
        primo_e_a = r.random() < 0.5             # ordine di presentazione
        p_b_eff = p_b - bias_posizione if primo_e_a else p_b + bias_posizione
        vinte_b += 1 if r.random() < p_b_eff else 0
    return vinte_b / n

print("  Il giudice preferisce B (il modello migliore) in questa frazione")
print("  di confronti, al variare del bias verso il candidato mostrato per primo:\n")
print(f"  {'bias':>8} {'ordine casuale':>16} {'A sempre primo':>16}")
for bias in (0.0, 0.05, 0.10, 0.20):
    casuale = giudica(2000, bias, 5)
    # se A e' SEMPRE primo, il bias agisce sempre nella stessa direzione
    r = random.Random(5)
    vb = 0
    for _ in range(2000):
        d = r.gauss(0, 1.2)
        qa, qb = sigmoid(ABILITA_A - d), sigmoid(ABILITA_B - d)
        p_b = qb / (qa + qb) - bias
        vb += 1 if r.random() < p_b else 0
    print(f"  {bias:>8.2f} {casuale:>16.3f} {vb / 2000:>16.3f}")

print("""
  Senza bias il giudice preferisce B nel 52% dei confronti: B e' meglio,
  di poco. L'effetto da misurare vale quindi 2 punti sopra il pareggio.

  Con ordine casuale il bias si compensa e la stima resta li'.
  Con ordine FISSO bastano 5 punti di bias per portare la stima a 0.448:
  sotto il pareggio. Il giudice ora dichiara migliore il modello PEGGIORE.

  La regola generale e' questa: quando il bias supera l'effetto che stai
  misurando, il verdetto lo decide il bias. E gli effetti che ti interessano
  sono quasi sempre piccoli.

  E' il difetto piu' comune di una valutazione con giudice automatico:
  non il giudice impreciso, ma il giudice con un bias COSTANTE che nessuno
  randomizza. Difesa: alterna l'ordine e verifica che invertendolo il
  verdetto non cambi. Se cambia, stai misurando l'ordine, non la qualita'.""")


# ----------------------------------------------------------------------
sez("7. Catastrophic forgetting: cosa misurare oltre al compito")

def esami(n, seed):
    """UNA lista di esempi per capacita': difficolta' + esito casuale.

    Generata una volta sola e usata per PRIMA e DOPO. Se prima e dopo
    usassero semi diversi cambierebbero anche gli esempi, e il delta
    mescolerebbe il cambiamento di abilita' con il rumore del campione -
    esattamente l'errore che la sezione 3 ha appena mostrato di evitare.
    """
    r = random.Random(seed)
    return [(r.gauss(0, 1.2), r.random()) for _ in range(n)]


def capacita(abilita, esempi):
    """Esiti per esempio sugli STESSI esempi. Cambia solo l'abilita'."""
    return [1 if u < sigmoid(abilita - d) else 0 for d, u in esempi]


print("  Un fine-tuning che migliora il compito bersaglio puo' peggiorare")
print("  capacita' generali che nessuno stava guardando.")
print("  Confronto APPAIATO: 1500 esempi per capacita', gli stessi prima e dopo.\n")
print(f"  {'':<22} {'prima':>8} {'dopo':>8} {'delta':>9}  {'IC 95% appaiato':>20}")
for i, (nome, prima_ab, dopo_ab) in enumerate([
    ("compito bersaglio", 0.20, 1.60),
    ("istruzioni generali", 1.10, 0.75),
    ("formato di uscita", 1.30, 1.28),
]):
    esempi = esami(1500, 100 + i)
    ep, ed = capacita(prima_ab, esempi), capacita(dopo_ab, esempi)
    p, d = statistics.mean(ep), statistics.mean(ed)
    lo, hi = bootstrap_ic(ep, ed, appaiato=True, n_ric=400)
    print(f"  {nome:<22} {p:>8.3f} {d:>8.3f} {d - p:>+9.3f}"
          f"  [{lo:>+7.3f}, {hi:>+7.3f}]")

print("""
  Il compito bersaglio migliora molto. Una capacita' generale peggiora, e
  l'intervallo appaiato NON contiene lo zero: la regressione non e' rumore.

  Se valuti solo il compito bersaglio, dichiari un successo e spedisci
  una regressione.

  GUARDA LA TERZA RIGA, che e' la piu' istruttiva. Il delta e' -0.004 -
  quattro decimi di punto - e anche il suo intervallo esclude lo zero.
  Il confronto appaiato e' cosi' sensibile che rileva un cambiamento
  che non ti interessa. 'Statisticamente distinguibile da zero' NON
  significa 'rilevante': sono due domande diverse, e la seconda te la
  devi porre tu, decidendo PRIMA quale differenza minima conta.
  Un intervallo stretto attorno a un effetto minuscolo e' il modo
  corretto di dire 'qui non e' successo niente che valga la pena'.

  Il controllo costa poco e va deciso PRIMA: fissa due o tre capacita'
  generali che il modello deve conservare, misurale prima del fine-tuning,
  rimisurale dopo SUGLI STESSI ESEMPI, e riporta l'intervallo appaiato
  della differenza. Senza la misura PRIMA non puoi dire niente dopo -
  e' il motivo per cui questo modulo viene prima di quelli sull'addestramento.""")

print()
