#!/usr/bin/env python3
"""
Lab 21.2 — Una diagnosi svolta per intero.   CPU.

    python3 corso/labs/lab_21_2_caso.py

Il lab 21.1 stampa il protocollo. Questo lo ESEGUE, su un fine-tuning
rotto in un punto che tu non conosci ancora e io si'.

La differenza non e' didattica: un metodo che non hai mai visto arrivare
in fondo e' un metodo di cui non sai se funziona. Qui la risposta e'
nota, cosi' puoi controllare che il protocollo la trovi - prima di
applicarlo a un caso in cui nessuno la conosce.

REGOLA DEL LAB: ogni criterio di decisione e' scritto PRIMA del
risultato. Nel codice li trovi come stringhe, stampate prima del numero
che devono giudicare. Se li leggessi dopo, non staresti falsificando
un'ipotesi: la staresti interpretando.
"""

import random
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F


def sez(t):
    print(f"\n{'═' * 74}\n{t}\n{'═' * 74}")


def sotto(t):
    print(f"\n{'─' * 74}\n{t}\n{'─' * 74}")


torch.manual_seed(0)
random.seed(0)

# ----------------------------------------------------------------------
# Il dominio: servizi e porte. Compito di pura memorizzazione, scelto
# apposta: se il modello sbaglia, non e' perche' il compito e' difficile.
NOMI = ["alfa", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "teta",
        "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi",
        "rho", "sigma", "tau", "ipsilon", "fi", "chi", "psi", "omega",
        "arco", "bravo", "carlo", "dado", "eco", "faro", "golf", "hotel",
        "india", "juliet", "kilo", "lima", "mike", "nord", "oscar", "papa",
        "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whisky",
        "raggio", "yankee", "zulu", "nebbia", "onda", "pietra", "quarzo",
        "rete", "sole", "torre", "urna", "vento", "zolla"]

rng_dati = random.Random(7)
PORTE = {n: f"{rng_dati.randrange(1000, 9999)}" for n in NOMI}

PROMPT = "D: porta di {}? R: "


def esempio(nome):
    return PROMPT.format(nome), PORTE[nome]


ALFA = "0123456789 ?:abcdefghijklmnopqrstuvwxyz"
STOI = {c: i for i, c in enumerate(ALFA)}
V = len(ALFA)


def cod(s):
    return [STOI[c] for c in s if c in STOI]


class Mini(nn.Module):
    def __init__(self, v, d=96):
        super().__init__()
        self.emb = nn.Embedding(v, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.testa = nn.Linear(d, v)

    def forward(self, x):
        h, _ = self.gru(self.emb(x))
        return self.testa(h)


def genera(m, prompt, quanti):
    """Decodifica greedy: nessun campionamento, nessun seme in gioco."""
    ids = cod(prompt)
    with torch.no_grad():
        for _ in range(quanti):
            logits = m(torch.tensor([ids]))[0, -1]
            ids.append(int(logits.argmax()))
    return "".join(ALFA[i] for i in ids[len(cod(prompt)):])


def esiti(m):
    """1 se la porta generata e' esatta, 0 altrimenti. Uno per servizio."""
    return [1 if genera(m, PROMPT.format(n), 4) == PORTE[n] else 0 for n in NOMI]


def ic_appaiato(a, b, n_ric=2000, seed=11):
    """IC 95% sulla differenza, APPAIATO: stessi servizi ai due modelli.
    E' il confronto di M12, sezione 4."""
    r = random.Random(seed)
    n = len(a)
    d = []
    for _ in range(n_ric):
        idx = [r.randrange(n) for _ in range(n)]
        d.append(statistics.mean(b[i] for i in idx) - statistics.mean(a[i] for i in idx))
    d.sort()
    return d[int(0.025 * n_ric)], d[int(0.975 * n_ric)]


# ======================================================================
sez("Il caso")

print(f"""  {len(NOMI)} servizi, ognuno con una porta di quattro cifre. Il compito e'
  memorizzazione pura: nessun ragionamento, nessuna generalizzazione.

  Il modello DI PARTENZA conosce gia' i fatti e risponde nel formato D/R.
  Non e' perfetto, ma funziona. Ci e' stato poi aggiunto un fine-tuning
  che avrebbe dovuto migliorarlo ancora.

  Il risultato e' deludente e nessuno sa perche' - che e' il caso di
  studio del corso. Questo lab non lo indovina: applica il protocollo
  di 21.1, ipotesi per ipotesi.""")


# ----------------------------------------------------------------------
# Costruzione degli artefatti del caso. Il "bug" e' qui dentro, e il
# lettore non ha bisogno di leggerlo: lo trovera' con la diagnosi.

def addestra_base(passi_grezzi=10 * len(NOMI), passi_qr=15 * len(NOMI)):
    """Il modello DI PARTENZA: conosce i fatti e sa gia' rispondere nel
    formato D/R. Non e' perfetto, ma funziona - e' lo stato da cui si
    parte quando si aggiunge un fine-tuning, non una tabula rasa.

    I passi sono PROPORZIONALI al numero di servizi, non costanti: con
    un dataset tre volte piu' grande e gli stessi passi il modello non
    memorizza abbastanza, il caso perde il sintomo e il fine-tuning
    sembra migliorare. Legare le due cose evita che il caso si rompa in
    silenzio la prossima volta che si cambia la taglia."""
    torch.manual_seed(1)
    m = Mini(V)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    r = random.Random(3)
    for _ in range(passi_grezzi):
        n = r.choice(NOMI)
        ids = torch.tensor([cod(f"il servizio {n} usa la porta {PORTE[n]}")])
        loss = F.cross_entropy(m(ids[:, :-1]).reshape(-1, V), ids[:, 1:].reshape(-1))
        loss.backward(); opt.step(); opt.zero_grad()
    for _ in range(passi_qr):
        n = r.choice(NOMI)
        p, risp = esempio(n)
        ids = torch.tensor([cod(p + risp)])
        perdite = F.cross_entropy(m(ids[:, :-1]).reshape(-1, V),
                                  ids[:, 1:].reshape(-1), reduction="none")
        msk = maschera(len(cod(p)), ids.shape[1], 0)
        loss = (perdite * msk).sum() / msk.sum()
        loss.backward(); opt.step(); opt.zero_grad()
    return m


def maschera(n_prompt, n_tot, sfasamento):
    """Posizioni supervisionate nello spazio dei BERSAGLI.

    Con ingressi ids[:-1] e bersagli ids[1:], il bersaglio in posizione i
    e' ids[i+1]. La risposta comincia a n_prompt nella sequenza intera,
    quindi il primo bersaglio della risposta sta in n_prompt - 1.
    """
    inizio = n_prompt - 1 + sfasamento
    m = torch.zeros(n_tot - 1)
    m[inizio:] = 1.0
    return m


def sft(base, sfasamento, passi=40 * len(NOMI)):
    """SFT con maschera sulla sola risposta. `sfasamento` e' il difetto.

    40 passi per servizio, non 15: con un fine-tuning breve il modello
    conserva quello che il base gli aveva insegnato sulla prima cifra,
    il sintomo resta debole e l'intervallo del passo zero si avvicina
    allo zero. Un margine sottile e' proprio cio' che si e' rotto
    cambiando piattaforma: qui il fine-tuning gira abbastanza da
    sovrascrivere davvero, come farebbe uno vero."""
    import copy
    torch.manual_seed(2)
    m = copy.deepcopy(base)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    r = random.Random(5)
    for _ in range(passi):
        n = r.choice(NOMI)
        p, risp = esempio(n)
        ids = torch.tensor([cod(p + risp)])
        logits = m(ids[:, :-1])
        perdite = F.cross_entropy(logits.reshape(-1, V), ids[:, 1:].reshape(-1),
                                  reduction="none")
        msk = maschera(len(cod(p)), ids.shape[1], sfasamento)
        loss = (perdite * msk).sum() / msk.sum()
        loss.backward(); opt.step(); opt.zero_grad()
    return m


print("\n  Addestro il modello base e il fine-tuning del caso. Qualche secondo...")
base = addestra_base()
SFASAMENTO_DEL_CASO = 1          # <- il difetto, che la diagnosi deve trovare
tunato = sft(base, SFASAMENTO_DEL_CASO)


# ======================================================================
sez("Passo zero — c'e' davvero un fenomeno da spiegare?")

print("""  CRITERIO, scritto prima di guardare i numeri:
    misuro l'accuratezza esatta di base e fine-tuned sugli STESSI servizi,
    e l'intervallo di confidenza al 95% della differenza appaiata.
    - se l'intervallo CONTIENE lo zero -> non ho evidenza sufficiente per
      attribuire una causa, e la prima cosa da fare e' la valutazione,
      non la diagnosi (M12);
    - se lo ESCLUDE ed e' negativo -> ho un fenomeno, e proseguo.
""")

e_base, e_tun = esiti(base), esiti(tunato)
a_base, a_tun = statistics.mean(e_base), statistics.mean(e_tun)
lo, hi = ic_appaiato(e_base, e_tun)

print(f"  {'modello':<26} {'accuratezza esatta':>20}")
print(f"  {'modello di partenza':<26} {a_base:>19.1%}")
print(f"  {'dopo il fine-tuning':<26} {a_tun:>19.1%}")
print(f"\n  differenza appaiata        {a_tun - a_base:>+19.1%}")
print(f"  IC 95% della differenza    [{lo:+.3f}, {hi:+.3f}]")

contiene_zero = lo <= 0 <= hi
print(f"  l'intervallo contiene lo zero?  {'SI' if contiene_zero else 'NO'}")
if contiene_zero:
    raise SystemExit(f"""
  VERDETTO DEL PASSO ZERO: MI FERMO QUI.

  L'intervallo [{lo:+.3f}, {hi:+.3f}] contiene lo zero: non ho evidenza
  sufficiente per attribuire una causa, e proseguire con le ipotesi
  sarebbe interpretare rumore.

  E questo lab si ferma DAVVERO, invece di dirlo e andare avanti. Se
  lo vedi, non e' una simulazione didattica: il caso e' sotto-dimensionato
  su questa piattaforma. La risposta corretta e' piu' esempi, non piu'
  ipotesi -> M12, sezione 5.""")

print(f"""
  VERDETTO DEL PASSO ZERO: c'e' un fenomeno, proseguo.

  Nota cosa NON ho fatto: non ho guardato la training loss. Scende in
  tutti gli scenari rotti costruiti nel corso, quindi non e' evidenza
  a favore di niente.""")


# ======================================================================
sez("Inventario degli artefatti — cosa e' verificabile e cosa no")

ARTEFATTI = [
    ("manifest dell'addestramento", True, "semi, lr, passi: sono nel codice"),
    ("sha256 del dataset", True, "il dizionario PORTE e' generato con seme fisso"),
    ("revisione del modello base", True, "e' l'oggetto `base` di questo processo"),
    ("template renderizzato salvato", True, "PROMPT e' una sola costante"),
    ("semi (pesi, ordine dati)", True, "torch.manual_seed e random.Random espliciti"),
    ("checkpoint valutato senza ambiguita'", True, "e' l'oggetto `tunato`"),
    ("configurazione di inferenza", True, "greedy, nessun campionamento"),
]
print(f"  {'artefatto':<42} {'c\'e\'?':>6}   perche'")
for nome, ok, nota in ARTEFATTI:
    print(f"  {nome:<42} {'SI' if ok else 'NO':>6}   {nota}")

print("""
  Tutti presenti: e' una situazione FORTUNATA, e va detto. In un caso
  reale ogni 'no' rimuove ipotesi dalla lista dei verificabili, e
  l'esito onesto diventa 'non determinabile' invece di una causa.
  Qui nessuna ipotesi e' esclusa in partenza.""")


# ======================================================================
sez("Le ipotesi, in ordine di costo — una alla volta")

esclusi = []

# ----------------------------------------------------------------------
sotto("Ipotesi 1 — disallineamento fra addestramento e inferenza (M20)")

print("""  CRITERIO, prima del test: valuto lo STESSO modello due volte, con lo
  stesso percorso di codice usato in addestramento e con quello usato in
  valutazione. Devono dare risultati identici.
    - identici     -> ipotesi ESCLUSA
    - diversi      -> mi fermo qui: le ipotesi successive non sarebbero
                      interpretabili.
""")

with torch.no_grad():
    n0 = NOMI[0]
    p0, r0 = esempio(n0)
    ids0 = torch.tensor([cod(p0 + r0)])
    via_addestramento = F.cross_entropy(
        tunato(ids0[:, :-1]).reshape(-1, V), ids0[:, 1:].reshape(-1)).item()
    tunato.eval()
    via_valutazione = F.cross_entropy(
        tunato(ids0[:, :-1]).reshape(-1, V), ids0[:, 1:].reshape(-1)).item()
    tunato.train()

d1 = abs(via_addestramento - via_valutazione)
print(f"  loss per il percorso di addestramento   {via_addestramento:.10f}")
print(f"  loss per il percorso di valutazione     {via_valutazione:.10f}")
print(f"  scarto                                  {d1:.3e}")
v1 = "ESCLUSA" if d1 == 0 else "CONFERMATA"
print(f"\n  VERDETTO: {v1}")
esclusi.append(("1. disallineamento train/inferenza", v1))

print("""
  ONESTA' SU QUESTO TEST: qui non poteva fallire, e va detto. Questo
  modellino non ha dropout ne' normalizzazioni con statistiche di batch,
  quindi train() ed eval() calcolano la STESSA funzione: lo scarto e'
  zero per costruzione, non perche' abbia verificato qualcosa.

  Un controllo che non puo' fallire non e' evidenza. Lo tengo perche' il
  protocollo lo prevede e perche' il codice che serve e' questo - ma su
  un modello vero, con dropout attivo o una scala di parity fra base,
  merged, converted e quantized (M20), lo stesso test ha un esito che
  non si conosce in anticipo. E' li' che diventa informativo.""")

# ----------------------------------------------------------------------
sotto("Ipotesi 2 — il fine-tuning non parte dal modello base (M02, M14)")

print("""  CRITERIO, prima del test: un SFT a zero passi deve dare esattamente la
  stessa loss del modello base, sullo stesso batch. E' il controllo di
  parita' di M02: esito binario, nessuna tolleranza da negoziare.
    - scarto esattamente nullo -> ESCLUSA
    - qualunque altro valore   -> CONFERMATA
""")

zero_passi = sft(base, SFASAMENTO_DEL_CASO, passi=0)
with torch.no_grad():
    l_base = F.cross_entropy(base(ids0[:, :-1]).reshape(-1, V),
                             ids0[:, 1:].reshape(-1)).item()
    l_zero = F.cross_entropy(zero_passi(ids0[:, :-1]).reshape(-1, V),
                             ids0[:, 1:].reshape(-1)).item()
d2 = abs(l_base - l_zero)
print(f"  loss del modello base                   {l_base:.10f}")
print(f"  loss del fine-tuning a zero passi       {l_zero:.10f}")
print(f"  scarto                                  {d2:.3e}")
v2 = "ESCLUSA" if d2 == 0 else "CONFERMATA"
print(f"\n  VERDETTO: {v2}")
esclusi.append(("2. init non neutro", v2))

# ----------------------------------------------------------------------
sotto("Ipotesi 3 — template diverso fra addestramento e valutazione (M10)")

print("""  CRITERIO, prima del test: confronto con repr() il testo renderizzato
  dai due percorsi, carattere per carattere.
    - identici  -> ESCLUSA
    - diversi   -> mi fermo qui
""")

t_addestramento = PROMPT.format("kilo")
t_valutazione = PROMPT.format("kilo")
print(f"  addestramento  {t_addestramento!r}")
print(f"  valutazione    {t_valutazione!r}")
uguali = t_addestramento == t_valutazione
print(f"  coincidono carattere per carattere?     {'SI' if uguali else 'NO'}")
v3 = "ESCLUSA" if uguali else "CONFERMATA"
print(f"\n  VERDETTO: {v3}")
esclusi.append(("3. template disallineato", v3))

print("""
  ONESTA' ANCHE QUI: in questo lab i due testi escono dalla stessa
  costante PROMPT, quindi coincidono per costruzione. Anche questo e' un
  controllo che non poteva fallire.

  Nel caso reale non e' cosi', ed e' la ragione per cui l'ipotesi sta
  cosi' in alto nella lista: il template di addestramento viene da una
  libreria, quello di inferenza dalla configurazione del tokenizer, e
  bastano uno spazio o un a-capo di differenza (M10). Il test da fare e'
  lo stesso - repr() dei due testi renderizzati - ma con due sorgenti
  DAVVERO indipendenti.""")

# ----------------------------------------------------------------------
sotto("Ipotesi 4 — maschera della loss sbagliata o sfasata (M10, M11)")

print("""  CRITERIO, prima del test: stampo (bersaglio, supervisionato) affiancati
  per un esempio reale, e guardo QUALI posizioni contribuiscono, non
  quante.
    - le posizioni attive sono ESATTAMENTE le cifre della risposta
      -> ESCLUSA
    - qualunque altro insieme -> CONFERMATA, ed e' la causa.

  Il conteggio da solo non basta: una maschera sfasata puo' avere il
  numero GIUSTO di posizioni attive e coprire quelle sbagliate.
""")

p4, r4 = esempio("kilo")
ids4 = torch.tensor([cod(p4 + r4)])
msk4 = maschera(len(cod(p4)), ids4.shape[1], SFASAMENTO_DEL_CASO)
attesa = maschera(len(cod(p4)), ids4.shape[1], 0)

bersagli = ids4[0, 1:]
print(f"  testo completo: {p4 + r4!r}")
print(f"  la risposta e': {r4!r}\n")
print(f"  {'pos':>4} {'bersaglio':>10} {'usata dal caso':>16} {'attesa':>9}")
for i in range(len(bersagli) - 8, len(bersagli)):
    print(f"  {i:>4} {ALFA[bersagli[i]]!r:>10} {int(msk4[i]):>16} {int(attesa[i]):>9}")

att_caso = [i for i in range(len(msk4)) if msk4[i] == 1]
att_attesa = [i for i in range(len(attesa)) if attesa[i] == 1]
tok_caso = "".join(ALFA[bersagli[i]] for i in att_caso)
tok_attesa = "".join(ALFA[bersagli[i]] for i in att_attesa)

print(f"\n  posizioni attive nel caso   {att_caso}  -> supervisiona {tok_caso!r}")
print(f"  posizioni attive attese     {att_attesa}  -> supervisiona {tok_attesa!r}")
print(f"  quante attive: caso {len(att_caso)}, attese {len(att_attesa)}")

v4 = "ESCLUSA" if tok_caso == tok_attesa else "CONFERMATA"
print(f"\n  VERDETTO: {v4}")
esclusi.append(("4. maschera della loss", v4))

if v4 == "CONFERMATA":
    print(f"""
  La maschera parte una posizione TROPPO AVANTI. Il modello non riceve
  mai segnale sulla prima cifra della porta: quella la deve indovinare.
  Con quattro cifre, sbagliarne una basta a fallire l'intera risposta.

  Nota che la training loss di questo run scende regolarmente: sta
  imparando benissimo le tre cifre che le vengono insegnate. La curva
  non poteva dirtelo.

  Le ipotesi 5-9 non le provo: il protocollo dice di fermarsi alla
  prima confermata e verificare la correzione. Se il sintomo non
  sparisse, tornerei alla lista.""")


# ======================================================================
sez("La correzione, e la verifica che il sintomo sparisca")

print("""  CRITERIO, prima del test: rifaccio lo stesso addestramento cambiando
  UNA SOLA COSA, la maschera. Tutto il resto identico: stesso seme,
  stessi dati, stessi passi, stesso learning rate.
    - il sintomo sparisce -> causa confermata
    - il sintomo resta    -> la causa era un'altra, torno alla lista

  Una correzione che cambia due cose insieme non verifica niente.
""")

corretto = sft(base, 0)
e_cor = esiti(corretto)
a_cor = statistics.mean(e_cor)
lo2, hi2 = ic_appaiato(e_tun, e_cor)

print(f"  {'modello':<30} {'accuratezza esatta':>20}")
print(f"  {'base':<30} {a_base:>19.1%}")
print(f"  {'SFT con maschera sfasata':<30} {a_tun:>19.1%}")
print(f"  {'SFT con maschera corretta':<30} {a_cor:>19.1%}")
print(f"\n  differenza corretto - rotto   {a_cor - a_tun:>+19.1%}")
print(f"  IC 95% della differenza       [{lo2:+.3f}, {hi2:+.3f}]")
sparito = lo2 > 0
print(f"  l'intervallo esclude lo zero?  {'SI' if sparito else 'NO'}")


# ======================================================================
sez("La scheda, compilata")

print(f"""  SINTOMO OSSERVATO
    che cosa:            il fine-tuning aggiuntivo PEGGIORA il modello
    su quale insieme:    {len(NOMI)} servizi, gli stessi per tutti i modelli
    metrica e valore:    accuratezza esatta {a_tun:.1%}
    stessa metrica base: {a_base:.1%}
    IC della differenza: [{lo:+.3f}, {hi:+.3f}]

  IPOTESI PROVATE, in ordine di costo""")
for nome, verdetto in esclusi:
    print(f"    {nome:<38} {verdetto}")

print(f"""
  VERDETTO: esito 1, CAUSA IDENTIFICATA.
    maschera della loss sfasata di una posizione: supervisionava
    {tok_caso!r} invece di {tok_attesa!r}.

  CORREZIONE E VERIFICA
    accuratezza dopo la correzione: {a_cor:.1%}
    IC della differenza rotto -> corretto: [{lo2:+.3f}, {hi2:+.3f}]
    il sintomo e' sparito: {'SI' if sparito else 'NO'}""")


# ======================================================================
sez("Cosa questo lab dimostra, e cosa no")

print(f"""  DIMOSTRA che il protocollo, applicato in ordine, arriva alla causa - e
  che le tre esclusioni che vengono prima non sono tempo sprecato: sono
  quello che rende interpretabile la quarta. Se avessi provato per prima
  la maschera e l'avessi corretta, non sapresti se il miglioramento
  viene da li' o da qualcos'altro che si e' mosso insieme.

  NON DIMOSTRA nulla sulla frequenza dei guasti nel mondo reale. Qui il
  difetto e' UNO, l'ho messo io, e tutti gli artefatti erano presenti.
  Un caso vero puo' avere piu' cause insieme, artefatti mancanti, e
  finire negli esiti 2 o 3 invece che nell'1.

  NON DIMOSTRA che le ipotesi 1 e 3 siano state verificate: in questo
  giocattolo passavano per costruzione, e l'ho scritto sotto ciascuna.
  Sono mostrate per il CODICE che serve a eseguirle, non come evidenza.

  NON DIMOSTRA che l'ordine delle ipotesi sia ottimo. E' l'ordine per
  costo del caso di studio: qui la maschera era la quarta e il conto
  ha funzionato, ma in un contesto con una valutazione gia' pronta e
  senza manifest l'ordine giusto sarebbe un altro. -> M21, sezione
  sull'ordine per costo.

  La cosa da portare via non e' 'la causa era la maschera': e' che il
  criterio va scritto prima, le ipotesi si escludono una alla volta, e
  la correzione si verifica cambiando UNA cosa sola.""")

print()
