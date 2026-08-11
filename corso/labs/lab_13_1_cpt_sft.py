#!/usr/bin/env python3
"""
Lab 13.1 — Continued pretraining contro SFT: due obiettivi diversi.   CPU.

    python3 corso/labs/lab_13_1_cpt_sft.py

Addestra davvero tre modellini identici con tre regimi diversi e misura
cosa impara ciascuno. Gira in pochi secondi su CPU.
"""

import math
import random
import warnings

import torch

warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
import torch.nn as nn
import torch.nn.functional as F


def sez(t):
    print(f"\n{'─' * 72}\n{t}\n{'─' * 72}")


torch.manual_seed(0)
random.seed(0)

# ----------------------------------------------------------------------
# Dominio sintetico: 12 servizi, ciascuno con una porta. Conoscenza da
# assorbire. Poi un formato di domanda/risposta da imparare a seguire.
SERVIZI = ["alfa", "beta", "gamma", "delta", "echo", "foxtrot",
           "golf", "hotel", "india", "juliet", "kilo", "lima"]
PORTE = {s: 2000 + 137 * i for i, s in enumerate(SERVIZI)}

VISTI = SERVIZI[:8]            # compaiono nei fatti E nelle coppie Q/R
NASCOSTI = SERVIZI[8:]         # compaiono SOLO nel testo grezzo

FATTI = [f"il servizio {s} usa la porta {p}.\n" for s, p in PORTE.items()]
QA = [f"D: porta di {s}?\nR: {PORTE[s]}\n" for s in VISTI]

ALFABETO = sorted(set("".join(FATTI + QA)))
STOI = {c: i for i, c in enumerate(ALFABETO)}
ITOS = {i: c for c, i in STOI.items()}
V = len(ALFABETO)
IGNORA = -100


def cod(s):
    return [STOI[c] for c in s]


# ----------------------------------------------------------------------
sez("1. Quanto segnale d'addestramento dà ciascun regime")

tok_cpt = sum(len(cod(f)) for f in FATTI)
tot_sft = sum(len(cod(q)) for q in QA)
# in SFT si supervisiona solo la risposta: da "R: " in poi
sup_sft = sum(len(cod(q)) - q.index("R: ") for q in QA)

print(f"  {'regime':<26} {'token totali':>13} {'supervisionati':>15} {'quota':>7}")
print(f"  {'continued pretraining':<26} {tok_cpt:>13} {tok_cpt:>15} {100.0:>6.0f}%")
print(f"  {'SFT (solo risposta)':<26} {tot_sft:>13} {sup_sft:>15}"
      f" {100 * sup_sft / tot_sft:>6.0f}%")

print(f"""
  Stesso corpus, stessa GPU, e il continued pretraining estrae
  {tok_cpt / sup_sft:.1f} volte piu' segnale per esempio.

  Non e' un dettaglio contabile: se il tuo problema e' che il modello
  NON SA una cosa, l'SFT spreca la maggior parte dei token a farti
  vedere una domanda che non deve imparare a produrre.""")


# ----------------------------------------------------------------------
# Un transformer minuscolo, lo stesso per tutti e tre i regimi
class Mini(nn.Module):
    def __init__(self, v, d=96, teste=4, strati=3, ctx=64):
        super().__init__()
        self.ctx = ctx
        self.emb = nn.Embedding(v, d)
        self.pos = nn.Embedding(ctx, d)
        strato = nn.TransformerEncoderLayer(d, teste, 4 * d, dropout=0.0,
                                            batch_first=True, norm_first=True)
        self.corpo = nn.TransformerEncoder(strato, strati)
        self.testa = nn.Linear(d, v)

    def forward(self, x):
        s = x.shape[1]
        h = self.emb(x) + self.pos(torch.arange(s, device=x.device))
        m = nn.Transformer.generate_square_subsequent_mask(s)
        return self.testa(self.corpo(h, mask=m, is_causal=True))


def batch(testi, mascherato, n=16, ctx=64):
    X, Y = [], []
    for _ in range(n):
        t = random.choice(testi)
        ids = cod(t)[:ctx + 1]
        if len(ids) < 4:
            continue
        x = ids[:-1] + [0] * (ctx - len(ids) + 1)
        y = ids[1:] + [IGNORA] * (ctx - len(ids) + 1)
        if mascherato:
            k = t.index("R: ") + 3          # supervisiona solo dopo "R: "
            y = [IGNORA if i < k - 1 else v for i, v in enumerate(y)]
        X.append(x[:ctx]); Y.append(y[:ctx])
    return torch.tensor(X), torch.tensor(Y)


def addestra(testi, mascherato, passi, modello=None, lr=3e-3):
    m = modello or Mini(V)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for _ in range(passi):
        x, y = batch(testi, mascherato)
        loss = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1),
                               ignore_index=IGNORA)
        loss.backward(); opt.step(); opt.zero_grad()
    return m


@torch.no_grad()
def perplessita(m, testi):
    """Quanto bene il modello modella questi testi. Piu' basso = meglio."""
    tot, n = 0.0, 0
    for t in testi:
        ids = cod(t)[:65]
        x = torch.tensor([ids[:-1]]); y = torch.tensor([ids[1:]])
        l = F.cross_entropy(m(x).reshape(-1, V), y.reshape(-1))
        tot += l.item() * y.numel(); n += y.numel()
    return math.exp(tot / n)


@torch.no_grad()
def risponde(m, servizi, k=12):
    """Generazione golosa dopo 'R: '. Conta le risposte esatte."""
    ok = 0
    for s in servizi:
        p = f"D: porta di {s}?\nR: "
        ids = cod(p)
        for _ in range(k):
            x = torch.tensor([ids[-64:]])
            ids.append(int(m(x)[0, -1].argmax()))
        uscita = "".join(ITOS[i] for i in ids[len(cod(p)):])
        if uscita.split("\n")[0].strip() == str(PORTE[s]):
            ok += 1
    return ok / len(servizi)


# ----------------------------------------------------------------------
sez("2. Tre regimi, stesso modello, stesso budget di passi")

PASSI = 700
print(f"  Ogni modello: {PASSI} passi totali. Attendi qualche secondo...\n")

m_cpt = addestra(FATTI, mascherato=False, passi=PASSI)
m_sft = addestra(QA, mascherato=True, passi=PASSI)
m_mix = addestra(FATTI, mascherato=False, passi=PASSI // 2)
m_mix = addestra(QA, mascherato=True, passi=PASSI // 2, modello=m_mix)

# quarto regime: durante l'SFT si continuano a mescolare i fatti grezzi
m_rep = addestra(FATTI, mascherato=False, passi=PASSI // 2)
opt = torch.optim.AdamW(m_rep.parameters(), lr=3e-3)
for i in range(PASSI // 2):
    if i % 2 == 0:                      # meta' passi su Q/R mascherati
        x, y = batch(QA, True)
    else:                               # meta' su fatti grezzi (replay)
        x, y = batch(FATTI, False)
    loss = F.cross_entropy(m_rep(x).reshape(-1, V), y.reshape(-1), ignore_index=IGNORA)
    loss.backward(); opt.step(); opt.zero_grad()

print(f"  8 servizi compaiono sia nei fatti sia nelle coppie Q/R (VISTI).")
print(f"  4 servizi compaiono SOLO nel testo grezzo (NASCOSTI): l'SFT non li")
print(f"  ha mai visti in forma di domanda.\n")
print(f"  {'regime':<32} {'ppl fatti':>10} {'R: visti':>10} {'R: nascosti':>12}")
for nome, m in [("solo continued pretraining", m_cpt),
                ("solo SFT", m_sft),
                ("CPT poi SFT", m_mix),
                ("CPT poi SFT+replay dei fatti", m_rep)]:
    print(f"  {nome:<32} {perplessita(m, FATTI):>10.2f}"
          f" {100 * risponde(m, VISTI):>9.0f}% {100 * risponde(m, NASCOSTI):>11.0f}%")

print("""
  Tre colonne, tre domande diverse:

    ppl fatti    ha ASSORBITO il dominio? (conoscenza)
    R: visti     sa rispondere su cio' che ha visto nel formato Q/R?
    R: nascosti  sa rispondere su cio' che ha visto SOLO come testo grezzo?

  L'ultima colonna e' quella che conta: misura se la conoscenza acquisita
  col continued pretraining si TRASFERISCE al formato insegnato dall'SFT.""")

print(f"""
  DUE RISULTATI SCOMODI, entrambi da leggere per bene.

  (1) 'CPT poi SFT' DIMENTICA. La perplessita' passa da {perplessita(m_cpt, FATTI):.2f}
      dopo il CPT a {perplessita(m_mix, FATTI):.0f} dopo l'SFT: la seconda fase ha
      cancellato la conoscenza della prima. E' catastrophic forgetting,
      misurato qui, in un lab che gira in dieci secondi.

      Il replay lo risolve: mescolando i fatti grezzi durante l'SFT la
      perplessita' torna a {perplessita(m_rep, FATTI):.2f}, identica al CPT puro, senza
      perdere nulla sul formato (100% sui visti). Per questo mescolare
      dati di replay e' pratica standard, non superstizione.

  (2) MA sui NASCOSTI restano tutti a 0%, replay compreso. Il modello
      modella perfettamente 'il servizio kilo usa la porta 3096' e non
      sa rispondere a 'D: porta di kilo? R:'. La conoscenza c'e', in
      forma grezza, e NON e' accessibile nel formato richiesto.

      Questo demolisce l'assunzione piu' comoda del modulo: che CPT e
      SFT si compongano automaticamente. In questo esperimento non e'
      successo.

  LIMITI DI QUESTA MISURA, perche' non generalizzi troppo: modello
  minuscolo, 12 fatti, poche centinaia di passi, nessuna varieta' di
  formulazione. Un modello grande con molti modi di esprimere lo stesso
  fatto puo' comportarsi diversamente. Quello che questi numeri
  dimostrano NON e' 'la conoscenza non si trasferisce mai': e' che
  il trasferimento VA MISURATO, perche' puo' benissimo non avvenire.

  E suggeriscono la contromisura da provare per prima: se vuoi che una
  conoscenza sia accessibile in un formato, mostragliela ANCHE in quel
  formato. Non e' un dettaglio del dataset: e' il dataset.""")


# ----------------------------------------------------------------------
sez("3. La differenza, in una riga")

print("""  CONTINUED PRETRAINING
    obiettivo: prevedere OGNI token di testo grezzo
    dati: documenti, senza struttura di conversazione
    a cosa serve: far assorbire conoscenza o uno stile linguistico
    non insegna: a seguire istruzioni, a fermarsi, a rispettare un formato

  SFT
    obiettivo: prevedere solo i token della RISPOSTA
    dati: coppie istruzione/risposta, con un template
    a cosa serve: insegnare un comportamento e un formato
    non insegna (bene): conoscenza nuova - troppo pochi token supervisionati

  La scelta si fa da una domanda sola:
    "il modello NON SA una cosa, o NON SI COMPORTA come voglio?"

  Non sa            -> continued pretraining (o RAG, che qui non trattiamo)
  Non si comporta   -> SFT
  Entrambi          -> CPT e poi SFT CON REPLAY dei dati grezzi, e con
                       la conoscenza presentata ANCHE nel formato
                       bersaglio. La sequenza ingenua CPT->SFT nel lab
                       dimentica quello che aveva imparato, e anche
                       correggendola il trasferimento fra formati NON
                       e' automatico. Vedi i numeri della sezione 2.""")


# ----------------------------------------------------------------------
sez("4. L'errore che questo modulo esiste per prevenire")

print(f"""  Il modo tipico in cui si sbaglia: si vuole insegnare CONOSCENZA nuova
  e si costruisce un dataset di domande e risposte, perche' e' il formato
  che tutti usano.

  Ma in SFT la maggior parte dei token non contribuisce alla loss: qui
  solo il {100 * sup_sft / tot_sft:.0f}%. Per far assorbire un fatto al modello stai usando
  una frazione dei token che avresti con lo stesso testo in forma grezza.

  Sintomo osservabile: il modello impara benissimo il FORMATO - risponde
  sempre nella forma giusta - ma il CONTENUTO resta sbagliato o inventato.
  Se lo vedi, non e' un problema di iperparametri: e' il regime sbagliato.""")


# ----------------------------------------------------------------------
sez("5. Il costo nascosto del continued pretraining")

print("""  Il CPT su testo di dominio grezzo puo' degradare cio' che il modello
  gia' sapeva fare - e' il catastrophic forgetting di M12.

  Qui non lo misuro: i modellini di questo lab non sanno fare
  nient'altro, quindi non c'e' niente da perdere. Su un modello vero
  invece si misura, e va misurato PRIMA e DOPO, come in M12.

  Difese comuni, che vanno verificate non assunte:
    - mescolare dati generici a quelli di dominio
    - learning rate piu' basso di quello del pretraining originale
    - meno epoche di quante l'istinto suggerisce

  Nessuna di queste e' gratis, e nessuna e' garantita: si sceglie
  guardando la regressione misurata sulle capacita' generali.""")

print()
