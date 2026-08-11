#!/usr/bin/env python3
"""
Lab 16.1 — DPO: cosa ottimizza davvero, e cosa NON garantisce.   CPU.

    python3 corso/labs/lab_16_1_dpo.py

Addestra un modellino con DPO e misura tre grandezze insieme. Due salgono
come previsto. La terza fa una cosa che sorprende quasi tutti.
"""

import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def sez(t):
    print(f"\n{'─' * 72}\n{t}\n{'─' * 72}")


torch.manual_seed(0)
random.seed(0)

# ----------------------------------------------------------------------
sez("1. I dati: coppie, non risposte")

COPPIE = [
    ("porta di alfa?",  "2000", "non lo so"),
    ("porta di beta?",  "2137", "boh"),
    ("porta di gamma?", "2274", "forse 80"),
    ("porta di delta?", "2411", "non saprei"),
]
print(f"  {'prompt':<18} {'PREFERITA':<10} {'RIFIUTATA':<12}")
for p, w, l in COPPIE:
    print(f"  {p:<18} {w:<10} {l:<12}")

print("""
  In SFT ogni esempio ha UNA risposta giusta. Qui ogni esempio ha una
  coppia: una preferita e una rifiutata. Il segnale non e' 'produci
  questo' ma 'preferisci questo A QUELLO'.

  Serve quando la qualita' e' relativa e non sai scrivere la risposta
  ideale: tono, concisione, rifiutare quando non si sa. Se sai gia'
  scrivere la risposta giusta, l'SFT e' piu' diretto e costa meno.""")


# ----------------------------------------------------------------------
ALFA = "0123456789 ?abcdefghilmnoprstuvz"
STOI = {c: i for i, c in enumerate(ALFA)}
V = len(ALFA)


def cod(s):
    return [STOI[c] for c in s if c in STOI]


class Mini(nn.Module):
    def __init__(self, v, d=64):
        super().__init__()
        self.emb = nn.Embedding(v, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.testa = nn.Linear(d, v)

    def forward(self, x):
        h, _ = self.gru(self.emb(x))
        return self.testa(h)


def logp_sequenza(m, prompt, risposta):
    """log-probabilita' TOTALE della risposta, dato il prompt."""
    ids = torch.tensor([cod(prompt + risposta)])
    n_p = len(cod(prompt))
    logits = m(ids[:, :-1])
    lp = F.log_softmax(logits, -1)
    bersagli = ids[:, 1:]
    scelti = lp.gather(-1, bersagli.unsqueeze(-1)).squeeze(-1)
    return scelti[:, n_p - 1:].sum()


sez("2. La loss DPO, scritta per intero")

print("""    r(x,y) = log pi(y|x) - log pi_rif(y|x)        "reward implicito"

    L = -log sigma( beta * ( r(x, preferita) - r(x, rifiutata) ) )

  Tre cose da notare subito:

  1. Serve una POLITICA DI RIFERIMENTO congelata: la loss ha bisogno dei
     log-rapporti rispetto a pi_rif. Quanto costi in MEMORIA e' invece
     una scelta di implementazione: questo lab tiene due copie residenti
     (il caso peggiore, buono per il conto di M06), ma su un dataset
     fisso le log-probabilita' del riferimento si precalcolano una volta,
     e con gli adattatori si condivide la base spegnendo l'adattatore.
     Verifica cosa fa la TUA libreria. -> M06

  2. La loss dipende solo dalla DIFFERENZA fra i due reward. Attenzione
     pero' a non trarne la conclusione sbagliata: NON e' vero che 'nulla
     spinge la preferita verso l'alto'. Il gradiente rispetto a
     log pi(preferita) e' negativo, quindi la discesa la spinge SU.
     Quello che manca e' la GARANZIA sull'esito assoluto. -> sezione 4b,
     dove lo si misura invece di discuterlo.

  3. beta controlla quanto ci si puo' allontanare dal riferimento.
     Piccolo = liberta', grande = ancoraggio.""")


# ----------------------------------------------------------------------
sez("3. Addestriamo e misuriamo TRE grandezze insieme")

modello = Mini(V)
# pre-addestramento breve, cosi' il riferimento sa gia' qualcosa
opt0 = torch.optim.AdamW(modello.parameters(), lr=5e-3)
for _ in range(400):
    p, w, _ = random.choice(COPPIE)
    ids = torch.tensor([cod(p + w)])
    loss = F.cross_entropy(modello(ids[:, :-1]).reshape(-1, V), ids[:, 1:].reshape(-1))
    loss.backward(); opt0.step(); opt0.zero_grad()

import copy
riferimento = copy.deepcopy(modello)
for q in riferimento.parameters():
    q.requires_grad_(False)

# Copia dello stato PRIMA del DPO: serve alla sezione 4b, dove un singolo
# passo va fatto da qui. A margine 0 il peso del gradiente e' massimo
# (sigma(0) = 0.5); dopo 300 passi sarebbe quasi nullo e il passo non
# mostrerebbe niente.
modello_iniziale = copy.deepcopy(modello)

BETA = 0.1
opt = torch.optim.AdamW(modello.parameters(), lr=1e-3)


def misura():
    with torch.no_grad():
        lw = sum(logp_sequenza(modello, p, w) for p, w, _ in COPPIE) / len(COPPIE)
        ll = sum(logp_sequenza(modello, p, l) for p, _, l in COPPIE) / len(COPPIE)
        rw = sum(logp_sequenza(modello, p, w) - logp_sequenza(riferimento, p, w)
                 for p, w, _ in COPPIE) / len(COPPIE)
        rl = sum(logp_sequenza(modello, p, l) - logp_sequenza(riferimento, p, l)
                 for p, _, l in COPPIE) / len(COPPIE)
        acc = sum(1 for p, w, l in COPPIE
                  if (logp_sequenza(modello, p, w) - logp_sequenza(riferimento, p, w))
                  > (logp_sequenza(modello, p, l) - logp_sequenza(riferimento, p, l)))
    return lw.item(), ll.item(), (rw - rl).item(), acc / len(COPPIE)


print(f"  {'passo':>6} {'loss DPO':>10} {'margine':>9} {'accuratezza':>12}"
      f" {'logp PREFERITA':>16} {'logp rifiutata':>16}")
storia = []
for passo in range(301):
    if passo % 50 == 0:
        lw, ll, marg, acc = misura()
        storia.append((passo, lw, ll, marg))
    tot = 0.0
    for p, w, l in COPPIE:
        rw = logp_sequenza(modello, p, w) - logp_sequenza(riferimento, p, w).detach()
        rl = logp_sequenza(modello, p, l) - logp_sequenza(riferimento, p, l).detach()
        tot = tot - F.logsigmoid(BETA * (rw - rl))
    tot = tot / len(COPPIE)
    if passo % 50 == 0:
        print(f"  {passo:>6} {tot.item():>10.4f} {marg:>9.3f} {100 * acc:>11.0f}%"
              f" {lw:>16.3f} {ll:>16.3f}")
    tot.backward(); opt.step(); opt.zero_grad()


sez("4. Il risultato che sorprende")

p0, lw0, ll0, m0 = storia[0]
p1, lw1, ll1, m1 = storia[-1]
print(f"  {'grandezza':<28} {'inizio':>10} {'fine':>10} {'variazione':>12}")
for nome, a, b in [("margine (cio' che ottimizzi)", m0, m1),
                   ("logp della PREFERITA", lw0, lw1),
                   ("logp della rifiutata", ll0, ll1)]:
    print(f"  {nome:<28} {a:>10.3f} {b:>10.3f} {b - a:>+12.3f}")

verso = "SALITA" if lw1 > lw0 else "SCESA"
print(f"""
  Il margine sale: la loss DPO sta facendo il suo lavoro.
  Ma la log-probabilita' della risposta PREFERITA e' {verso}.

  La spiegazione facile - 'nulla spinge la preferita verso l'alto' -
  e' SBAGLIATA, e la sezione 4b lo misura. La formula una spinta verso
  l'alto ce l'ha; quello che non ha e' una garanzia sull'esito, perche'
  l'aggiornamento avviene nello spazio dei PARAMETRI CONDIVISI e le
  probabilita' sono NORMALIZZATE.

  COSA QUESTO LAB NON DIMOSTRA. Non dimostra che il modello 'generi
  peggio'. Qui si misura la log-probabilita' di otto stringhe precise:
  non si genera testo, non si valuta un insieme tenuto da parte, non si
  guardano lunghezze ne' terminatori. E una distribuzione normalizzata
  non puo' diventare 'meno propensa a produrre qualunque cosa': la massa
  si sposta, non sparisce. Dove sia finita, qui, non e' misurato.

  La conclusione onesta e' piu' stretta, e resta preoccupante: le quattro
  risposte preferite sono diventate meno probabili mentre TUTTE le
  metriche di DPO miglioravano. Per dire qualcosa sulla qualita' serve
  una misura indipendente. -> M12""")


sez("4b. Il gradiente diretto CONTRO l'aggiornamento parametrico")

print("""  La sezione 4 ha mostrato COSA succede. Questa mostra PERCHE', in due
  esperimenti che danno risposte opposte solo in apparenza.

  PARTE A - le due log-probabilita' come variabili INDIPENDENTI.
  Le tratto come due scalari slegati e chiedo ad autograd i segni.
""")

# I valori di partenza sono quelli veri, misurati sopra al passo 0.
lw_s = torch.tensor(float(lw0), requires_grad=True)
ll_s = torch.tensor(float(ll0), requires_grad=True)
# le logp del riferimento sono costanti: al passo 0 coincidono con quelle
# del modello, perche' il riferimento e' una copia fatta in quel momento
rif_w, rif_l = float(lw0), float(ll0)

z = BETA * ((lw_s - rif_w) - (ll_s - rif_l))
L_scalare = -F.logsigmoid(z)
L_scalare.backward()

peso = torch.sigmoid(-z).item()
print(f"  {'grandezza':<34} {'valore':>12}")
print(f"  {'z = beta * margine':<34} {z.item():>12.6f}")
print(f"  {'sigma(-z)  (il PESO del gradiente)':<34} {peso:>12.6f}")
print(f"  {'dL/dlogp(PREFERITA)':<34} {lw_s.grad.item():>12.6f}   <- NEGATIVO")
print(f"  {'dL/dlogp(rifiutata)':<34} {ll_s.grad.item():>12.6f}   <- POSITIVO")
print(f"  {'atteso: -beta*sigma(-z)':<34} {-BETA * peso:>12.6f}")
print(f"  {'atteso: +beta*sigma(-z)':<34} {BETA * peso:>12.6f}")

print("""
  Derivata NEGATIVA sulla preferita significa che la discesa del
  gradiente la spinge IN ALTO. Il segnale locale c'e', ed e' nel verso
  che ti aspetti. Chi dice 'nulla la spinge verso l'alto' sta leggendo
  male la formula.

  Nota il PESO: entrambe le derivate hanno per fattore sigma(-z). Vale
  0.5 quando il margine e' zero, e crolla appena il margine cresce -
  alla fine dell'addestramento di sopra vale circa 0.002, 250 volte
  meno. La spinta non cambia mai segno: SVANISCE.

  PARTE B - le stesse due quantita' sul modello VERO, dove i parametri
  sono condivisi. Un solo passo, dallo stato precedente al DPO.
""")


def logp_medie(m):
    with torch.no_grad():
        a = sum(logp_sequenza(m, p, w) for p, w, _ in COPPIE) / len(COPPIE)
        b = sum(logp_sequenza(m, p, l) for p, _, l in COPPIE) / len(COPPIE)
    return a.item(), b.item()


prima_w, prima_l = logp_medie(modello_iniziale)
opt_b = torch.optim.AdamW(modello_iniziale.parameters(), lr=1e-3)
tot = 0.0
for p, w, l in COPPIE:
    rw = logp_sequenza(modello_iniziale, p, w) - logp_sequenza(riferimento, p, w).detach()
    rl = logp_sequenza(modello_iniziale, p, l) - logp_sequenza(riferimento, p, l).detach()
    tot = tot - F.logsigmoid(BETA * (rw - rl))
(tot / len(COPPIE)).backward()
opt_b.step(); opt_b.zero_grad()
dopo_w, dopo_l = logp_medie(modello_iniziale)

print(f"  {'':<24} {'prima':>10} {'dopo 1 passo':>14} {'variazione':>12}")
print(f"  {'logp PREFERITA':<24} {prima_w:>10.4f} {dopo_w:>14.4f} {dopo_w - prima_w:>+12.4f}")
print(f"  {'logp rifiutata':<24} {prima_l:>10.4f} {dopo_l:>14.4f} {dopo_l - prima_l:>+12.4f}")
print(f"  {'margine':<24} {0.0:>10.4f} "
      f"{(dopo_w - prima_w) - (dopo_l - prima_l):>14.4f} "
      f"{(dopo_w - prima_w) - (dopo_l - prima_l):>+12.4f}")

segue = "SALITA" if dopo_w > prima_w else "SCESA"
print(f"""
  Al PRIMO passo la preferita e' {segue}: l'update segue il segnale
  locale della parte A. Chi si fermasse qui concluderebbe che tutto
  torna, e sbaglierebbe - perche' la sezione 4 ha misurato l'opposto
  sui 300 passi. Ecco le due misure una accanto all'altra:
""")
print(f"  {'orizzonte':<26} {'variazione di logp(preferita)':>30}")
print(f"  {'1 passo (qui sopra)':<26} {dopo_w - prima_w:>+30.4f}")
print(f"  {'300 passi (sezione 4)':<26} {lw1 - lw0:>+30.4f}")
print("\n  E la traiettoria completa, gia' misurata nella sezione 3:\n")
print(f"  {'passo':>7} {'logp PREFERITA':>16} {'logp rifiutata':>16} {'margine':>10}")
for pa, a, b, mg in storia:
    print(f"  {pa:>7} {a:>16.3f} {b:>16.3f} {mg:>10.3f}")

print("""
  IL PUNTO DELL'ESERCIZIO. Il segnale locale e' corretto e all'inizio
  viene pure seguito. Poi perde, per due motivi che si sommano:

  1. il PESO sigma(-z) crolla appena il margine cresce (parte A), quindi
     la spinta verso l'alto si spegne mentre l'ottimizzazione continua;
  2. i PARAMETRI SONO CONDIVISI: i passi che continuano ad abbassare la
     rifiutata muovono pesi che servono anche alla preferita, e il
     softmax ridistribuisce la massa fra TUTTE le continuazioni, non
     solo fra queste due.

  Quello che si osserva non e' il gradiente: e' la sua PROIEZIONE nello
  spazio dei parametri, sommata su molti passi con un peso che svanisce.

  Da portare via: 'la loss spinge X verso l'alto' e 'X sale dopo
  l'addestramento' sono due affermazioni diverse. La prima si legge
  nella formula, la seconda si misura - e su UN passo puo' confermarla
  mentre su trecento la smentisce. Vale per DPO e per qualunque loss su
  parametri condivisi.""")


sez("5. Cosa guardare davvero durante un DPO")

print("""  {:<34} {}""".format("metrica", "cosa ti dice"))
for m, d in [
    ("loss DPO", "che l'ottimizzazione procede. Nient'altro."),
    ("margine dei reward", "di quanto separa preferite e rifiutate."),
    ("accuratezza di preferenza", "in che frazione di coppie l'ordine e' giusto."),
    ("logp(preferita) ASSOLUTA", "SE STA CROLLANDO, ALLARME. -> sezione 4"),
    ("KL dal riferimento", "quanto ti sei allontanato. Se esplode, beta e' troppo basso."),
    ("eval held-out (M12)", "l'unica che dice qualcosa sulla QUALITA'."),
]:
    print(f"  {m:<34} {d}")

print("""
  Le prime tre salgono anche quando il modello peggiora. La quarta e' il
  campanello. La sesta e' l'unica che risponde alla domanda che ti fai.""")


sez("6. ORPO: togliere il modello di riferimento")

print("""  Il riferimento congelato costa memoria (un secondo modello) e un
  secondo forward a ogni passo. ORPO propone di eliminarlo, combinando
  in un'unica loss il termine di SFT sulla risposta preferita e un
  termine che penalizza il rapporto di probabilita' (odds ratio) fra
  preferita e rifiutata.

  Il vantaggio strutturale e' chiaro e verificabile con l'aritmetica
  di M06: meno memoria, meno calcolo.

  Cosa NON puoi dedurre da qui: se dia risultati migliori o peggiori
  sul tuo compito. Quello dipende dai dati e va misurato - e i numeri
  pubblicati sono ottenuti su altri compiti, con altri modelli.

  I dettagli di implementazione, i nomi degli iperparametri e i default
  cambiano fra librerie e versioni: verificali sul tuo stack.""")


sez("7. Quando DPO e' la scelta SBAGLIATA")

print("""  1. SAI SCRIVERE LA RISPOSTA GIUSTA.
     Allora fai SFT: e' piu' diretto, costa meno memoria (niente
     riferimento) e il segnale e' piu' forte.

  2. IL PROBLEMA E' CONOSCENZA, NON PREFERENZA.
     Il testo delle risposte preferite CONTIENE segnale fattuale, quindi
     non e' vero che le coppie non insegnino nulla. Ma e' un segnale
     indiretto - l'obiettivo premia la differenza fra due risposte, non
     la produzione di quella giusta - e a parita' di esempi l'SFT ne da'
     molto di piu'. Se devi installare fatti, confronta le due strade
     MISURANDO, non aspettarti che il DPO ci arrivi con piu' coppie.
     -> M13, M12

  3. LE COPPIE SONO RUMOROSE.
     Se le preferenze sono incoerenti fra loro, il margine puo' salire
     su rumore. Misura la concordanza fra annotatori prima di fidarti
     dei dati, esattamente come misureresti un giudice automatico (M12).

  4. NON HAI UN EVAL INDIPENDENTE.
     Senza M12 non hai modo di accorgerti del caso della sezione 4, e
     il margine ti dira' che va tutto bene fino al giorno del rilascio.""")

print()
